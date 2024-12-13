import anndata
import h5py
import gzip
import json
import numpy as np
import pathlib
import tempfile
import traceback
import warnings

from cell_type_mapper.utils.utils import (
    choose_int_dtype,
    get_timestamp,
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.utils.h5_utils import (
    copy_h5_excluding_data)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad,
    write_df_to_h5ad,
    copy_layer_to_x,
    read_uns_from_h5ad,
    write_uns_to_h5ad,
    update_uns)

from cell_type_mapper.validation.utils import (
    is_x_integers,
    round_x_to_integers,
    get_minmax_x_from_h5ad,
    map_gene_ids_in_var)


def validate_h5ad(
        h5ad_path,
        gene_id_mapper,
        log=None,
        expected_max=20,
        tmp_dir=None,
        layer='X',
        round_to_int=True,
        output_dir=None,
        valid_h5ad_path=None):
    """
    Perform validation transformations on h5ad file.

    Parameters
    ----------
    h5ad_path:
        Path to the source h5ad file
    gene_id_mapper:
        the GeneIdMapper that will handle the mapping of gene
        identifiers into the expected form. If None, infer the
        mapper based on the species implied by the input gene IDs.
    log:
        Optional logger to log messages for CLI
    expected_max:
        If the max X value is less than this, emit a warning
        indicating that we think the normalization of the
        data is incorrect. If this is None, such test is done.
    tmp_dir:
       Dir where scratch data products can be written if needed
    layer:
        The layer in the source h5ad file where the cell by gene
        data will be retrieved. Regardless, it will be written to
        'X' in the validated file.
    round_to_int:
        If True, cast the cell by gene matrix to an integer.
        If False, leave it untouched relative to input.
    valid_h5ad_path:
        Where to write the output file
    output_dir:
        Dir where new h5ad file can be written (if necessary)


    Returns
    -------
    Path to validated h5ad (if relevant).
    Returns None if no action was taken

    Also a boolean indicating if there were warnings raised
    by the validation process.

    Notes
    -----
    If valid_h5ad_path is specified, this is where the validated file
    will be written. If output_dir is specified, a new file will
    be written with a name like

    output_file/input_file_name_VALIDATED_{timestamp}.h5ad

    Both valid_h5ad_path and output_dir cannot be non-None
    """
    tmp_dir = tempfile.mkdtemp(dir=tmp_dir)
    try:
        result = _validate_h5ad(
            h5ad_path=h5ad_path,
            gene_id_mapper=gene_id_mapper,
            log=log,
            expected_max=expected_max,
            tmp_dir=tmp_dir,
            layer=layer,
            round_to_int=round_to_int,
            output_dir=output_dir,
            valid_h5ad_path=valid_h5ad_path)
    finally:
        _clean_up(tmp_dir)

    return result


def _validate_h5ad(
        h5ad_path,
        gene_id_mapper,
        log=None,
        expected_max=20,
        tmp_dir=None,
        layer='X',
        round_to_int=True,
        output_dir=None,
        valid_h5ad_path=None):

    (original_h5ad_path,
     write_to_new_path) = _convert_csv_to_h5ad(
         src_path=h5ad_path,
         log=log
     )

    has_warnings = write_to_new_path

    tmp_h5ad_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir,
            prefix=original_h5ad_path.name,
            suffix='.h5ad')
        )

    if valid_h5ad_path is not None and output_dir is not None:
        raise RuntimeError(
            "Cannot specify both valid_h5ad_path and output_dir; "
            "only specify one")

    if valid_h5ad_path is None and output_dir is None:
        raise RuntimeError(
            "Must specify one of either valid_h5ad_path or output_dir")

    h5ad_name = original_h5ad_path.name.replace(
                    original_h5ad_path.suffix, '')

    if valid_h5ad_path is None:
        output_dir = pathlib.Path(output_dir)
        timestamp = get_timestamp().replace('-', '')
        new_h5ad_path = output_dir / f'{h5ad_name}_VALIDATED_{timestamp}.h5ad'
    else:
        new_h5ad_path = pathlib.Path(valid_h5ad_path)

    # check that file can even be open
    try:
        with h5py.File(original_h5ad_path, 'r') as _:
            pass
    except Exception:
        error_msg = f"\n{traceback.format_exc()}\n"
        error_msg += (
            "This h5ad file is corrupted such that it could not "
            "even be opened with h5py. See above for the specific "
            f"error message raised by h5py {original_h5ad_path}."
        )
        if log is None:
            raise RuntimeError(error_msg)
        else:
            log.error(error_msg)

    # check that cell names are not repeated
    obs_original = read_df_from_h5ad(
        h5ad_path=original_h5ad_path,
        df_name='obs')

    cell_id_census = dict()
    for cell_id in obs_original.index.values:
        if cell_id not in cell_id_census:
            cell_id_census[cell_id] = 1
        else:
            cell_id_census[cell_id] += 1
    msg = ''
    for cell_id in cell_id_census:
        if cell_id_census[cell_id] > 1:
            msg += f"'{cell_id}' occurred {cell_id_census[cell_id]} times\n"
    if len(msg) > 0:
        msg = (
            "Cell IDs need to be unique. The following cell IDs were "
            "repeated in your obs dataframe:\n"
            f"{msg}"
        )
        raise RuntimeError(msg)

    # check that gene names are not repeated
    var_original = read_df_from_h5ad(
            h5ad_path=original_h5ad_path,
            df_name='var')

    _check_input_gene_names(
        var_df=var_original,
        log=log)

    cast_to_int = False
    if round_to_int:
        is_int = is_x_integers(
            h5ad_path=original_h5ad_path,
            layer=layer)
        if not is_int:
            cast_to_int = True

    (mapped_var,
     n_unmapped_genes) = map_gene_ids_in_var(
        var_df=var_original,
        gene_id_mapper=gene_id_mapper,
        log=log)

    if expected_max is not None or cast_to_int:
        x_minmax = get_minmax_x_from_h5ad(
            h5ad_path=original_h5ad_path,
            layer=layer)

        if expected_max is not None and x_minmax[1] < expected_max:
            msg = "VALIDATION: CDM expects raw counts data. The maximum value "
            msg += f"of the X matrix in ../{original_h5ad_path.name} is "
            msg += f"{x_minmax[1]}, indicating that this may be "
            msg += "log normalized data. CDM will proceed, but results "
            msg += "may be suspect."
            if log is not None:
                log.warn(msg)
            else:
                warnings.warn(msg)
            has_warnings = True

    if layer != 'X' or mapped_var is not None or cast_to_int:
        # Copy data into new file, moving cell by gene data from
        # layer to X
        copy_layer_to_x(
            original_h5ad_path=original_h5ad_path,
            new_h5ad_path=tmp_h5ad_path,
            layer=layer)

        write_to_new_path = True

        if log is not None:
            msg = (f"VALIDATION: copied ../{original_h5ad_path.name} "
                   f"to ../{new_h5ad_path.name}")
            log.info(msg)

        if mapped_var is not None:
            if log is not None:
                msg = "VALIDATION: modifying var dataframe of "
                msg += f"../{original_h5ad_path.name} to include "
                msg += "proper gene identifiers"
                log.info(msg)

            # check if genes are repeated in the mapped var DataFrame
            if len(mapped_var) != len(set(mapped_var.index.values)):
                repeats = dict()
                for orig, mapped in zip(var_original.index.values,
                                        mapped_var.index.values):
                    if mapped not in repeats:
                        repeats[mapped] = []
                    repeats[mapped].append(orig)
                for mapped in mapped_var.index.values:
                    if len(repeats[mapped]) == 1:
                        repeats.pop(mapped)
                error_msg = (
                    "The following gene symbols in your h5ad file "
                    "mapped to identical gene identifiers in the "
                    "validated h5ad file. The validated h5ad file must "
                    "contain unique gene identifiers.\n"
                )
                for mapped in repeats:
                    error_msg += (
                        f"{json.dumps(repeats[mapped])} "
                        "all mapped to "
                        f"{mapped}\n"
                    )
                if log is not None:
                    log.error(error_msg)
                else:
                    raise RuntimeError(error_msg)

            write_df_to_h5ad(
                h5ad_path=tmp_h5ad_path,
                df_name='var',
                df_value=mapped_var)

            gene_mapping = {
                orig: new
                for orig, new in zip(var_original.index.values,
                                     mapped_var.index.values)
                if orig != new}

            uns = read_uns_from_h5ad(tmp_h5ad_path)
            uns['AIBS_CDM_gene_mapping'] = gene_mapping
            write_uns_to_h5ad(tmp_h5ad_path, uns)
            has_warnings = True

        if cast_to_int:
            output_dtype = choose_int_dtype(x_minmax)

            msg = "VALIDATION: rounding X matrix of "
            msg += f"{original_h5ad_path.name} to integer values"
            if log is not None:
                log.warn(msg)
            else:
                warnings.warn(msg)
            round_x_to_integers(
                h5ad_path=tmp_h5ad_path,
                tmp_dir=tmp_dir,
                output_dtype=output_dtype)
            has_warnings = True

    if write_to_new_path:
        n_genes = len(var_original)
        update_uns(
            tmp_h5ad_path,
            {'AIBS_CDM_n_mapped_genes': n_genes-n_unmapped_genes})

        copy_h5_excluding_data(
            src_path=tmp_h5ad_path,
            dst_path=new_h5ad_path,
            excluded_groups=None,
            excluded_datasets=None)
    else:
        if new_h5ad_path.exists():
            new_h5ad_path.unlink()

    if log is not None:
        msg = f"DONE VALIDATING ../{original_h5ad_path.name}; "
        if write_to_new_path:
            msg += f"reformatted file written to ../{new_h5ad_path.name}\n"
        else:
            msg += "no changes required"
        log.info(msg)

    if write_to_new_path:
        output_path = new_h5ad_path
    else:
        output_path = None

    return output_path, has_warnings


def _check_input_gene_names(
        var_df,
        log):
    """
    Check that the var dataframe in var_df has unique
    gene names and that none of the gene names are blank

    Parameters
    ----------
    var_df:
        a pandas dataframe. The 'var' element in an h5ad file
    log:
       Optional CommandLog. If None, errors will be thrown straight
       to stderr

    Returns
    -------
    None
        Just throws errors if anything is amiss
    """

    error_msg = ''

    try:
        unq_genes, unq_gene_count = np.unique(
            var_df.index.values,
            return_counts=True)
    except TypeError:
        error_msg = f"{traceback.format_exc()}\n"
        error_msg += (
            "====hint====\n"
            "We expect the index of var in your h5ad "
            "file to be a list of unique gene names that are "
            "strings (or string-like). Your h5ad has var.index.values "
            "of type:\n"
            f"{var_df.index.values.to_numpy().dtype}"
        )
        raise TypeError(error_msg)

    repeated = np.where(unq_gene_count > 1)[0]
    for idx in repeated:
        error_msg += (
            f"gene name '{unq_genes[idx]}' "
            f"occurs {unq_gene_count[idx]} times "
            "in var; gene names must be unique\n")

    gene_names = set(var_df.index.values)
    for bad_val in ('',):
        if bad_val in gene_names:
            error_msg += (f"gene name '{bad_val}' is invalid; "
                          "if you cannot remove this column, just "
                          "change the gene name to a (unique) "
                          "nonsense string.")
    if len(error_msg) > 0:
        if log is not None:
            log.error(error_msg)
        else:
            raise RuntimeError(error_msg)


def _convert_csv_to_h5ad(
        src_path,
        log):
    """
    Convert CSV file to h5ad file (if necessary)

    Parameters
    ----------
    src_path:
        Path to the src file
    log:
        Optional logger to log messages for CLI

    Returns
    -------
    h5ad_path:
        Path to the h5ad file (this will be src_path
        if src_path ends in '.h5ad')
    was_converted:
        Boolean indicating if the file was converted to an
        h5ad file

    Notes
    -----
    This function has to determine whether or not to set first_column_names
    to True when reading the CSV with anndata (i.e. it has to determine if
    the first column in the file is a list of cell labels, or is just another
    gene).

    To make this determination, it applies the following test:

    - If the first entry in the header column is '', then
       first_column_names=True

    - If the first entry in the first data row cannot be converted to a float,
      (i.e. if it is just a string), then first_column_names=True

    - Otherwise, we assume the file is purely made up of gene expression data
      and first_column_names=False

    We believe this will be save because, even if th first column is supposed
    to be cell labels and those labels are numerical, the first column header
    (which must, in this case, not be blank) should not map to any gene
    identifiers.
    """
    src_path = pathlib.Path(src_path)
    src_name = src_path.name

    if not src_name.endswith('.csv') and not src_name.endswith('.csv.gz'):
        return (src_path, False)

    if src_name.endswith('.csv.gz'):
        src_suffix = '.csv.gz'
    elif src_name.endswith('.csv'):
        src_suffix = '.csv'

    dst_path = src_path.parent / src_name.replace(src_suffix, '.h5ad')
    if dst_path.exists():
        dst_path = mkstemp_clean(
            dir=src_path.parent,
            prefix=src_name.replace(src_suffix, '_'),
            suffix='.h5ad'
        )

    warning_msg = (
        "Input data is in CSV format; converting to h5ad file at "
        f"{dst_path}"
    )

    if log is None:
        warnings.warn(warning_msg)
    else:
        log.warn(warning_msg)

    if src_suffix == '.csv':
        open_fn = open
        mode = 'r'
        is_gzip = False
    else:
        open_fn = gzip.open
        mode = 'rb'
        is_gzip = True

    with open_fn(src_path, mode) as src:
        header = src.readline()
        first_row = src.readline()

    if is_gzip:
        header = header.decode()
        first_row = first_row.decode()

    header_params = header.split(',')
    first_row_params = first_row.split(',')

    first_column_names = False
    if header_params[0] == '':
        first_column_names = True
    else:
        try:
            float(first_row_params[0])
        except ValueError:
            first_column_names = True

    adata = anndata.read_csv(
        src_path,
        first_column_names=first_column_names)

    adata.write_h5ad(dst_path)

    return (dst_path, True)
