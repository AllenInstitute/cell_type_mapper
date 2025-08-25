import h5py
import numpy as np
import pathlib
import tempfile
import traceback
import warnings

import mmc_gene_mapper.mapper.species_detection as species_detection

from cell_type_mapper.utils.utils import (
    choose_int_dtype,
    get_timestamp,
    mkstemp_clean,
    _clean_up)

from cell_type_mapper.utils.h5_utils import (
    copy_h5_excluding_data)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad,
    copy_layer_to_x,
    transpose_h5ad_file)

from cell_type_mapper.validation.utils import (
    is_x_integers,
    round_x_to_integers,
    get_minmax_x_from_h5ad,
    create_uniquely_indexed_df)

from cell_type_mapper.validation.csv_utils import (
    convert_csv_to_h5ad
)


def validate_h5ad(
        h5ad_path,
        gene_mapper_db_path=None,
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
    gene_mapper_db_path:
        Path to sqlite db file that we will consult to determine
        if the index of var contains valid genes or not
        (produced by mmc_gene_mapper library)
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
            gene_mapper_db_path=gene_mapper_db_path,
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
        gene_mapper_db_path=None,
        log=None,
        expected_max=20,
        tmp_dir=None,
        layer='X',
        round_to_int=True,
        output_dir=None,
        valid_h5ad_path=None):

    input_h5ad_path = pathlib.Path(h5ad_path)

    if valid_h5ad_path is not None and output_dir is not None:
        raise RuntimeError(
            "Cannot specify both valid_h5ad_path and output_dir; "
            "only specify one")

    if valid_h5ad_path is None and output_dir is None:
        raise RuntimeError(
            "Must specify one of either valid_h5ad_path or output_dir")

    (active_h5ad_path,
     write_to_new_path) = convert_csv_to_h5ad(
         src_path=h5ad_path,
         log=log
     )

    has_warnings = write_to_new_path

    _check_h5ad_integrity(
        active_h5ad_path=active_h5ad_path,
        layer=layer,
        log=log
    )

    (active_h5ad_path,
     was_transposed) = _transpose_file_if_necessary(
         src_path=active_h5ad_path,
         gene_mapper_db_path=gene_mapper_db_path,
         layer=layer,
         tmp_dir=tmp_dir,
         log=log
    )

    if was_transposed:
        has_warnings = True
        write_to_new_path = True

    active_h5ad_path = pathlib.Path(active_h5ad_path)

    if valid_h5ad_path is None:
        new_h5ad_path = _create_valid_h5ad_path(
            output_dir=output_dir,
            active_h5ad_path=active_h5ad_path
        )
    else:
        new_h5ad_path = pathlib.Path(valid_h5ad_path)

    new_obs = _create_new_obs(
        active_h5ad_path=active_h5ad_path,
        log=log
    )

    if new_obs is not None:
        has_warnings = True

    var_original = _check_var(
         active_h5ad_path=active_h5ad_path,
         log=log)

    (cast_to_int,
     val_warnings,
     output_dtype) = _check_values(
         active_h5ad_path=active_h5ad_path,
         layer=layer,
         round_to_int=round_to_int,
         expected_max=expected_max,
         log=log)

    if val_warnings:
        has_warnings = True

    if (layer != 'X' or new_obs is not None or cast_to_int):

        write_to_new_path = True

        (active_h5ad_path,
         tmp_warnings) = _write_to_tmp_file(
             active_h5ad_path=active_h5ad_path,
             layer=layer,
             var=var_original,
             new_obs=new_obs,
             cast_to_int=cast_to_int,
             output_dtype=output_dtype,
             log=log,
             tmp_dir=tmp_dir)

        if tmp_warnings:
            has_warnings = True

    if write_to_new_path:
        copy_h5_excluding_data(
            src_path=active_h5ad_path,
            dst_path=new_h5ad_path,
            excluded_groups=None,
            excluded_datasets=None)
    else:
        if new_h5ad_path.exists():
            new_h5ad_path.unlink()

    if log is not None:
        msg = f"DONE VALIDATING ../{input_h5ad_path.name}; "
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


def _create_valid_h5ad_path(
        output_dir,
        active_h5ad_path):
    """
    Create and return path for valid h5ad path

    Parameters
    ----------
    output_dir:
        path to the directory where the valid h5ad file
        will be written
    active_h5ad_path:
        path to the h5ad path being validated

    Returns
    -------
    Path where we can write valid h5ad file
    """
    output_dir = pathlib.Path(output_dir)
    h5ad_name = active_h5ad_path.name.replace(
                    active_h5ad_path.suffix, '')
    timestamp = get_timestamp().replace('-', '')
    new_h5ad_path = output_dir / f'{h5ad_name}_VALIDATED_{timestamp}.h5ad'
    if new_h5ad_path.exists():
        salt = 0
        suffix = new_h5ad_path.suffix
        base_name = new_h5ad_path.name
        while True:
            new_h5ad_path = (
                output_dir / base_name.replace(suffix, f'{salt}{suffix}')
            )
            salt += 1
            if not new_h5ad_path.exists():
                break
    return new_h5ad_path


def _check_h5ad_integrity(
        active_h5ad_path,
        layer,
        log):
    """
    Check that file can even be open and that it contains
    layer, obs, and var datasets/groups

    Parameters
    ----------
    active_h5ad_path:
        path to the h5ad path being checked
    layer:
        the name of the layer being checked for
    log:
        optional CLI log for recording messages
    """
    missing_elements = []
    if layer == 'X':
        full_layer = layer
    elif '/' in layer:
        full_layer = layer
    else:
        full_layer = f'layers/{layer}'

    try:
        with h5py.File(active_h5ad_path, 'r') as src:
            for el in (full_layer, 'var', 'obs'):
                if el not in src:
                    missing_elements.append(el)
    except Exception:
        error_msg = f"\n{traceback.format_exc()}\n"
        error_msg += (
            "This h5ad file is corrupted such that it could not "
            "even be opened with h5py. See above for the specific "
            f"error message raised by h5py {active_h5ad_path}."
        )
        if log is None:
            raise RuntimeError(error_msg)
        else:
            log.error(error_msg)

    if len(missing_elements) > 0:
        msg = (
            "Cannot process this h5ad file. It is missing "
            "the following required elements\n"
            f"{missing_elements}"
        )
        if log is None:
            raise RuntimeError(msg)
        else:
            log.error(msg)


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


def _transpose_file_if_necessary(
        src_path,
        gene_mapper_db_path,
        tmp_dir,
        log,
        layer='X'):
    """
    Check the indices of obs and var in an h5ad file.
    If it seems likely that obs actually points to genes, then
    transpose the h5ad file to a new file in tmp_dir.

    Parameters
    ----------
    src_path:
        the path to the h5ad file being assessed
    tmp_dir:
        path to the directory where a new file can be written, if necessary
    log:
        optional CommandLog
    layer:
        the layer in the h5ad file containing the matrix to transpose
        (currently, only supports 'X'; if layer is anything else, this
        is function becomes a no-op)

    Returns
    -------
    valid_path:
        path to properly shaped (rows are cells, columns are genes)
        h5ad file. Can be src_path if src_path is valid
    has_warnings:
        boolean indicating if any warnings were emitted during this
        process
    """
    if layer != 'X':
        return (src_path, False)

    var = read_df_from_h5ad(src_path, df_name='var')
    if are_valid_genes(
            gene_list=var.index.values,
            gene_mapper_db_path=gene_mapper_db_path):
        return (src_path, False)

    obs = read_df_from_h5ad(src_path, df_name='obs')
    if not are_valid_genes(
            gene_list=obs.index.values,
            gene_mapper_db_path=gene_mapper_db_path):
        return (src_path, False)

    msg = (
        "It appears that your h5ad file has genes as rows and cells "
        f"as columns\nExample row indices {obs.index.values[:5]}\n"
        f"Example column indices {var.index.values[:5]}\n"
        "We will transpose this for you so that rows are cells and columns "
        "are genes"
    )
    if log is not None:
        log.warn(msg)
    else:
        warnings.warn(msg)

    src_path = pathlib.Path(src_path)
    new_path = mkstemp_clean(
        dir=tmp_dir,
        prefix=f'{src_path.name}_TRANSPOSED_',
        suffix='.h5ad'
    )
    transpose_h5ad_file(
        src_path=src_path,
        dst_path=new_path
    )

    return (new_path, True)


def _create_new_obs(
        active_h5ad_path,
        log):
    """
    Create an obs dataframe with unique index values
    (in the event that this is not already true)

    Parameters
    ----------
    active_h5ad_path:
        the path to the h5ad file being validated
    log:
        optional CLI log for recording errors and
        warnigns

    Returns
    -------
    an obs dataframe with unique index values
    (returns None if the original obs dataframe
    satisfied this requirement)
    """
    # check that cell names are not repeated
    obs_original = read_df_from_h5ad(
        h5ad_path=active_h5ad_path,
        df_name='obs')

    obs_unique_index = create_uniquely_indexed_df(
        obs_original)

    if obs_unique_index is not obs_original:
        msg = (
            "The index values in the obs dataframe of your file "
            "are not unique. We are modifying them to be unique. "
        )
        if log is not None:
            log.warn(msg)
        else:
            warnings.warn(msg)
        return obs_unique_index
    return None


def _check_var(
        active_h5ad_path,
        log):
    """
    Check uniqueness of genes. Do mapping to ENSEMBL if needed.

    Parameters
    ----------
    active_h5ad_path:
        path to h5ad file being validated
    log:
        optional CLI log for recording errors and warnings

    Returns
    -------
    var_original:
        original var dataframe
    """
    # check that gene names are not repeated
    var_original = read_df_from_h5ad(
            h5ad_path=active_h5ad_path,
            df_name='var')

    _check_input_gene_names(
        var_df=var_original,
        log=log)

    return (
        var_original
    )


def _check_values(
        active_h5ad_path,
        layer,
        round_to_int,
        expected_max,
        log):
    """
    Check the values of the X matrix

    Parameters
    ----------
    active_h5ad_path:
        path to the h5ad file being validated
    layer:
        the layer of the h5ad file being validated
    round_to_int:
        boolean indicating if we are to round
        values to integers or not
    expected_max:
        expected maximum value of X matrix
        (optional; only triggers a warning)
    log:
        optional CLI log for recording errors
        and warnings

    Returns
    -------
    A boolean indicating whether or not we need
    to cast the X matrix to integers

    A boolean indicating if a warning was emitted

    The dtype of the output matrix (in the event
    rounding is necessary)
    """
    cast_to_int = False
    has_warnings = False
    if round_to_int:
        is_int = is_x_integers(
            h5ad_path=active_h5ad_path,
            layer=layer)
        if not is_int:
            cast_to_int = True

    if expected_max is not None or cast_to_int:
        x_minmax = get_minmax_x_from_h5ad(
            h5ad_path=active_h5ad_path,
            layer=layer)

        if expected_max is not None and x_minmax[1] < expected_max:
            msg = "VALIDATION: CDM expects raw counts data. The maximum value "
            msg += f"of the X matrix in ../{active_h5ad_path.name} is "
            msg += f"{x_minmax[1]}, indicating that this may be "
            msg += "log normalized data. CDM will proceed, but results "
            msg += "may be suspect."
            if log is not None:
                log.warn(msg)
            else:
                warnings.warn(msg)
            has_warnings = True

    output_dtype = None
    if cast_to_int:
        output_dtype = choose_int_dtype(x_minmax)

    return (
        cast_to_int,
        has_warnings,
        output_dtype
    )


def _write_to_tmp_file(
        active_h5ad_path,
        layer,
        var,
        new_obs,
        cast_to_int,
        output_dtype,
        log,
        tmp_dir):

    # Copy data into new file, moving cell by gene data from
    # layer to X

    has_warnings = False
    tmp_h5ad_path = pathlib.Path(
        mkstemp_clean(
            dir=tmp_dir,
            prefix=active_h5ad_path.name,
            suffix='.h5ad')
    )

    copy_layer_to_x(
        original_h5ad_path=active_h5ad_path,
        new_h5ad_path=tmp_h5ad_path,
        layer=layer,
        new_var=var,
        new_obs=new_obs)

    active_h5ad_path = tmp_h5ad_path

    if log is not None:
        msg = (f"VALIDATION: copied "
               f"to ../{active_h5ad_path.name}")
        log.info(msg)

    if cast_to_int:
        msg = "VALIDATION: rounding X matrix of "
        msg += f"{active_h5ad_path.name} to integer values"
        if log is not None:
            log.warn(msg)
        else:
            warnings.warn(msg)
        round_x_to_integers(
            h5ad_path=active_h5ad_path,
            tmp_dir=tmp_dir,
            output_dtype=output_dtype)
        has_warnings = True

    return (
        active_h5ad_path,
        has_warnings
    )


def are_valid_genes(
        gene_list,
        gene_mapper_db_path):
    """
    Accept a list of gene names/identifiers/etc.
    Return True if they are recognized genes.
    False otherwise (in which case, they might be
    cells and the h5ad file may need to be pivoted)
    """
    return species_detection.detect_if_genes(
        db_path=gene_mapper_db_path,
        gene_list=gene_list,
        chunk_size=100000
    )
