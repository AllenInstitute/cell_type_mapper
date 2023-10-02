import numpy as np
import pathlib
import warnings

from cell_type_mapper.utils.utils import (
    choose_int_dtype,
    get_timestamp)

from cell_type_mapper.utils.anndata_utils import (
    read_df_from_h5ad,
    write_df_to_h5ad,
    copy_layer_to_x,
    read_uns_from_h5ad,
    write_uns_to_h5ad)

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
        identifiers into the expected form.
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

    Notes
    -----
    If valid_h5ad_path is specified, this is where the validated file
    will be written. If output_dir is specified, a new file will
    be written with a name like

    output_file/input_file_name_VALIDATED_{timestamp}.h5ad

    Both valid_h5ad_path and output_dir cannot be non-None
    """

    h5ad_path = pathlib.Path(h5ad_path)

    if valid_h5ad_path is not None and output_dir is not None:
        raise RuntimeError(
            "Cannot specify both valid_h5ad_path and output_dir; "
            "only specify one")

    if valid_h5ad_path is None and output_dir is None:
        raise RuntimeError(
            "Must specify one of either valid_h5ad_path or output_dir")

    # somewhere in here, check to see if we are working on a layer;
    # if that happens, just go ahead and copy and set current_h5ad_path
    # to new_h5ad_path

    has_warnings = False

    output_path = None

    original_h5ad_path = pathlib.Path(h5ad_path)
    current_h5ad_path = original_h5ad_path

    h5ad_name = original_h5ad_path.name.replace(
                    original_h5ad_path.suffix, '')

    if valid_h5ad_path is None:
        output_dir = pathlib.Path(output_dir)
        timestamp = get_timestamp().replace('-', '')
        new_h5ad_path = output_dir / f'{h5ad_name}_VALIDATED_{timestamp}.h5ad'
    else:
        new_h5ad_path = pathlib.Path(valid_h5ad_path)

    if layer != 'X':
        # Copy data into new file, moving cell by gene data from
        # layer to X
        copy_layer_to_x(
            original_h5ad_path=original_h5ad_path,
            new_h5ad_path=new_h5ad_path,
            layer=layer)
        output_path = new_h5ad_path
        current_h5ad_path = new_h5ad_path

    var_original = read_df_from_h5ad(
            h5ad_path=current_h5ad_path,
            df_name='var')

    # check gene name contents
    error_msg = ''
    unq_genes, unq_gene_count = np.unique(
        var_original.index.values,
        return_counts=True)
    repeated = np.where(unq_gene_count > 1)[0]
    for idx in repeated:
        error_msg += (
            f"gene name '{unq_genes[idx]}' "
            f"occurs {unq_gene_count[idx]} times; "
            "gene names must be unique\n")

    gene_names = set(var_original.index.values)
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

    mapped_var = map_gene_ids_in_var(
        var_df=var_original,
        gene_id_mapper=gene_id_mapper)

    cast_to_int = False
    if round_to_int:
        is_int = is_x_integers(h5ad_path=current_h5ad_path)
        if not is_int:
            cast_to_int = True

    if expected_max is not None or cast_to_int:
        x_minmax = get_minmax_x_from_h5ad(h5ad_path=current_h5ad_path)

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

    if mapped_var is not None or cast_to_int:
        # Copy data over, if it has not already been copied
        if output_path is None:
            copy_layer_to_x(
                original_h5ad_path=original_h5ad_path,
                new_h5ad_path=new_h5ad_path,
                layer='X')

            output_path = new_h5ad_path

    if output_path is not None:
        if log is not None:
            msg = (f"VALIDATION: copied ../{h5ad_path.name} "
                   f"to ../{new_h5ad_path.name}")
            log.info(msg)

    if mapped_var is not None:
        if log is not None:
            msg = "VALIDATION: modifying var dataframe of "
            msg += f"../{original_h5ad_path.name} to include "
            msg += "proper gene identifiers"
            log.info(msg)

        write_df_to_h5ad(
            h5ad_path=new_h5ad_path,
            df_name='var',
            df_value=mapped_var)

        gene_mapping = {
            orig: new
            for orig, new in zip(var_original.index.values,
                                 mapped_var.index.values)
            if orig != new}

        inverse_mapping = dict()
        for orig in gene_mapping:
            new = gene_mapping[orig]
            if new not in inverse_mapping:
                inverse_mapping[new] = []
            inverse_mapping[new].append(orig)

        # check uniqueness of output gene names
        error_msg = ""
        for new in inverse_mapping:
            if len(inverse_mapping[new]) > 1:
                error_msg += (
                    f"gene '{new}' occurs more than once "
                    "in validated h5ad file; originally "
                    f"mapped from '{inverse_mapping[new]}'\n")
        if len(error_msg) > 0:
            if log:
                log.error(error_msg)
            else:
                raise RuntimeError(error_msg)

        uns = read_uns_from_h5ad(new_h5ad_path)
        uns['AIBS_CDM_gene_mapping'] = gene_mapping
        write_uns_to_h5ad(new_h5ad_path, uns)
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
            h5ad_path=new_h5ad_path,
            tmp_dir=tmp_dir,
            output_dtype=output_dtype)
        has_warnings = True

    if log is not None:
        msg = f"DONE VALIDATING ../{h5ad_path.name}; "
        if output_path is not None:
            msg += f"reformatted file written to ../{output_path.name}\n"
        else:
            msg += "no changes required"
        log.info(msg)

    if output_path is None:
        if new_h5ad_path.exists():
            new_h5ad_path.unlink()

    return output_path, has_warnings
