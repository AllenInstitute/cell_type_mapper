import pathlib
import shutil
import warnings

from hierarchical_mapping.utils.utils import (
    choose_int_dtype,
    get_timestamp)

from hierarchical_mapping.utils.anndata_utils import (
    read_df_from_h5ad,
    write_df_to_h5ad,
    copy_layer_to_x,
    read_uns_from_h5ad,
    write_uns_to_h5ad)

from hierarchical_mapping.validation.utils import (
    is_x_integers,
    round_x_to_integers,
    get_minmax_x_from_h5ad,
    map_gene_ids_in_var)


def validate_h5ad(
        h5ad_path,
        output_dir,
        gene_id_mapper,
        log=None,
        expected_max=20,
        tmp_dir=None,
        layer='X',
        normalize_to_int=True,
        valid_h5ad_path=None):
    """
    Perform validation transformations on h5ad file.

    Parameters
    ----------
    h5ad_path:
        Path to the source h5ad file
    output_dir:
        Dir where new h5ad file can be written (if necessary)
    gene_id_mapper:
        the GeneIdMapper that will handle the mapping of gene
        identifiers into the expected form.
    log:
        Optional logger to log messages for CLI
    expected_max:
        If the max X value is less than this, emit a warning
        indicating that we think the normalization of the
        data is incorrect
    tmp_dir:
       Dir where scratch data products can be written if needed
    layer:
        The layer in the source h5ad file where the cell by gene
        data will be retrieved. Regardless, it will be written to
        'X' in the validated file.
    normalize_to_int:
        If True, cast the cell by gene matrix to an integer.
        If False, leave it untouched relative to input.

    Returns
    -------
    Path to validated h5ad (if relevant).
    Returns None if no action was taken
    """

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

    mapped_var = map_gene_ids_in_var(
        var_df=var_original,
        gene_id_mapper=gene_id_mapper)

    cast_to_int = False
    if normalize_to_int:
        is_int = is_x_integers(h5ad_path=current_h5ad_path)
        if not is_int:
            cast_to_int = True

    x_minmax = get_minmax_x_from_h5ad(h5ad_path=current_h5ad_path)

    if x_minmax[1] < expected_max:
        msg = "VALIDATION: CDM expects raw counts data. The maximum value "
        msg += f"of the X matrix in {original_h5ad_path} is "
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
            shutil.copy(
                src=h5ad_path,
                dst=new_h5ad_path)

            output_path = new_h5ad_path

    if output_path is not None:
        if log is not None:
            msg = f"VALIDATION: copyied {h5ad_path} to {new_h5ad_path}"
            log.info(msg)

    if mapped_var is not None:
        if log is not None:
            msg = "VALIDATION: modifying var dataframe of "
            msg += f"{original_h5ad_path} to include "
            msg += "proper gene identifiers"
            log.info(msg)

        write_df_to_h5ad(
            h5ad_path=new_h5ad_path,
            df_name='var',
            df_value=mapped_var)

        gene_mapping = {
            orig: new
            for orig, new in zip(var_original.index.values,
                                 mapped_var.index.values)}

        uns = read_uns_from_h5ad(new_h5ad_path)
        uns['AIBS_CDM_gene_mapping'] = gene_mapping
        write_uns_to_h5ad(new_h5ad_path, uns)
        has_warnings = True

    if cast_to_int:
        output_dtype = choose_int_dtype(x_minmax)

        msg = "VALIDATION: rounding X matrix of "
        msg += f"{original_h5ad_path} to integer values"
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
        msg = f"DONE VALIDATING {h5ad_path}; "
        if output_path is not None:
            msg += f"reformatted file written to {output_path}\n"
        else:
            msg += "no changes required"
        log.info(msg)

    if output_path is None:
        if new_h5ad_path.exists():
            new_h5ad_path.unlink()

    return output_path, has_warnings
