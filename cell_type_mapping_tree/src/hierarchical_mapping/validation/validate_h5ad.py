import numpy as np
import pathlib
import shutil
import warnings

from hierarchical_mapping.utils.utils import (
    mkstemp_clean)

from hierarchical_mapping.utils.anndata_utils import (
    read_df_from_h5ad,
    write_df_to_h5ad)

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
        tmp_dir=None):
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

    Returns
    -------
    Path to validated h5ad (if relevant).
    Returns None if no action was taken
    """

    output_path = None

    original_h5ad_path = pathlib.Path(h5ad_path)

    h5ad_name = original_h5ad_path.name.replace(
                    original_h5ad_path.suffix, '')

    new_h5ad_path = mkstemp_clean(
        dir=output_dir,
        prefix=f'{h5ad_name}_VALIDATED_',
        suffix='.h5ad')

    new_h5ad_path = pathlib.Path(new_h5ad_path)

    var_original = read_df_from_h5ad(
            h5ad_path=original_h5ad_path,
            df_name='var')

    mapped_var = map_gene_ids_in_var(
        var_df=var_original,
        gene_id_mapper=gene_id_mapper)

    is_int = is_x_integers(h5ad_path=original_h5ad_path)

    x_minmax = get_minmax_x_from_h5ad(h5ad_path=original_h5ad_path)

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

    if mapped_var is not None or not is_int:
        output_path = new_h5ad_path

        if log is not None:
            msg = f"VALIDATION: copying {h5ad_path} to {new_h5ad_path}"
            log.info(msg)

        shutil.copy(
            src=h5ad_path,
            dst=new_h5ad_path)

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

    if not is_int:
        output_dtype = _choose_dtype(x_minmax)

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

    if log is not None:
        msg = f"DONE VALIDATING {h5ad_path}; "
        if output_path is not None:
            msg += f"reformatted file written to \n{output_path}\n"
        else:
            msg += "no changes required"
        log.info(msg)

    return output_path


def _choose_dtype(
        x_minmax):
    output_dtype = None
    int_min = np.round(x_minmax[0])
    int_max = np.round(x_minmax[1])

    for candidate in (np.uint8, np.int8, np.uint16, np.int16,
                      np.uint32, np.int32, np.uint64, np.int64):
        this_info = np.iinfo(candidate)
        if int_min >= this_info.min and int_max <= this_info.max:
            output_dtype = candidate
            break
    if output_dtype is None:
        output_dtype = int
    return output_dtype
