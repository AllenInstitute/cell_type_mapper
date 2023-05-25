import anndata
import pathlib
import shutil
import gc
import time

from hierarchical_mapping.utils.utils import (
    mkstemp_clean)


def _copy_over_file(file_path, tmp_dir, log, copy=True):
    """
    If a file exists, copy it into the tmp_dir.

    Parameters
    ----------
    file_path:
        the path to the file we are considering
    tmp_dir:
        the path to the fast tmp_dir
    log:
        CommandLog to record actions

    Returns
    -------
    new_path:
        Where the file was copied (even if file was not copied,
        return a to a file in tmp_dir)

    valid:
        boolean indicating whether this file can be used (True)
        or if it is just a placeholder (False)
    """
    file_path = pathlib.Path(file_path)
    tmp_dir = pathlib.Path(tmp_dir)
    if copy:
        new_path = mkstemp_clean(
                dir=tmp_dir,
                prefix=f"{file_path.name.replace(file_path.suffix, '')}_",
                suffix=file_path.suffix)
    else:
        new_path = str(file_path)

    is_valid = False
    if file_path.exists():
        if not file_path.is_file():
            log.error(
                f"{file_path} exists but is not a file")
        else:
            if copy:
                t0 = time.time()
                log.info(f"copying {file_path}")
                shutil.copy(src=file_path, dst=new_path)
                duration = time.time()-t0
                log.info(f"copied {file_path} to {new_path} "
                        f"in {duration:.4e} seconds")
            is_valid = True
    else:
        # check that we can write the specified file
        try:
            with open(file_path, 'w') as out_file:
                out_file.write("junk")
            file_path.unlink()
        except FileNotFoundError:
            raise RuntimeError(
                "could not write to "
                f"{file_path.resolve().absolute()}")

    return new_path, is_valid


def _make_temp_path(
        config_dict,
        tmp_dir,
        log,
        suffix,
        prefix,
        copy=True):
    """
    Create a temp path for an actual file.

    Returns
    -------
    {'tmp': tmp_path created
     'path': path in actual storage (can be None)
     'is_valid': True if 'path' exists; False if must be created}
    """

    if "path" in config_dict:
        file_path = pathlib.Path(
            config_dict["path"])
        if copy:
            (tmp_path,
            is_valid) = _copy_over_file(
                    file_path=file_path,
                    tmp_dir=tmp_dir,
                    log=log)
        else:
            tmp_path = str(file_path)
            is_valid = file_path.exists() and file_path.is_file()
    else:
        tmp_path = pathlib.Path(
            mkstemp_clean(
                dir=tmp_dir,
                prefix=prefix,
                suffix=suffix))
        is_valid = False
        file_path = None

    return {'tmp': tmp_path,
            'path': file_path,
            'is_valid': is_valid}


def _check_config(config_dict, config_name, key_name, log):
    if isinstance(key_name, list):
        for el in key_name:
            _check_config(
                config_dict=config_dict,
                config_name=config_name,
                key_name=el,
                log=log)
    else:
        if key_name not in config_dict:
            log.error(f"'{config_name}' config missing key '{key_name}'")


def _get_query_gene_names(query_gene_path):
    result = _get_query_gene_names_worker(query_gene_path)
    gc.collect()
    return result


def _get_query_gene_names_worker(query_gene_path):
    a_data = anndata.read_h5ad(query_gene_path, backed='r')
    gene_names = list(a_data.var_names)
    return gene_names
