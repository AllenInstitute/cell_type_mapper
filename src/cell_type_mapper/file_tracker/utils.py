import pathlib


def is_file_under_dir(
        file_path,
        dir_path):
    """
    Is the file at file_path somewhere in the tree under the
    directory at dir_path.

    Return a boolean indicating the answer.
    """
    file_path = pathlib.Path(file_path).resolve().absolute()
    dir_path = pathlib.Path(dir_path).resolve().absolute()
    if not file_path.exists():
        return False
    if not dir_path.exists():
        return False

    try:
        file_path.relative_to(dir_path)
    except ValueError:
        return False
    return True
