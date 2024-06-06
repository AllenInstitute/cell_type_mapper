import copy
import pathlib

import cell_type_mapper


def sanitize_paths(
        input_structure):
    """
    Input_structure is a list, a dict, or a string.
    Absolute file paths in that structure are truncated
    to avoid exposing internal file structures in output logs.
    The returned data is the sanitized version of input_structure.
    """
    mapper_path = pathlib.Path(
        cell_type_mapper.__file__).resolve().absolute().parent.parent
    if isinstance(input_structure, str):
        substitutions = dict()
        for word in input_structure.split():
            path = _word_to_path(word)
            if is_exposed(path):
                abs_path = path.resolve().absolute()
                if is_relative_to(
                        child_path=abs_path,
                        parent_path=mapper_path):
                    safe_path = str(abs_path.relative_to(mapper_path))
                else:
                    safe_path = path.name
                substitutions[word] = safe_path
        if len(substitutions) > 0:
            result = copy.deepcopy(input_structure)
            for old in substitutions:
                result = result.replace(old, substitutions[old])
        else:
            result = input_structure
        return result
    elif isinstance(input_structure, dict):
        new_dict = dict()
        for k in input_structure:
            new_dict[k] = sanitize_paths(input_structure[k])
        return new_dict
    elif isinstance(input_structure, list):
        new_list = [sanitize_paths(w) for w in input_structure]
        return new_list

    return input_structure


def is_exposed(input_path):
    """
    input_path is a pathlib.Path

    Returns True if any part of input_path is a valid path to somewhere
    in the file system. Returns False otherwise.
    """
    if input_path == pathlib.Path('.'):
        return False
    if input_path == pathlib.Path('/'):
        return False
    if input_path.is_file() or input_path.is_dir():
        return True
    return is_exposed(input_path.parent)


def _word_to_path(word):
    """
    Take a string, remove all quotation marks. Return a pathlib.Path
    """
    for char in ('"', "'"):
        word = word.replace(char, '')
    return pathlib.Path(word)


def is_relative_to(
        child_path,
        parent_path):
    """
    Returns True if child_path is a child of parent path. Returns
    False otherwise
    """
    child_path = str(child_path)
    parent_path = str(parent_path)
    return child_path.startswith(parent_path)
