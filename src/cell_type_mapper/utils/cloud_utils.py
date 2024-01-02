import pathlib


def sanitize_paths(
        input_structure):
    """
    Input_structure is a list, a dict, or a string.
    Absolute file paths in that structure are truncated
    to avoid exposing internal file structures in output logs.
    The returned data is the sanitized version of input_structure.
    """
    if isinstance(input_structure, str):
        new_words = []
        for word in input_structure.split():
            path = pathlib.Path(word)
            parent = path.parent
            if parent.is_file() or parent.is_dir():
                new_words.append(path.name)
            else:
                new_words.append(word)
        return " ".join(new_words)
    elif isinstance(input_structure, dict):
        new_dict = dict()
        for k in input_structure:
            new_dict[k] = sanitize_paths(input_structure[k])
        return new_dict
    elif isinstance(input_structure, list):
        new_list = [sanitize_paths(w) for w in input_structure]
        return new_list

    return input_structure
