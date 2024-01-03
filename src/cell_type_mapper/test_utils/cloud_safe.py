import pathlib


def check_not_file(element):
    """
    Check that the contents of element are not absolute file paths

    For use in unit tests verifying that we are not exposing the
    absolute paths to files in the on line MapMyCells app
    """
    if isinstance(element, str):
        msg = ""
        for word in element.split():
            word = word.replace('../', '')
            this = pathlib.Path(word)
            if this.is_file():
                msg += f"{this} is a file\n"
            elif this.is_dir():
                msg += f"{this} is a dir\n"
        if len(msg) > 0:
            raise RuntimeError(msg)
    elif isinstance(element, list):
        for sub in element:
            check_not_file(sub)
    elif isinstance(element, dict):
        for k in element:
            check_not_file(element[k])
