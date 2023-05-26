import pathlib
import shutil
import tempfile

from hierarchical_mapping.utils.utils import (
    mkstemp_clean,
    _clean_up)


class FileTracker(object):
    """
    A class to manage the moving of files to and from
    a fast temp dir
    """

    def __init__(
            self,
            tmp_dir):
        if tmp_dir is None:
            self.tmp_dir = tmp_dir
        else:
            self.tmp_dir = pathlib.Path(
                    tempfile.mkdtemp(dir=tmp_dir))

        self._path_to_location = dict()
        self._to_write_out = []

    def __del__(self):
        for dst_path in self._to_write_out:
            src_path = self._path_to_location[dst_path]
            shutil.copy(
                src=src_path,
                dst=dst_path)

        if self.tmp_dir is not None:
            _clean_up(self.tmp_dir)

    def add_file(
            self,
            file_path,
            input_only=True):
        """
        if input_only, then the file will not be created
        (if it does not exist)

        Note: even if input_only == False, if the file exists,
        it will not be written out again
        """
        file_path = pathlib.Path(file_path)
        if input_only:
            if not file_path.is_file():
                raise RuntimeError(
                    f"{file_path}\nis not a file")

        path_str = str(file_path.resolve().absolute())
        if self.tmp_dir is None:
            # if there is no tmp_dir, then nothing will be
            # copied anywhere
            self._path_to_location[path_str] = path_str
            return

        suffix = file_path.suffix
        prefix = file_path.name.replace(suffix, '')
        tmp_path = pathlib.Path(
                mkstemp_clean(
                    dir=self.tmp_dir,
                    prefix=f'{prefix}_',
                    suffix=suffix))

        if file_path.is_file():
            shutil.copy(
                src=file_path,
                dst=tmp_path)

        tmp_path = str(tmp_path.resolve().absolute())
        self._path_to_location[path_str] = tmp_path
        if not input_only:
            if not file_path.exists():
                self._to_write_out.append(path_str)

    def real_location(self, file_path):
        """
        Return the location in the temp drive of the
        file pointed to by file_path
        """
        file_path = pathlib.Path(file_path)
        file_path = str(file_path.resolve().absolute())
        if file_path not in self._path_to_location:
            raise RuntimeError(
                f"{file_path}\nnot listed in this FileTracker")
        return pathlib.Path(self._path_to_location[file_path])
