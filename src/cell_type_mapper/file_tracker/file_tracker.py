import pathlib
import shutil
import tempfile

from cell_type_mapper.utils.utils import (
    mkstemp_clean,
    _clean_up)


class FileTracker(object):
    """
    A class to manage the moving of files to and from
    a fast temp dir
    """

    def __init__(
            self,
            tmp_dir,
            log=None):
        if tmp_dir is None:
            self.tmp_dir = tmp_dir
        else:
            self.tmp_dir = pathlib.Path(
                    tempfile.mkdtemp(dir=tmp_dir,
                                     prefix='file_tracker_'))

        self._path_to_location = dict()
        self._file_pre_exists = dict()
        self._to_write_out = []
        self.log = log

    def __del__(self):
        for dst_path in self._to_write_out:
            src_path = pathlib.Path(self._path_to_location[dst_path])
            dst_path = pathlib.Path(dst_path)
            shutil.copy(
                src=src_path,
                dst=dst_path)
            if self.log is not None:
                msg = (f"FILE TRACKER: copied ../{src_path.name} "
                       f"to ../{dst_path.name}")
                self.log.info(msg)

        if self.tmp_dir is not None:
            if self.log is not None:
                msg = f"FILE TRACKER: cleaning up ../{self.tmp_dir.name}"
                self.log.info(msg)
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
        file_str = f'{file_path.parent.name}/{file_path.name}'
        if not file_path.is_file():
            if file_path.exists():
                raise RuntimeError(
                    f"../{file_str}\nexists but is not a file")
            elif input_only:
                raise RuntimeError(
                    f"../{file_str}\nis not a file")
            else:
                if not file_path.parent.is_dir():
                    raise RuntimeError(
                        f"will not be able to write to ../{file_str}\n"
                        f"../{file_path.parent.name} is not a dir")

        path_str = str(file_path.resolve().absolute())

        if file_path.is_file():
            self._file_pre_exists[path_str] = True
        else:
            self._file_pre_exists[path_str] = False

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
            if self.log is not None:
                msg = (f"FILE TRACKER: copied ../{file_path.name} "
                       f"to ../{tmp_path.name}")
                self.log.info(msg)

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
        file_path_str = str(file_path.resolve().absolute())
        if file_path_str not in self._path_to_location:
            raise RuntimeError(
                f"../{file_path.name}\nnot listed in this FileTracker")
        return pathlib.Path(self._path_to_location[file_path_str])

    def file_exists(self, file_path):
        """
        Did the tracked file exist before we created it?
        """
        file_path = pathlib.Path(file_path)
        file_path_str = str(file_path.resolve().absolute())
        if file_path_str not in self._file_pre_exists:
            raise RuntimeError(
                f"../{file_path.name}\nnot listed in this FileTracker")
        return self._file_pre_exists[file_path_str]
