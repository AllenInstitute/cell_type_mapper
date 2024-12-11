import anndata
import copy
import numpy as np
import sys
import time
import warnings

try:
    import torch
except ImportError:
    pass


from cell_type_mapper.utils.torch_utils import (
    is_torch_available)

from cell_type_mapper.utils.cloud_utils import (
    sanitize_paths)


class CommandLog(object):

    def __init__(
            self,
            verbose_stdout=True):
        """
        Parameters
        ----------
        verbose_stdout:
            boolean controlling whether or not
            info messages get written to stdout
            *in addition* to being written to the
            log.
        """
        self.verbose_stdout = verbose_stdout
        self.t0 = time.time()
        self._log = []

    def add_msg(self, msg):
        self._log.append(self._prepend_time(msg))

    def _prepend_time(self, msg):
        timestamp = time.time()-self.t0
        full_msg = f"{timestamp:.5e} seconds == {str(msg)}"
        return full_msg

    def env(self, msg):
        """
        Print a message about the current operating environment
        (really just prepends 'ENV' to the message and passes it
        along to self.info)
        """
        full_msg = f"ENV: {msg}"
        self.info(full_msg)

    def info(self, msg, to_stdout=False):
        """
        Parameters
        ----------
        msg:
            The string to be logged
        to_stdout:
            boolean overriding self.verbose_stdout
            (so we can force a message to be written
            to stdout, regardless)
        """
        full_msg = self._prepend_time(msg)
        if to_stdout or self.verbose_stdout:
            print(msg)
        self._log.append(full_msg)

    def warn(self, msg):
        warnings.warn(msg)
        msg = f"WARNING: {msg}"
        full_msg = self._prepend_time(msg)
        self._log.append(full_msg)

    def benchmark(self, msg, duration):
        new_msg = f"BENCHMARK: spent {duration:.4e} seconds "
        new_msg += f"{msg}"
        self.info(new_msg)

    def error(self, msg):
        raise RuntimeError(msg)

    @property
    def log(self):
        return self._log

    def write_log(self, output_path, cloud_safe=False):
        to_write = copy.deepcopy(self.log)
        if cloud_safe:
            to_write = sanitize_paths(to_write)
        with open(output_path, 'a') as out_file:
            for line in to_write:
                out_file.write(line + "\n")

    def log_software_env(self):
        """
        Record some boilerplate messages about the versions of software
        being used
        """
        self.env(f"Python version: {sys.version}")
        self.env(f"anndata version: {anndata.__version__}")
        self.env(f"numpy version: {np.__version__}")
        if is_torch_available():
            self.env(f"torch version: {torch.__version__}")
