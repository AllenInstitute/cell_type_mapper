import json
import time


class CommandLog(object):

    def __init__(self):
        self.t0 = time.time()
        self._log = []

    def add_msg(self, msg):
        self._log.append(self._prepend_time(msg))

    def _prepend_time(self, msg):
        timestamp = time.time()-self.t0
        full_msg = f"{timestamp:.5e} seconds == {str(msg)}"
        return full_msg

    def info(self, msg):
        full_msg = self._prepend_time(msg)
        print(msg)
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

    def write_log(self, output_path):
        with open(output_path, 'w') as out_file:
            out_file.write(json.dumps(self.log, indent=2))
