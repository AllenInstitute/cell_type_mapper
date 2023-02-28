class DummyLock(object):

    def __enter__(self):
        pass

    def __exit__(
            self,
            exception_type,
            exception_value,
            exception_traceback):
        pass
