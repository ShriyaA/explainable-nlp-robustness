import errno
import os

def check_paths_exist(*args):
    for path_str in args:
        if not os.path.exists(path_str):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_str)