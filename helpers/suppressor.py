import os
import sys

class suppress_stdout_stderr:
    """
    A context manager to suppress stdout and stderr at the file descriptor level.
    Useful for silencing C-level libraries (like SuperLU/LAPACK) that print to stdout.
    """
    def __init__(self):
        # Open a dummy file to redirect stdout/stderr to
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        # Save the original file descriptors for stdout and stderr
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Flush Python-level buffers before redirecting FDs
        sys.stdout.flush()
        sys.stderr.flush()
        # Assign the null fds to stdout (1) and stderr (2)
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Restore the original file descriptors
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Flush Python-level buffers again
        sys.stdout.flush()
        sys.stderr.flush()
        # Close the saved and null fds
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
