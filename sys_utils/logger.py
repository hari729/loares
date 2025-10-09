import sys
import shutil
import io
import traceback
import atexit
import os
import signal

class Tee:
    def __init__(self, filename, mode='w'):
        self.file = open(filename, mode)
        self.stdout = sys.__stdout__  # Ensures actual terminal output

        # Dynamically get terminal width and apply to NumPy
        try:
            self.columns = shutil.get_terminal_size().columns
        except:
            self.columns = 120  # Fallback if no terminal (e.g., redirected environments)

        try:
            import numpy as np
            np.set_printoptions(linewidth=self.columns)
        except ImportError:
            pass

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


class Tee_general:
    """
    A dual-output stream that writes everything to both
    the terminal (stdout/stderr) and a log file.

    Usage:
        tee = Tee("results/log.txt")
        sys.stdout = tee
        sys.stderr = tee
    """

    def __init__(self, filename, mode='w'):
        self.file = open(filename, mode, buffering=1)  # line-buffered for immediate writes
        self.stdout = sys.__stdout__  # real console stdout
        self.stderr = sys.__stderr__  # real console stderr

        # Optional: set numpy printing width nicely
        try:
            import numpy as np
            cols = shutil.get_terminal_size().columns
            np.set_printoptions(linewidth=cols)
        except Exception:
            pass

    def write(self, data):
        """Write text to both file and terminal immediately."""
        if not data:
            return
        self.file.write(data)
        self.file.flush()            # ensures file is updated in real time
        self.stdout.write(data)      # mirror to terminal
        self.stdout.flush()

    def flush(self):
        """Ensure both streams are flushed."""
        self.file.flush()
        self.stdout.flush()
        self.stderr.flush()

    def close(self):
        """Restore stdout/stderr and close file cleanly (optional)."""
        try:
            self.file.close()
        except Exception:
            pass