import sys
from typing import Any


class Logger:

    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')

    def __call__(self, message) -> Any:
        self.write(message)
        self.flush()

    def write(self, message, end='\n'):
        self.console.write(message + end)
        self.file.write(message + end)

    def flush(self):
        self.console.flush()
        self.file.flush()
