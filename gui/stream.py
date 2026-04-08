"""
Console output redirection for Qt GUI.
"""
from PyQt5.QtCore import QObject, pyqtSignal


class Stream(QObject):
    """Redirect stdout/stderr to Qt signal."""
    
    newText = pyqtSignal(str)
    
    def write(self, text: str):
        """Write text to signal."""
        self.newText.emit(str(text))
    
    def flush(self):
        """Flush (no-op for compatibility)."""
        pass