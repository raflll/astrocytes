import sys
import time
import datetime
from typing import Iterable, TypeVar, Iterator
from PySide6.QtWidgets import QApplication, QProgressDialog
from PySide6.QtCore import Qt

T = TypeVar('T')

class pyside_progress:
    """
    A tqdm-like progress bar wrapper for PySide6 that can be used exactly like tqdm.
    
    Usage:
    ```python
    for item in pyside_progress(iterable):
        # Your processing logic here
        time.sleep(0.1)
    ```
    
    Advanced usage:
    ```python
    # With custom title and description
    for item in pyside_progress(iterable, title="Custom Title", desc="Processing..."):
        # Your processing logic here
        pass
    ```
    """
    def __init__(
        self, 
        iterable: Iterable[T], 
        total: int = None, 
        desc: str = "Processing...", 
        title: str = "Progress",
        disable: bool = False
    ):
        # Ensure we have a QApplication
        if not QApplication.instance():
            self.app = QApplication(sys.argv)
        
        # Determine total iterations
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                # For generators or iterators without length
                iterable = list(iterable)
                total = len(iterable)
        
        # Store parameters
        self.iterable = iterable
        self.total = total
        self.desc = desc
        self.title = title
        self.disable = disable
        
        # Progress dialog setup
        if not disable:
            self.progress = QProgressDialog(desc, "Cancel", 0, total)
            self.progress.setWindowModality(Qt.WindowModality.WindowModal)
            self.progress.setWindowTitle(title)
            self.progress.show()
        else:
            self.progress = None
        
        # Timing setup
        self.start_time = time.time()
        self.current = 0
    
    def __iter__(self) -> Iterator[T]:
        # If disabled, just return the original iterable
        if self.disable:
            yield from self.iterable
            return
        
        try:
            for item in self.iterable:
                # Update progress
                self.current += 1
                
                # Calculate and update estimated time
                current_time = time.time()
                elapsed_time = current_time - self.start_time
                
                if self.current > 0:
                    # Estimate time remaining
                    time_per_iteration = elapsed_time / self.current
                    remaining_iterations = self.total - self.current
                    estimated_remaining_time = remaining_iterations * time_per_iteration
                    
                    # Format remaining time
                    remaining_str = str(datetime.timedelta(seconds=int(estimated_remaining_time)))
                    
                    # Update progress label
                    self.progress.setLabelText(
                        f"{self.desc} (Est. time remaining: {remaining_str})"
                    )
                
                # Set progress value
                self.progress.setValue(self.current)
                
                # Process Qt events
                QApplication.processEvents()
                
                # Check for cancel
                if self.progress.wasCanceled():
                    break
                
                # Yield the current item
                yield item
        
        finally:
            # Ensure progress dialog closes
            if self.progress:
                self.progress.setValue(self.total)
                self.progress.close()

# Convenience function to use like tqdm
def pyside_tqdm(
    iterable: Iterable[T] = None, 
    total: int = None, 
    desc: str = "Processing...", 
    title: str = "Progress",
    disable: bool = False
):
    """
    Drop-in replacement for tqdm that uses PySide6 progress dialog.
    
    Can be used with or without an iterable.
    """
    if iterable is None:
        # If no iterable provided, return a function that takes an iterable
        return lambda x: pyside_progress(x, total, desc, title, disable)
    return pyside_progress(iterable, total, desc, title, disable)