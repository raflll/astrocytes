import sys
import os
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QStyleFactory
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import QThread, pyqtSignal
from preprocessing import *
from postprocessing import *
from charts import *

# Worker Thread for Image Processing
class ImageProcessingThread(QThread):
    progress_signal = pyqtSignal(str)  # Signal to update UI with messages

    def __init__(self, image_files, visuals_enabled, charts_enabled):
        super().__init__()
        self.image_files = image_files
        self.visuals_enabled = visuals_enabled
        self.charts_enabled = charts_enabled

    def run(self):
        # Run the image processing in a separate thread
        total_images = len(self.image_files)
        self.progress_signal.emit(f"Processing images...")

        features = process_directory(self.image_files)
        print("Binarization complete")
        binarized_path = "binarized_images"
        skeleton_path = "skeletonized_images"
        setup_extract_features(skeleton_path, binarized_path, self.image_files, features, self.visuals_enabled)
        print("Features extracted to extracted_features")
        if self.charts_enabled: charts(self.visuals_enabled)

        self.progress_signal.emit("Processing Complete!")


# Main Application Window
class FolderSelector(QWidget):
    def __init__(self):
        super().__init__()

        # Boolean flags for visuals and charts
        self.visuals_enabled = True
        self.charts_enabled = True
        self.selected_folder = None
        self.processing_thread = None  # Placeholder for thread

        self.init_ui()

    def init_ui(self):
        # Setup window
        self.setWindowTitle("Image Processing Tool")
        self.setGeometry(100, 100, 600, 300)
        self.set_dark_mode()

        layout = QVBoxLayout()

        # Label to display selected folder
        self.label = QLabel("No folder selected")
        layout.addWidget(self.label)

        # Button to open folder dialog
        self.button = QPushButton("Select Folder")
        self.button.clicked.connect(self.select_folder)
        layout.addWidget(self.button)

        # Process button
        self.process_button = QPushButton("Process Images")
        self.process_button.setEnabled(False)  # Disabled until folder is selected
        self.process_button.clicked.connect(self.process_images)
        layout.addWidget(self.process_button)

        # Toggle Visuals Button
        self.visuals_button = QPushButton("Toggle Visuals: ON")
        self.visuals_button.clicked.connect(self.toggle_visuals)
        layout.addWidget(self.visuals_button)

        # Toggle Charts Button
        self.charts_button = QPushButton("Toggle Charts: ON")
        self.charts_button.clicked.connect(self.toggle_charts)
        layout.addWidget(self.charts_button)

        self.setLayout(layout)

    def set_dark_mode(self):
        """Apply a dark mode theme to the application."""
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(55, 55, 55))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(45, 45, 45))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
        dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(30, 30, 30))

        self.setPalette(dark_palette)
        QApplication.setStyle(QStyleFactory.create("Fusion"))

    def select_folder(self):
        """Open a folder selection dialog."""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.selected_folder = folder_path
            self.label.setText(f"Selected: {folder_path}")
            self.process_button.setEnabled(True)  # Enable processing button

    def process_images(self):
        """Start image processing in a separate thread."""
        if self.selected_folder:
            self.label.setText(f"Starting processing of {len(self.selected_folder)} images...")

            print(f"Processing: {self.selected_folder}")

            # Create and start the processing thread
            self.processing_thread = ImageProcessingThread(self.selected_folder, self.visuals_enabled, self.charts_enabled)
            self.processing_thread.progress_signal.connect(self.update_status)  # Connect signal to update UI
            self.processing_thread.start()

    def update_status(self, message):
        """Update the UI label with progress messages from the thread."""
        self.label.setText(message)

    def toggle_visuals(self):
        """Toggle the visuals boolean flag."""
        self.visuals_enabled = not self.visuals_enabled
        self.visuals_button.setText(f"Toggle Visuals: {'ON' if self.visuals_enabled else 'OFF'}")

    def toggle_charts(self):
        """Toggle the charts boolean flag."""
        self.charts_enabled = not self.charts_enabled
        self.charts_button.setText(f"Toggle Charts: {'ON' if self.charts_enabled else 'OFF'}")
