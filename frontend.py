import sys
import os
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                            QLabel, QFileDialog, QStyleFactory, QMainWindow,
                            QTabWidget, QFrame, QProgressBar, QHBoxLayout,
                            QScrollArea)
from PyQt6.QtGui import QPalette, QColor, QFont, QPixmap
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from preprocessing import *
from postprocessing import *
from charts import *
from feature_comparison import *

class ImageProcessingThread(QThread):
    progress_signal = pyqtSignal(str)

    def __init__(self, image_files, charts_enabled):
        super().__init__()
        self.image_files = image_files
        self.charts_enabled = charts_enabled

    def run(self):
        total_images = len(self.image_files)
        self.progress_signal.emit(f"Processing images...")

        process_directory(self.image_files)
        print("Binarization complete")
        binarized_path = "binarized_images"
        skeleton_path = "skeletonized_images"
        setup_extract_features(skeleton_path, binarized_path, self.image_files, False)
        print("Features extracted to extracted_features")
        if self.charts_enabled: charts(False)

        self.progress_signal.emit("Processing Complete!")

class ModelTrainingThread(QThread):
    training_complete = pyqtSignal()
    progress_signal = pyqtSignal(str)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def run(self):
        self.progress_signal.emit(f"Training {self.model_name} model...")
        train_model(self.model_name, False)
        self.progress_signal.emit(f"Training {self.model_name} model complete!")
        self.training_complete.emit()

class ModernUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.charts_enabled = True
        self.selected_folder = None
        self.processing_thread = None
        self.training_thread = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Image Analysis Tool")
        self.setGeometry(100, 100, 800, 600)
        self.set_dark_mode()

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create tab widget
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background: #2b2b2b;
                border-radius: 5px;
            }
            QTabBar::tab {
                background: #353535;
                color: #ffffff;
                padding: 10px 20px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #2b2b2b;
                border-bottom: 2px solid #0078d4;
            }
            QTabBar::tab:hover {
                background: #404040;
            }
        """)

        # Create tabs
        files_tab = QWidget()
        stats_tab = QWidget()
        viz_tab = QWidget()

        self.setup_files_tab(files_tab)
        self.setup_stats_tab(stats_tab)
        self.setup_viz_tab(viz_tab)

        tabs.addTab(files_tab, "Files")
        tabs.addTab(stats_tab, "Statistics")
        tabs.addTab(viz_tab, "Visualizations")

        layout.addWidget(tabs)

    def setup_files_tab(self, tab):
        layout = QVBoxLayout(tab)

        # Status frame
        status_frame = QFrame()
        status_frame.setStyleSheet("""
            QFrame {
                background: #333333;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        status_layout = QVBoxLayout(status_frame)

        # Status label with modern font
        self.status_label = QLabel("No folder selected")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
                padding: 10px;
            }
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(self.status_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #2b2b2b;
                border-radius: 5px;
                text-align: center;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 3px;
            }
        """)
        self.progress_bar.hide()
        status_layout.addWidget(self.progress_bar)

        layout.addWidget(status_frame)

        # Buttons frame
        buttons_frame = QFrame()
        buttons_frame.setStyleSheet("""
            QFrame {
                background: #333333;
                border-radius: 10px;
                padding: 20px;
                margin-top: 20px;
            }
        """)
        buttons_layout = QVBoxLayout(buttons_frame)

        # Create styled buttons
        button_style = """
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #006cbd;
            }
            QPushButton:disabled {
                background-color: #454545;
                color: #888888;
            }
        """

        self.select_button = QPushButton("Select Folder")
        self.select_button.setStyleSheet(button_style)
        self.select_button.clicked.connect(self.select_folder)
        buttons_layout.addWidget(self.select_button)

        self.process_button = QPushButton("Process Images")
        self.process_button.setStyleSheet(button_style)
        self.process_button.setEnabled(False)
        self.process_button.clicked.connect(self.process_images)
        buttons_layout.addWidget(self.process_button)

        toggle_style = button_style.replace("#0078d4", "#444444")
        self.charts_button = QPushButton("Toggle Charts: ON")
        self.charts_button.setStyleSheet(toggle_style)
        self.charts_button.clicked.connect(self.toggle_charts)
        buttons_layout.addWidget(self.charts_button)

        layout.addWidget(buttons_frame)
        layout.addStretch()

    def setup_stats_tab(self, tab):
        layout = QVBoxLayout(tab)

        # Status frame for model training
        model_status_frame = QFrame()
        model_status_frame.setStyleSheet("""
            QFrame {
                background: #333333;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
            }
        """)
        model_status_layout = QVBoxLayout(model_status_frame)

        # Status label for model training
        self.model_status_label = QLabel("No model trained")
        self.model_status_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
                padding: 10px;
            }
        """)
        self.model_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        model_status_layout.addWidget(self.model_status_label)

        # Train model button
        button_style = """
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #006cbd;
            }
            QPushButton:disabled {
                background-color: #454545;
                color: #888888;
            }
        """

        self.train_model_button = QPushButton("Train Model (ENet)")
        self.train_model_button.setStyleSheet(button_style)
        self.train_model_button.clicked.connect(lambda: self.train_model("ENet"))
        model_status_layout.addWidget(self.train_model_button)

        layout.addWidget(model_status_frame)

        # Create a scroll area for plots
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #2b2b2b;
                width: 10px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #555555;
                min-height: 20px;
                border-radius: 5px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        # Plot display frame (inside scroll area)
        self.plot_display_frame = QFrame()
        self.plot_display_frame.setStyleSheet("""
            QFrame {
                background: #333333;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        plot_layout = QVBoxLayout(self.plot_display_frame)

        # Feature importance section
        importance_section = QFrame()
        importance_layout = QVBoxLayout(importance_section)

        importance_title = QLabel("Feature Importance")
        importance_title.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold;")
        importance_layout.addWidget(importance_title)

        self.importance_plot_label = QLabel("Feature importance plot will appear here after training")
        self.importance_plot_label.setStyleSheet("color: #888888; font-size: 14px;")
        self.importance_plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.importance_plot_label.setMinimumHeight(400)  # Give enough space for the plot
        importance_layout.addWidget(self.importance_plot_label)

        plot_layout.addWidget(importance_section)

        # Add a separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #444444; min-height: 2px; margin: 20px 0;")
        plot_layout.addWidget(separator)

        # SHAP section
        shap_section = QFrame()
        shap_layout = QVBoxLayout(shap_section)

        shap_title = QLabel("SHAP Values")
        shap_title.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold;")
        shap_layout.addWidget(shap_title)

        self.shap_plot_label = QLabel("SHAP values plot will appear here after training")
        self.shap_plot_label.setStyleSheet("color: #888888; font-size: 14px;")
        self.shap_plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.shap_plot_label.setMinimumHeight(400)  # Give enough space for the plot
        shap_layout.addWidget(self.shap_plot_label)

        plot_layout.addWidget(shap_section)

        # Container for class-specific SHAP plots
        self.class_shap_container = QFrame()
        self.class_shap_layout = QVBoxLayout(self.class_shap_container)
        plot_layout.addWidget(self.class_shap_container)

        # Set the plot display frame as the widget for the scroll area
        scroll_area.setWidget(self.plot_display_frame)
        layout.addWidget(scroll_area)

    def setup_viz_tab(self, tab):
        layout = QVBoxLayout(tab)
        placeholder = QLabel("Visualizations will be displayed here")
        placeholder.setStyleSheet("""
            QLabel {
                color: #888888;
                font-size: 16px;
            }
        """)
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(placeholder)

    def set_dark_mode(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
        dark_palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

        self.setPalette(dark_palette)
        QApplication.setStyle(QStyleFactory.create("Fusion"))

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.selected_folder = folder_path
            self.status_label.setText(f"Selected: {folder_path}")
            self.process_button.setEnabled(True)

    def process_images(self):
        if self.selected_folder:
            self.status_label.setText("Processing images...")
            self.progress_bar.show()
            self.progress_bar.setRange(0, 0)  # Infinite progress bar

            self.processing_thread = ImageProcessingThread(
                self.selected_folder,
                self.charts_enabled
            )
            self.processing_thread.progress_signal.connect(self.update_status)
            self.processing_thread.finished.connect(self.processing_complete)
            self.processing_thread.start()

    def processing_complete(self):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)

    def update_status(self, message):
        self.status_label.setText(message)

    def update_model_status(self, message):
        self.model_status_label.setText(message)

    def toggle_charts(self):
        self.charts_enabled = not self.charts_enabled
        self.charts_button.setText(f"Toggle Charts: {'ON' if self.charts_enabled else 'OFF'}")

    def train_model(self, model_name):
        self.update_model_status(f"Training {model_name} model...")
        self.train_model_button.setEnabled(False)

        # Clear any previous class-specific SHAP plots
        for i in reversed(range(self.class_shap_layout.count())):
            self.class_shap_layout.itemAt(i).widget().setParent(None)

        # Create and start the training thread
        self.training_thread = ModelTrainingThread(model_name)
        self.training_thread.progress_signal.connect(self.update_model_status)
        self.training_thread.training_complete.connect(self.training_complete)
        self.training_thread.start()

    def training_complete(self):
        self.train_model_button.setEnabled(True)
        self.update_model_status("Model training complete")

        # Display plots
        self.display_plots("ENet")

    def display_plots(self, model_name):
        # Check if plot files exist
        importance_path = f"{model_name}_feature_importance.png"
        shap_path = f"{model_name}_shap.png"

        # Load and display importance plot if it exists
        if os.path.exists(importance_path):
            importance_pixmap = QPixmap(importance_path)

            # Scale while maintaining aspect ratio, but allow full height
            scaled_pixmap = importance_pixmap.scaled(
                self.importance_plot_label.width(),
                importance_pixmap.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.importance_plot_label.setPixmap(scaled_pixmap)
            # Adjust the label height to fit the scaled pixmap
            self.importance_plot_label.setMinimumHeight(scaled_pixmap.height())
        else:
            self.importance_plot_label.setText(f"Feature importance plot not found ({importance_path})")

        # Load and display SHAP plot if it exists
        if os.path.exists(shap_path):
            shap_pixmap = QPixmap(shap_path)
            scaled_shap = shap_pixmap.scaled(
                self.shap_plot_label.width(),
                shap_pixmap.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            self.shap_plot_label.setPixmap(scaled_shap)
            self.shap_plot_label.setMinimumHeight(scaled_shap.height())
        else:
            # Check for class-specific SHAP plots
            found_class_plots = False
            for i in range(3):  # Assuming at most 3 classes (0, 1, 2)
                class_shap_path = f"{model_name}_shap_class_{i}.png"
                if os.path.exists(class_shap_path):
                    found_class_plots = True

                    # Create section title for this class
                    class_title = QLabel(f"SHAP Values for Class {i}")
                    class_title.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold; margin-top: 20px;")
                    self.class_shap_layout.addWidget(class_title)

                    # Create image label
                    class_label = QLabel()
                    class_pixmap = QPixmap(class_shap_path)
                    scaled_class = class_pixmap.scaled(
                        self.plot_display_frame.width() - 60,  # Account for padding
                        class_pixmap.height(),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )

                    class_label.setPixmap(scaled_class)
                    class_label.setMinimumHeight(scaled_class.height())
                    self.class_shap_layout.addWidget(class_label)

            if not found_class_plots:
                self.shap_plot_label.setText(f"SHAP plot not found ({shap_path})")
