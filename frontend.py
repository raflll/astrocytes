import sys
import os
import random
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QPushButton,
                            QLabel, QFileDialog, QStyleFactory, QMainWindow,
                            QTabWidget, QFrame, QProgressBar, QHBoxLayout,
                            QScrollArea, QGridLayout, QSizePolicy)
from PyQt6.QtGui import QPalette, QColor, QFont, QPixmap, QImage
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

class FeatureVisualizationThread(QThread):
    visualization_complete = pyqtSignal(dict)
    progress_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

    def run(self):
        self.progress_signal.emit("Finding random feature...")

        # Get all available feature CSV files
        feature_files = []
        features_dir = "extracted_features"
        if os.path.exists(features_dir):
            feature_files = [os.path.join(features_dir, f) for f in os.listdir(features_dir)
                            if f.endswith('_features.csv')]

        if not feature_files:
            self.progress_signal.emit("No feature files found!")
            return

        # Randomly select a CSV file
        selected_file = random.choice(feature_files)
        self.progress_signal.emit(f"Selected: {os.path.basename(selected_file)}")

        try:
            # Load the CSV file
            df = pd.read_csv(selected_file)

            if df.empty:
                self.progress_signal.emit("Selected CSV file is empty!")
                return

            # Randomly select a row from the dataframe
            selected_row = df.sample(1).iloc[0]

            # Get the file name and object label for this feature
            file_name = selected_row['file_name']
            object_label = selected_row['object_label']

            # Check if we can find the original, skeleton, and binarized images
            data_image_path = self.find_image_file(file_name)
            skeleton_image_path = self.find_skeleton_file(file_name)
            binarized_image_path = self.find_binarized_file(file_name)

            if not (data_image_path and skeleton_image_path and binarized_image_path):
                self.progress_signal.emit("Could not find all necessary image files!")
                return

            # Load the images
            data_image = cv2.imread(data_image_path, cv2.IMREAD_GRAYSCALE)
            skeleton_image = cv2.imread(skeleton_image_path, cv2.IMREAD_GRAYSCALE)
            binarized_image = cv2.imread(binarized_image_path, cv2.IMREAD_GRAYSCALE)

            if data_image is None or skeleton_image is None or binarized_image is None:
                self.progress_signal.emit("Failed to load image files!")
                return

            # Create a colored version of the binarized image for overlay
            binarized_colored = cv2.cvtColor(binarized_image, cv2.COLOR_GRAY2BGR)
            data_colored = cv2.cvtColor(data_image, cv2.COLOR_GRAY2BGR)

            # Extract the mask for the selected object
            component_mask = (self.get_labeled_components(binarized_image) == object_label).astype(np.uint8) * 255

            if np.sum(component_mask) == 0:
                self.progress_signal.emit(f"Could not find object with label {object_label}!")
                return

            # Get the bounding box for the feature
            x, y, w, h = cv2.boundingRect(component_mask)

            # Create a copy of the skeleton image with a red overlay for the feature
            skeleton_colored = cv2.cvtColor(skeleton_image, cv2.COLOR_GRAY2BGR)
            feature_mask = np.zeros_like(skeleton_colored)
            feature_mask[component_mask > 0] = [0, 0, 255]  # Red overlay
            blended = cv2.addWeighted(skeleton_colored, 1.0, feature_mask, 0.5, 0)

            # Draw a green rectangle around the feature
            cv2.rectangle(blended, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Also create a blended image with the binarized and skeleton
            blended_skeleton = self.blend_skeleton(binarized_colored, skeleton_image)

            # Save temporary images for display
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)

            blended_path = os.path.join(temp_dir, "feature_overlay.png")
            original_path = os.path.join(temp_dir, "original.png")
            skeleton_path = os.path.join(temp_dir, "skeleton_overlay.png")

            cv2.imwrite(blended_path, blended)
            cv2.imwrite(original_path, data_colored)
            cv2.imwrite(skeleton_path, blended_skeleton)

            # Prepare stats to display
            stats = {
                'file_name': file_name,
                'object_label': object_label,
                'area': selected_row.get('area', 'N/A'),
                'perimeter': selected_row.get('perimeter', 'N/A'),
                'num_branches': selected_row.get('num_branches', 'N/A'),
                'num_projections': selected_row.get('num_projections', 'N/A'),
                'circularity': selected_row.get('circularity', 'N/A'),
                'fractal_dim': selected_row.get('fractal_dim', 'N/A'),
                'roundness': selected_row.get('roundness', 'N/A'),
                'total_skeleton_length': selected_row.get('total_skeleton_length', 'N/A'),
                'feature_overlay_path': blended_path,
                'original_path': original_path,
                'skeleton_overlay_path': skeleton_path,
                'x': x,
                'y': y,
                'width': w,
                'height': h
            }

            self.progress_signal.emit("Feature visualization ready!")
            self.visualization_complete.emit(stats)

        except Exception as e:
            self.progress_signal.emit(f"Error during visualization: {str(e)}")

    def get_labeled_components(self, binary_image):
        # Run connected components on the binary image
        num_labels, labels = cv2.connectedComponents(binary_image)
        return labels

    def blend_skeleton(self, original, skeleton):
        # Create a copy of the original
        og = original.copy()
        # Create a mask from the skeleton
        skeleton_mask = skeleton > 0
        # Apply red color to skeleton pixels
        og[skeleton_mask] = [0, 0, 255]
        return og

    def find_image_file(self, file_name):
        # Look for the file in potential data directories
        data_dirs = ["data", "data/Control", "data/Images", "data/Phenotype 1", "data/Phenotype 2"]

        for data_dir in data_dirs:
            if not os.path.exists(data_dir):
                continue

            # Check if the file exists directly in the directory
            path = os.path.join(data_dir, file_name)
            if os.path.exists(path):
                return path

            # Also check subdirectories if any
            for root, dirs, files in os.walk(data_dir):
                if file_name in files:
                    return os.path.join(root, file_name)

        return None

    def find_skeleton_file(self, file_name):
        # Look for the file in the skeletonized directory
        skeleton_dirs = ["skeletonized_images", "skeletonized_images/Control",
                        "skeletonized_images/Images", "skeletonized_images/Phenotype 1",
                        "skeletonized_images/Phenotype 2"]

        for skel_dir in skeleton_dirs:
            if not os.path.exists(skel_dir):
                continue

            # Check if the file exists directly in the directory
            path = os.path.join(skel_dir, file_name)
            if os.path.exists(path):
                return path

            # Also check subdirectories if any
            for root, dirs, files in os.walk(skel_dir):
                if file_name in files:
                    return os.path.join(root, file_name)

        return None

    def find_binarized_file(self, file_name):
        # Look for the file in the binarized directory
        binary_dirs = ["binarized_images", "binarized_images/Control",
                      "binarized_images/Images", "binarized_images/Phenotype 1",
                      "binarized_images/Phenotype 2"]

        for bin_dir in binary_dirs:
            if not os.path.exists(bin_dir):
                continue

            # Check if the file exists directly in the directory
            path = os.path.join(bin_dir, file_name)
            if os.path.exists(path):
                return path

            # Also check subdirectories if any
            for root, dirs, files in os.walk(bin_dir):
                if file_name in files:
                    return os.path.join(root, file_name)

        return None

class ModernUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.charts_enabled = True
        self.selected_folder = None
        self.processing_thread = None
        self.training_thread = None
        self.visualization_thread = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Image Analysis Tool")
        self.setGeometry(100, 100, 1200, 800)
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

        # Status frame for visualization
        viz_status_frame = QFrame()
        viz_status_frame.setStyleSheet("""
            QFrame {
                background: #333333;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
            }
        """)
        viz_status_layout = QVBoxLayout(viz_status_frame)

        # Status label for visualization
        self.viz_status_label = QLabel("No feature visualized")
        self.viz_status_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
                padding: 10px;
            }
        """)
        self.viz_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        viz_status_layout.addWidget(self.viz_status_label)

        # Random feature button
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

        self.viz_button = QPushButton("Show Random Feature")
        self.viz_button.setStyleSheet(button_style)
        self.viz_button.clicked.connect(self.visualize_random_feature)
        viz_status_layout.addWidget(self.viz_button)

        layout.addWidget(viz_status_frame)

        # Create a scroll area for visualization
        viz_scroll_area = QScrollArea()
        viz_scroll_area.setWidgetResizable(True)
        viz_scroll_area.setStyleSheet("""
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

        # Visualization display frame (inside scroll area)
        self.viz_display_frame = QFrame()
        self.viz_display_frame.setStyleSheet("""
            QFrame {
                background: #333333;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        self.viz_layout = QGridLayout(self.viz_display_frame)

        # Feature visualization section
        self.feature_image_label = QLabel("Feature visualization will appear here")
        self.feature_image_label.setStyleSheet("color: #888888; font-size: 14px;")
        self.feature_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.feature_image_label.setMinimumHeight(300)
        self.viz_layout.addWidget(self.feature_image_label, 0, 0)

        self.original_image_label = QLabel("Original image will appear here")
        self.original_image_label.setStyleSheet("color: #888888; font-size: 14px;")
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image_label.setMinimumHeight(300)
        self.viz_layout.addWidget(self.original_image_label, 0, 1)

        self.skeleton_image_label = QLabel("Skeleton overlay will appear here")
        self.skeleton_image_label.setStyleSheet("color: #888888; font-size: 14px;")
        self.skeleton_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.skeleton_image_label.setMinimumHeight(300)
        self.viz_layout.addWidget(self.skeleton_image_label, 1, 0, 1, 2)

        # Feature stats section
        self.stats_frame = QFrame()
        self.stats_frame.setStyleSheet("""
            QFrame {
                background: #3d3d3d;
                border-radius: 10px;
                padding: 15px;
                margin-top: 20px;
            }
        """)
        self.stats_layout = QVBoxLayout(self.stats_frame)

        stats_title = QLabel("Feature Statistics")
        stats_title.setStyleSheet("color: #ffffff; font-size: 16px; font-weight: bold;")
        stats_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stats_layout.addWidget(stats_title)

        self.stats_grid = QGridLayout()

        stat_labels = [
            ("File Name:", "file_name_value"),
            ("Object Label:", "object_label_value"),
            ("Area:", "area_value"),
            ("Perimeter:", "perimeter_value"),
            ("Circularity:", "circularity_value"),
            ("Roundness:", "roundness_value"),
            ("Number of Branches:", "branches_value"),
            ("Number of Projections:", "projections_value"),
            ("Total Skeleton Length:", "skeleton_length_value"),
            ("Fractal Dimension:", "fractal_dim_value")
        ]

        # Create labels for all stats
        self.stat_value_labels = {}

        for idx, (label_text, value_id) in enumerate(stat_labels):
            row = idx // 2
            col_start = (idx % 2) * 2

            label = QLabel(label_text)
            label.setStyleSheet("color: #cccccc; font-size: 13px;")

            value_label = QLabel("N/A")
            value_label.setStyleSheet("color: #ffffff; font-size: 13px; font-weight: bold;")
            self.stat_value_labels[value_id] = value_label

            self.stats_grid.addWidget(label, row, col_start)
            self.stats_grid.addWidget(value_label, row, col_start + 1)

        self.stats_layout.addLayout(self.stats_grid)
        self.viz_layout.addWidget(self.stats_frame, 2, 0, 1, 2)

        # Set the visualization display frame as the widget for the scroll area
        viz_scroll_area.setWidget(self.viz_display_frame)
        layout.addWidget(viz_scroll_area)

    def toggle_charts(self):
        self.charts_enabled = not self.charts_enabled
        self.charts_button.setText(f"Toggle Charts: {'ON' if self.charts_enabled else 'OFF'}")

    def visualize_random_feature(self):
        """Selects a random feature from the extracted features and displays it"""
        # Check if extracted features exist
        if not os.path.exists("extracted_features"):
            self.viz_status_label.setText("No extracted features found! Process images first.")
            return

        # Update UI
        self.viz_status_label.setText("Finding random feature...")
        self.viz_button.setEnabled(False)

        # Start the visualization thread
        self.visualization_thread = FeatureVisualizationThread()
        self.visualization_thread.progress_signal.connect(self.update_viz_status)
        self.visualization_thread.visualization_complete.connect(self.display_feature)
        self.visualization_thread.start()

    def update_viz_status(self, message):
        """Updates the visualization status label"""
        self.viz_status_label.setText(message)

    def display_feature(self, stats):
        """Displays the feature visualization and statistics"""
        # Re-enable the button
        self.viz_button.setEnabled(True)

        # Display the feature overlay image
        if os.path.exists(stats['feature_overlay_path']):
            pixmap = QPixmap(stats['feature_overlay_path'])
            scaled_pixmap = pixmap.scaled(
                min(400, self.feature_image_label.width()),
                min(400, self.feature_image_label.height()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.feature_image_label.setPixmap(scaled_pixmap)
            self.feature_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Display the original image
        if os.path.exists(stats['original_path']):
            pixmap = QPixmap(stats['original_path'])
            scaled_pixmap = pixmap.scaled(
                min(400, self.original_image_label.width()),
                min(400, self.original_image_label.height()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.original_image_label.setPixmap(scaled_pixmap)
            self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Display the skeleton overlay
        if os.path.exists(stats['skeleton_overlay_path']):
            pixmap = QPixmap(stats['skeleton_overlay_path'])
            scaled_pixmap = pixmap.scaled(
                min(800, self.skeleton_image_label.width()),
                min(400, self.skeleton_image_label.height()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.skeleton_image_label.setPixmap(scaled_pixmap)
            self.skeleton_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Update statistics
        self.stat_value_labels['file_name_value'].setText(str(stats['file_name']))
        self.stat_value_labels['object_label_value'].setText(str(stats['object_label']))
        self.stat_value_labels['area_value'].setText(str(stats['area']))
        self.stat_value_labels['perimeter_value'].setText(str(stats['perimeter']))
        self.stat_value_labels['circularity_value'].setText(str(stats['circularity']))
        self.stat_value_labels['roundness_value'].setText(str(stats['roundness']))
        self.stat_value_labels['branches_value'].setText(str(stats['num_branches']))
        self.stat_value_labels['projections_value'].setText(str(stats['num_projections']))
        self.stat_value_labels['skeleton_length_value'].setText(str(stats['total_skeleton_length']))
        self.stat_value_labels['fractal_dim_value'].setText(str(stats['fractal_dim']))

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
