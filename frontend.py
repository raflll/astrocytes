from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
import sys

class MyWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # Create a label
        self.label = QLabel("Hello! Click the button below.", self)

        # Create a button
        self.button = QPushButton("Open File", self)
        self.button.clicked.connect(self.open_file_dialog)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.button)

        self.setLayout(layout)
        self.setWindowTitle("PyQt6 Frontend")
        self.resize(400, 200)

    def open_file_dialog(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open File", "", "All Files (*)")
        if file_name:
            self.label.setText(f"Selected File:\n{file_name}")

app = QApplication(sys.argv)
window = MyWindow()
window.show()
sys.exit(app.exec())
