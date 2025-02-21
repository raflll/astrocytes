from frontend import *

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FolderSelector()
    window.show()
    sys.exit(app.exec())
