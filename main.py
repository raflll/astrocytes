from frontend import *

if __name__ == "__main__":

    GUI = False
    # If GUI is enabled, it will run the front end
    if GUI:
        app = QApplication(sys.argv)
        ex = ModernUI()
        ex.show()
        sys.exit(app.exec())

    # Used for testing
    else:
        # process_directory("data")
        # train_model("ENet", True)
        charts(True)
