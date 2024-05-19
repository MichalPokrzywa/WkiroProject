import sys
from Front import FrontApp
from PyQt5.QtWidgets import QApplication
from PyQt5 import QtWidgets


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app_instance = FrontApp()
    app_instance.initUI()
    sys.exit(app.exec_())