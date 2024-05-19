# front.py

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QLabel, QVBoxLayout, QGridLayout
from PyQt5 import  QtCore, QtGui
from PyQt5.QtWidgets import QLineEdit, QMessageBox
from imgLoader import load_images_from_folder, images_to_pixels
from PyQt5.QtGui import QPixmap

class FrontApp(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Folder Selection'
        self.folder_selected=None

    def initUI(self):
        self.setWindowTitle(self.title)
        self.resize(800,600)
        
        layout = QVBoxLayout()

        self.select_button = QPushButton('Select Folder', self)
        self.select_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.select_button)

        self.process_button = QPushButton('Process Images', self)
        self.process_button.clicked.connect(self.process_images)
        layout.addWidget(self.process_button)

        self.width_input = QLineEdit(self)
        self.width_input.setPlaceholderText('Enter width')
        layout.addWidget(self.width_input)

        self.height_input = QLineEdit(self)
        self.height_input.setPlaceholderText('Enter height')
        layout.addWidget(self.height_input)

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        self.show()
        

    def open_file_dialog(self):
        self.folder_selected = QFileDialog.getExistingDirectory(self, "Select Folder")
        if self.folder_selected:
            print(f"Selected folder: {self.folder_selected}")

    def process_images(self):
        if not self.folder_selected:
            QMessageBox.warning(self, "Warning", "Select a folder first.")
            return

        try:
            new_width = int(self.width_input.text())
            new_height = int(self.height_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid width and height.")
            return
        
        print(new_height, new_width)
        images = load_images_from_folder(self.folder_selected, new_width, new_height)
        pixels = images_to_pixels(images)

        # Wyświetlenie rozmiaru każdej tablicy pikseli
        for i, pixels_array in enumerate(pixels):
                print(f"Rozmiar tablicy pikseli {i + 1}: {pixels_array.shape}")
        else:
            print("Select a folder first.")

        QMessageBox.information(self, "Info", "Images processed successfully.")

    if __name__ == '__main__':
        app = QApplication(sys.argv)
        ex = FrontApp()
        sys.exit(app.exec_())