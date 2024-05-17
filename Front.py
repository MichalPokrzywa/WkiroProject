# front.py

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QLabel, QVBoxLayout, QGridLayout
from PyQt5 import  QtCore, QtGui

from imgLoader import load_images_from_folder, images_to_pixels, image_names_list
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

        # Dodaj etykietę do wyświetlania obrazów
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        self.show()
        

    def open_file_dialog(self):
        self.folder_selected = QFileDialog.getExistingDirectory(self, "Select Folder")
        if self.folder_selected:
            print(f"Selected folder: {self.folder_selected}")
            image_names = image_names_list(self.folder_selected)
            '''
            for i, path in enumerate(image_names):
                pixmap = QPixmap(path)
                print(path)
                self.image_label.setPixmap(pixmap)
                '''
            pixmap = QPixmap(QtGui.QPixmap(image_names[0]))
            print(image_names[0])
            self.image_label.setPixmap(pixmap)

    def process_images(self):
        if hasattr(self, 'folder_selected'):
            images = load_images_from_folder(self.folder_selected)
            pixels = images_to_pixels(images)

            # Wyświetlenie rozmiaru każdej tablicy pikseli
            for i, pixels_array in enumerate(pixels):
                print(f"Rozmiar tablicy pikseli {i + 1}: {pixels_array.shape}")
        else:
            print("Select a folder first.")
