# front.py

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QLabel, QVBoxLayout, \
    QGridLayout, QLineEdit, QMessageBox, QComboBox
from PyQt5.QtGui import QPixmap
from imgLoader import load_images_from_folder, images_to_pixels

from NaiveBayesClassifier import *

class FrontApp(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Folder Selection'
        self.folder_selected = None
        self.images = None
        self.folder_selected_skin = None
        self.images_skin = None
        self.folder_selected_test = None
        self.images_test = None
        self.label = None

    def initUI(self):
        self.setWindowTitle(self.title)
        self.resize(800, 600)

        layout = QVBoxLayout()

        self.select_button = QPushButton('Select Folder', self)
        self.select_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.select_button)

        self.process_button = QPushButton('Process Images', self)
        self.process_button.clicked.connect(self.process_images)
        layout.addWidget(self.process_button)

        self.width_input = QLineEdit(self)
        self.width_input.setPlaceholderText('Enter width')
        self.width_input.setText('100')
        layout.addWidget(self.width_input)

        self.height_input = QLineEdit(self)
        self.height_input.setPlaceholderText('Enter height')
        self.height_input.setText('100')
        layout.addWidget(self.height_input)

        self.label = QLabel('Select an item from the dropdown', self)
        layout.addWidget(self.label)
        self.comboBox = QComboBox(self)
        self.comboBox.addItems(['Item 1', 'Item 2', 'Item 3', 'Item 4'])
        self.comboBox.currentIndexChanged.connect(self.on_selection_change)

        layout.addWidget(self.comboBox)
        #self.process_button = QPushButton('Create skin map', self)
        #self.process_button.clicked.connect(self.create_skin_map)
        #layout.addWidget(self.process_button)

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.setLayout(layout)

        self.show()

    def open_file_dialog(self):
        self.folder_selected = QFileDialog.getExistingDirectory(self, "Select Original Folder")
        if self.folder_selected:
            print(f"Selected original folder: {self.folder_selected}")

        self.folder_selected_skin = QFileDialog.getExistingDirectory(self, "Select Skin Folder")
        if self.folder_selected_skin:
            print(f"Selected skin folder: {self.folder_selected_skin}")

        self.folder_selected_test = QFileDialog.getExistingDirectory(self, "Select Test Folder")
        if self.folder_selected_test:
            print(f"Selected test folder: {self.folder_selected_test}")

    def on_selection_change(self, i):
        # Get the selected item text
        selected_item = self.comboBox.currentText()
        self.label.setText(f'Selected: {selected_item}')
    def process_images(self):
        if not self.folder_selected:
            QMessageBox.warning(self, "Warning", "Select original folder first.")
            return

        if not self.folder_selected_skin:
            QMessageBox.warning(self, "Warning", "Select skin folder first.")
            return

        if not self.folder_selected_test:
            QMessageBox.warning(self, "Warning", "Select test folder first.")
            return

        try:
            new_width = int(self.width_input.text())
            new_height = int(self.height_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid width and height.")
            return

        print(new_height, new_width)
        self.images = load_images_from_folder(self.folder_selected, new_width, new_height)
        self.pixels = images_to_pixels(self.images)

        self.images_skin = load_images_from_folder(self.folder_selected_skin, new_width, new_height)
        #self.pixels_skin = images_to_pixels(self.images_skin)

        self.images_test = load_images_from_folder(self.folder_selected_test, new_width, new_height)
        #self.pixels_test = images_to_pixels(self.images_test)

        # Wyświetlenie rozmiaru każdej tablicy pikseli
        for i, pixels_array in enumerate(self.pixels):
            print(f"Rozmiar tablicy pikseli {i + 1}: {pixels_array.shape}")
        else:
            print("Select a folder first.")

        classifier = NaiveBayesClassifier(self.images, self.images_skin, self.images_test)
        classifier.create_skin_map()

        QMessageBox.information(self, "Info", "Images processed successfully.")



    if __name__ == '__main__':
        app = QApplication(sys.argv)
        ex = FrontApp()
        sys.exit(app.exec_())