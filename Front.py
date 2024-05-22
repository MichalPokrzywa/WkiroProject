import sys
from PyQt5.QtCore import QRect
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QLabel, QLineEdit,
                             QMessageBox)

from imgLoader import load_images_from_folder, images_to_pixels
from NaiveBayesClassifier import NaiveBayesClassifier, evaluate_parameters, plot_results


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
        self.folder_selected_test_mask = None
        self.images_test_mask = None
        self.to_resize = False
        self.scale = 1
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.resize(800, 600)

        layout = QVBoxLayout()

        self.select_button = QPushButton('Select Folders', self)
        self.select_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(self.select_button)

        self.scale_input = QLineEdit(self)
        self.scale_input.setPlaceholderText('Enter size image multiplier width')
        self.scale_input.setText('1')
        self.scale_input.editingFinished.connect(self.on_text_changed)
        self.scale_input.textChanged.connect(self.on_text_changed)  # Connect textChanged signal as well
        layout.addWidget(self.scale_input)

        self.image_amounts_input = QLineEdit(self)
        self.image_amounts_input.setPlaceholderText('Enter image amounts (comma-separated)')
        layout.addWidget(self.image_amounts_input)

        self.bin_sizes_input = QLineEdit(self)
        self.bin_sizes_input.setPlaceholderText('Enter histogram bin sizes (comma-separated)')
        layout.addWidget(self.bin_sizes_input)

        self.button = QPushButton("Use Resize", self)
        self.button.setCheckable(True)
        self.button.clicked.connect(self.changeColor)
        layout.addWidget(self.button)

        self.process_button = QPushButton('Process Images', self)
        self.process_button.clicked.connect(self.process_images)
        layout.addWidget(self.process_button)

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.setLayout(layout)
        self.show()

    def changeColor(self):
        if self.button.isChecked():
            self.button.setStyleSheet("background-color : lightblue")
            self.to_resize = True
        else:
            self.button.setStyleSheet("background-color : lightgrey")
            self.to_resize = False

    def on_text_changed(self):
        try:
            self.scale = float(self.scale_input.text())
            print(f'Text changed to: {self.scale}')
        except ValueError:
            print(f'Invalid input for scale: {self.scale_input.text()}')
            # Optionally reset to previous valid value or default
            self.scale = 1
            #self.scale_input.setText(str(self.scale))

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

        self.folder_selected_test_mask = QFileDialog.getExistingDirectory(self, "Select Test Mask Folder")
        if self.folder_selected_test_mask:
            print(f"Selected test mask folder: {self.folder_selected_test_mask}")

    def process_images(self):
        if not self.folder_selected or not self.folder_selected_skin or not self.folder_selected_test or not self.folder_selected_test_mask:
            QMessageBox.warning(self, "Warning", "Select all folders first.")
            self.folder_selected = "./Photos/Original/001"
            self.folder_selected_skin = "./Photos/Skin/001"
            self.folder_selected_test = "./Photos/Original/003"
            self.folder_selected_test_mask = "./Photos/Skin/003"

        try:
            self.scale = float(self.scale_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid width and height.")
            return

        try:
            image_amounts = [int(s.strip()) for s in self.image_amounts_input.text().split(',')]
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid image amounts.")
            image_amounts = [10, 20, 40, 80, 160, 320, 640, 1280]

        try:
            bin_sizes = [int(s.strip()) for s in self.bin_sizes_input.text().split(',')]
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter valid histogram bin sizes.")
            bin_sizes = [8, 16, 32, 64, 128, 256]

        print(f"Scale: {self.scale}")
        self.images = load_images_from_folder(self.folder_selected, self.scale, self.to_resize)
        self.images_skin = load_images_from_folder(self.folder_selected_skin, self.scale, self.to_resize)
        self.images_test = load_images_from_folder(self.folder_selected_test, self.scale, self.to_resize)
        self.images_test_mask = load_images_from_folder(self.folder_selected_test_mask, self.scale, self.to_resize)

        classifier = NaiveBayesClassifier(self.images, self.images_skin, self.images_test, self.images_test_mask)
        image_amount_results, bin_size_results = evaluate_parameters(
            self.images, self.images_skin, self.images_test, self.images_test_mask, image_amounts, bin_sizes)

        plot_results(image_amount_results, bin_size_results)

        QMessageBox.information(self, "Info", "Images processed and results plotted successfully.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FrontApp()
    sys.exit(app.exec_())
