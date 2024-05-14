import os
import cv2 
import numpy as np 

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
    return images

def images_to_pixels(images):
    pixels_list = []
    for img in images:
        # Przekształć obraz na tablicę pikseli
        pixels = np.array(img)
        pixels_list.append(pixels)
    return pixels_list

folder_path = r"C:\Users\wojte\studia\wikro\WkiroProject\Photos\Original\001"
images = load_images_from_folder(folder_path)
pixels = images_to_pixels(images)

# Wyświetlenie rozmiaru każdej tablicy pikseli
for i, pixels_array in enumerate(pixels):
    print(f"Rozmiar tablicy pikseli {i + 1}: {pixels_array.shape}")
