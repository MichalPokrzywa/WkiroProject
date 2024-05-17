# imgLoader.py

import os
import cv2
import numpy as np

def load_images_from_folder(folder):
    images = []
    try:
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if os.path.isfile(path):
                img = cv2.imread(path)
                if img is not None:
                    images.append(img)
    except Exception as e:
        print(f"Error loading images: {e}")
    return images


def images_to_pixels(images):
    pixels_list = []
    try:
        for img in images:
            pixels = np.array(img)
            pixels_list.append(pixels)
    except Exception as e:
        print(f"Error converting images to pixels: {e}")
    return pixels_list

def image_names_list(folder):
    image_names = []
    try:
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if os.path.isfile(path) and any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                image_names.append(filename)
    except Exception as e:
        print(f"Error loading image names: {e}")
    return image_names

