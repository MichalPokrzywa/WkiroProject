# imgLoader.py

import os
import cv2
import numpy as np
from PIL import Image

def load_images_from_folder(folder, new_width, new_height):
    images = []
    try:
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if os.path.isfile(path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                img = cv2.imread(path)
                img_resize = cv2.resize(img, (new_width, new_height))
                img_normalize = normalize_image(img_resize)
                if img_normalize is not None:
                    images.append(img_normalize)
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

def normalize_image(image):
    # Konwersja obrazu do przestrzeni YCbCr
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Rozdzielenie kanałów
    Y, Cr, Cb = cv2.split(ycbcr_image)
    
    # Równoważenie histogramu na kanale luminancji (Y)
    Y_eq = cv2.equalizeHist(Y)
    
    # Połączenie z powrotem do przestrzeni YCbCr
    ycbcr_image_eq = cv2.merge([Y_eq, Cr, Cb])
    
    # Konwersja z powrotem do przestrzeni RGB
    normalized_image = cv2.cvtColor(ycbcr_image_eq, cv2.COLOR_YCrCb2BGR)
    
    return normalized_image

def normalize_pixel_values(image):
    # Normalizacja wartości pikseli do przedziału [0, 1]
    normalized_image = image / 255.0
    return normalized_image
