import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import time
def load_images_from_folder(folder, new_width, new_height):
    images = []
    start_time = time.time()  # Start timing
    try:
        filenames = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            results = executor.map(lambda f: process_image(os.path.join(folder, f), new_width, new_height), filenames)
            for result in results:
                if result is not None:
                    images.append(result)
    except Exception as e:
        print(f"Error loading images: {e}")
    end_time = time.time()  # End timing
    print(f"Time taken to load images: {end_time - start_time:.2f} seconds")
    return images

def process_image(path, new_width, new_height):
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        img_resize = cv2.resize(img, (new_width, new_height))
        #img_normalize = normalize_image(img_resize)
        return img_resize
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        return None

def images_to_pixels(images):
    try:
        return [np.array(img) for img in images]
    except Exception as e:
        print(f"Error converting images to pixels: {e}")
        return []

def normalize_image(image):
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycbcr_image)
    Y_eq = cv2.equalizeHist(Y)
    ycbcr_image_eq = cv2.merge([Y_eq, Cr, Cb])
    normalized_image = cv2.cvtColor(ycbcr_image_eq, cv2.COLOR_YCrCb2BGR)
    return normalized_image

def normalize_pixel_values(image):
    return image / 255.0
