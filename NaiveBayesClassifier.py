import cv2
import numpy as np

class NaiveBayesClassifier:
    def __init__(self, images, images_skin, images_test):
        self.images = [cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) for img in images]
        self.images_skin = images_skin
        self.images_test = images_test
        self.skin_pixels = None
        self.non_skin_pixels = None
        self.skin_hist = None
        self.non_skin_hist = None
        self.skin_edges = None
        self.non_skin_edges = None
        self.p_non_skin = None
        self.p_skin = None

    def separate_skin_pixels(self):
        skin_pixels = []
        non_skin_pixels = []

        for original_image_np, skin_mask_np in zip(self.images, self.images_skin):
            #original_image_np = np.array(original_image)
            #skin_mask_np = np.array(skin_mask)

            mask_skin = skin_mask_np[:, :, 0] != 255
            mask_non_skin = skin_mask_np[:, :, 0] == 255

            skin_pixels.append(original_image_np[mask_skin])
            non_skin_pixels.append(original_image_np[mask_non_skin])

        skin_pixels = np.vstack(skin_pixels)
        non_skin_pixels = np.vstack(non_skin_pixels)

        return skin_pixels, non_skin_pixels

    def estimate_pdf(self, pixels, bins=32):
        hist, edges = np.histogramdd(pixels, bins=bins, density=True)
        hist /= np.sum(hist)  # Normalize histogram
        return hist, edges

    def find_bin_index(self, value, edges):
        return tuple(np.digitize(value[i], edges[i]) - 1 for i in range(3))

    def class_conditional_prob(self, rgb, hist, edges):
        bin_index = self.find_bin_index(rgb, edges)
        if all(0 <= idx < hist.shape[i] for i, idx in enumerate(bin_index)):
            return hist[bin_index]
        else:
            return 1e-6  # Small non-zero probability to avoid division by zero

    def posterior_prob(self, rgb):
        p_rgb_given_skin = self.class_conditional_prob(rgb, self.skin_hist, self.skin_edges)
        p_rgb_given_non_skin = self.class_conditional_prob(rgb, self.non_skin_hist, self.non_skin_edges)

        p_rgb = p_rgb_given_skin * self.p_skin + p_rgb_given_non_skin * self.p_non_skin

        if p_rgb == 0:  # Avoid division by zero
            return 0

        p_skin_given_rgb = (p_rgb_given_skin * self.p_skin) / p_rgb
        return p_skin_given_rgb

    def classify_image(self, image, threshold=0.5):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        skin_mask = np.zeros(image.shape[:2], dtype=bool)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rgb = image[i, j]
                if self.posterior_prob(rgb) > threshold:
                    skin_mask[i, j] = True
        return skin_mask

    def create_skin_map(self):
        self.skin_pixels, self.non_skin_pixels = self.separate_skin_pixels()
        self.skin_hist, self.skin_edges = self.estimate_pdf(self.skin_pixels)
        self.non_skin_hist, self.non_skin_edges = self.estimate_pdf(self.non_skin_pixels)

        self.p_skin = len(self.skin_pixels) / (len(self.skin_pixels) + len(self.non_skin_pixels))
        self.p_non_skin = len(self.non_skin_pixels) / (len(self.skin_pixels) + len(self.non_skin_pixels))

        # Example usage
        skin_mask = self.classify_image(self.images_test[0])

        # Converting boolean mask to uint8 image for display
        skin_mask_display = (skin_mask * 255).astype(np.uint8)

        # Display results using OpenCV
        cv2.imshow('Original Image', self.images_test[0])
        cv2.imshow('Skin Detection Mask', skin_mask_display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
