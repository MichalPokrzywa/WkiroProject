import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
import time
from PIL import Image
import os

class NaiveBayesClassifier:
    def __init__(self, images, images_skin, images_test, images_test_mask):
        self.images = [cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) for img in images]
        self.images_skin = images_skin
        self.images_test = images_test
        self.images_test_mask = images_test_mask
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

        # Vectorized operation
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rgb = image[i, j]
                if self.posterior_prob(rgb) > threshold:
                    skin_mask[i, j] = True

        return skin_mask

    def create_skin_map(self, bins=32):
        self.skin_pixels, self.non_skin_pixels = self.separate_skin_pixels()
        self.skin_hist, self.skin_edges = self.estimate_pdf(self.skin_pixels, bins=bins)
        self.non_skin_hist, self.non_skin_edges = self.estimate_pdf(self.non_skin_pixels, bins=bins)

        self.p_skin = len(self.skin_pixels) / (len(self.skin_pixels) + len(self.non_skin_pixels))
        self.p_non_skin = len(self.non_skin_pixels) / (len(self.skin_pixels) + len(self.non_skin_pixels))

        # Initialize metrics
        all_true_labels = []
        all_pred_labels = []

        # Iterate over all test images and calculate metrics
        for test_image, test_mask in zip(self.images_test, self.images_test_mask):
            skin_mask = self.classify_image(test_image)

            # Flatten the masks for metric calculation
            true_labels = test_mask[:, :, 0].flatten() != 255  # True skin mask
            pred_labels = skin_mask.flatten()  # Predicted skin mask

            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)

        # Calculate metrics
        precision = precision_score(all_true_labels, all_pred_labels)
        recall = recall_score(all_true_labels, all_pred_labels)
        f1 = f1_score(all_true_labels, all_pred_labels)

        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

        return precision, recall, f1, skin_mask


def evaluate_parameters(images, images_skin, images_test, images_test_mask, image_amounts, bin_sizes):
    image_amount_results = {}
    bin_size_results = {}

    # Iterate over the number of images to use
    for amount in image_amounts:
        print(f"Evaluating with {amount} images...")
        image_amount_results[amount] = {}

        # Calculate the timestamp before the loops
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        save_path = "./results/"
        save_directory=f"{save_path}/{timestamp}/"
        os.makedirs(save_directory, exist_ok=True)
        for bins in bin_sizes:
            print(f"Evaluating with {bins} histogram bins...")
            classifier = NaiveBayesClassifier(images[:amount], images_skin[:amount], images_test, images_test_mask)
            precision, recall, f1, gen_skin = classifier.create_skin_map(bins=bins)

            # Save the generated skin map as an image
            skin_gen_amount_bins = Image.fromarray((gen_skin * 255).astype(np.uint8))
            save_filename = f"{save_directory}gen_skin_{amount}_images_{bins}_bins.png"
            skin_gen_amount_bins.save(save_filename)

            image_amount_results[amount][bins] = (precision, recall, f1)
            if bins not in bin_size_results:
                bin_size_results[bins] = {}
            bin_size_results[bins][amount] = (precision, recall, f1)

    return image_amount_results, bin_size_results


import matplotlib.pyplot as plt
import numpy as np

def plot_results(image_amount_results, bin_size_results):
    # Plot for image amounts
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    for bins, results in bin_size_results.items():
        image_amounts = list(results.keys())
        detection_errors = [1 - results[amount][2] for amount in image_amounts]  # Detection error
        plt.plot(image_amounts, detection_errors, label=f'{bins} bins', marker='o')
    plt.xlabel('Number of Images')
    plt.ylabel('Detection Error')
    plt.title('Detection Error vs Number of Images')
    plt.legend()
    plt.grid(True)

    # Plot for bin sizes
    plt.subplot(1, 2, 2)
    bin_sizes = sorted(next(iter(image_amount_results.values())).keys())
    num_groups = len(image_amount_results)
    bar_width = 0.8 / num_groups  # Adjust bar width to ensure space between groups
    offsets = np.linspace(-0.4, 0.4, num_groups)  # Center the bars around each bin size

    for i, (amount, results) in enumerate(image_amount_results.items()):
        detection_errors = [1 - results[bins][2] for bins in bin_sizes]  # Detection error
        bar_positions = np.arange(len(bin_sizes)) + offsets[i]
        plt.bar(bar_positions, detection_errors, width=bar_width, label=f'{amount} images')

    plt.xlabel('Histogram Bin Size')
    plt.ylabel('Detection Error')
    plt.title('Detection Error vs Histogram Bin Size')
    plt.xticks(np.arange(len(bin_sizes)), bin_sizes)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Example usage:
# image_amount_results = {50: {10: [0, 0, 0.1], 20: [0, 0, 0.15], 30: [0, 0, 0.2]}, 100: {10: [0, 0, 0.05], 20: [0, 0, 0.1], 30: [0, 0, 0.15]}}
# bin_size_results = {10: {50: [0, 0, 0.1], 100: [0, 0, 0.05]}, 20: {50: [0, 0, 0.15], 100: [0, 0, 0.1]}, 30: {50: [0, 0, 0.2], 100: [0, 0, 0.15]}}
# plot_results(image_amount_results, bin_size_results)


# Example usage
# Load images, images_skin, images_test, and images_test_mask before this
# image_amounts = [10, 20, 30]  # Example values for different number of images to use
# bin_sizes = [16, 32, 64]      # Example values for different histogram bin sizes
# image_amount_results, bin_size_results = evaluate_parameters(images, images_skin, images_test, images_test_mask, image_amounts, bin_sizes)
# plot_results(image_amount_results, bin_size_results)
