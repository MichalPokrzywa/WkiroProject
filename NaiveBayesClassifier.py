import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


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

    def classify_image(self, image, threshold=0.9):
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

        return precision, recall, f1


def evaluate_parameters(images, images_skin, images_test, images_test_mask, image_amounts, bin_sizes):
    image_amount_results = {}
    bin_size_results = {}

    # Iterate over the number of images to use
    for amount in image_amounts:
        print(f"Evaluating with {amount} images...")
        image_amount_results[amount] = {}

        for bins in bin_sizes:
            print(f"Evaluating with {bins} histogram bins...")
            classifier = NaiveBayesClassifier(images[:amount], images_skin[:amount], images_test, images_test_mask)
            precision, recall, f1 = classifier.create_skin_map(bins=bins)

            image_amount_results[amount][bins] = (precision, recall, f1)
            if bins not in bin_size_results:
                bin_size_results[bins] = {}
            bin_size_results[bins][amount] = (precision, recall, f1)

    return image_amount_results, bin_size_results


def plot_results(image_amount_results, bin_size_results):
    # Plot for image amounts
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    for bins, results in bin_size_results.items():
        image_amounts = list(results.keys())
        #detection_errors = [1 - results[amount][2] for amount in image_amounts]  # Detection error = 1 - F1 Score
        detection_errors = [results[amount][2] for amount in image_amounts]  # Detection error = 1 - F1 Score
        plt.plot(image_amounts, detection_errors, label=f'{bins} bins', marker='o')
    plt.xlabel('Number of Images')
    plt.ylabel('Detection Error')
    plt.title('Detection Error vs Number of Images')
    plt.legend()
    plt.grid(True)

    # Plot for bin sizes
    plt.subplot(1, 2, 2)
    for amount, results in image_amount_results.items():
        bin_sizes = list(results.keys())
        #detection_errors = [1 - results[bins][2] for bins in bin_sizes]  # Detection error = 1 - F1 Score
        detection_errors = [results[bins][2] for bins in bin_sizes]  # Detection error = 1 - F1 Score

        plt.plot(bin_sizes, detection_errors, label=f'{amount} images', marker='o')
    plt.xlabel('Histogram Bin Size')
    plt.ylabel('Detection Error')
    plt.title('Detection Error vs Histogram Bin Size')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Example usage
# Load images, images_skin, images_test, and images_test_mask before this
# image_amounts = [10, 20, 30]  # Example values for different number of images to use
# bin_sizes = [16, 32, 64]      # Example values for different histogram bin sizes
# image_amount_results, bin_size_results = evaluate_parameters(images, images_skin, images_test, images_test_mask, image_amounts, bin_sizes)
# plot_results(image_amount_results, bin_size_results)
