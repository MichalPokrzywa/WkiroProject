import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.skin_pdf = None
        self.non_skin_pdf = None

    def train(self, images, skin_masks):
        skin_pixels = []
        non_skin_pixels = []

        for image, mask in zip(images, skin_masks):
            skin_pixels.extend(image[mask > 0])
            non_skin_pixels.extend(image[mask == 0])

        self.skin_pdf = self._compute_pdf(skin_pixels)
        self.non_skin_pdf = self._compute_pdf(non_skin_pixels)

    def _compute_pdf(self, pixels):
        pdf, _ = np.histogramdd(pixels, bins=256, range=((0, 255), (0, 255), (0, 255)), density=True)
        return pdf

    def classify(self, pixel):
        skin_prob = self._get_probability(pixel, self.skin_pdf)
        non_skin_prob = self._get_probability(pixel, self.non_skin_pdf)
        return skin_prob > non_skin_prob

    def _get_probability(self, pixel, pdf):
        r, g, b = pixel
        return pdf[r, g, b]

    def test(self, test_images):
        results = []
        for image in test_images:
            classified_image = np.zeros(image.shape[:2])
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    if self.classify(image[i, j]):
                        classified_image[i, j] = 1
            results.append(classified_image)
        return results
