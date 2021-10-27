import cv2
import numpy as np

from . import RecognitionModel


class DescriptorModel(RecognitionModel):
	""" Models that compute hand-crafted features with descriptor-based methods. """

	def __init__(self, descriptor='sift', dense=False, *args, **kw):
		#pylint: disable=no-member
		if descriptor.lower() == 'sift':
			self.alg = cv2.xfeatures2d.SIFT_create(*args, **kw)
		elif descriptor.lower() == 'surf':
			self.alg = cv2.xfeatures2d.SURF_create(*args, **kw)
		elif descriptor.lower() == 'orb':
			self.alg = cv2.ORB_create(*args, **kw)
		else:
			raise ValueError(f"Unsupported descriptor algorithm {descriptor}.")

		self.grid_step = dense
		if self.grid_step is True:
			self.grid_step = int(np.sqrt(np.mean(self.image_size)))
		self.matcher = cv2.BFMatcher()

	def extract_features(self, img):
		# Detected keypoints
		if not self.grid_step:
			return self.alg.detectAndCompute(img, None)[1]

		# Dense grid keypoints
		return self.alg.compute(img, [
			cv2.KeyPoint(x, y, self.grid_step)
			for y in range(0, img.shape[0], self.grid_step)
			for x in range(0, img.shape[1], self.grid_step)
		])[1]

	def dist_matrix(self, descriptors):
		matrix = np.array([[self._dist(des1, des2) for des2 in descriptors] for des1 in descriptors])
		np.fill_diagonal(matrix, 0)
		matrix[np.isnan(matrix)] = 1
		return matrix

	def _dist(self, des1, des2):
		if des1 is None or len(des1) == 0 or des2 is None or len(des2) == 0:
			return np.nan
		matches = self.matcher.knnMatch(des1, des2, k=2)
		good = [m[0] for m in matches if len(m) > 1 and m[0].distance < .75 * m[1].distance]
		#return sum(m.distance for m in good) / len(good) if good else np.nan
		return 1 - len(good) / len(matches)
