import cv2
from matej.collections import lzip
import numpy as np
from tqdm import tqdm

from . import RecognitionModel


class DescriptorModel(RecognitionModel, rgb=False):
	""" Models that compute hand-crafted features with descriptor-based methods. """

	def __init__(self, descriptor='sift', dense=False, *args, **kw):
		self.grid_step = dense
		self.name = f'd{descriptor.upper()}' if dense else descriptor.upper()

		if descriptor.lower() == 'sift':
			self.alg = cv2.xfeatures2d.SIFT_create(*args, **kw)
		elif descriptor.lower() == 'surf':
			self.alg = cv2.xfeatures2d.SURF_create(*args, **kw)
		elif descriptor.lower() == 'orb':
			self.alg = cv2.ORB_create(*args, **kw)
		else:
			raise ValueError(f"Unsupported descriptor algorithm {descriptor}.")

		self.matcher = cv2.BFMatcher()

	def extract_features(self, img):
		step = int(np.sqrt(np.mean(img.shape))) if self.grid_step is True else self.grid_step

		# Detected keypoints
		if not step:
			return self.alg.detectAndCompute(img, None)[1]

		# Dense grid keypoints
		return self.alg.compute(img, [
			cv2.KeyPoint(x, y, step)
			for y in range(0, img.shape[0], step)
			for x in range(0, img.shape[1], step)
		])[1]

	def dist_matrix(self, descriptors, **kw):
		matrix = np.zeros((len(descriptors), len(descriptors)))  # Diagonal will stay 0 (since those are self-comparisons)
		idx = np.triu_indices_from(matrix, 1)  # Indices for upper triangle to be filled with distances
		matrix[idx] = [  # Compute distances and fill upper triangle of distance matrix
			self._dist(descriptors[i], descriptors[j])
			for i, j in tqdm(lzip(*idx), desc=f"Computing {self.name} dist matrix", leave=kw.get('tqdm_leave_pbar', True))
		]
		matrix[np.isnan(matrix)] = 1  # Replace NaNs with maximal distance
		matrix += matrix.T  # Copy upper triangle to lower triangle (since diagonal is 0, we don't need to worry about it))
		return matrix

	def _dist(self, des1, des2):
		if des1 is None or len(des1) == 0 or des2 is None or len(des2) == 0:
			return np.nan
		matches = self.matcher.knnMatch(des1, des2, k=2)
		good = [m[0] for m in matches if len(m) > 1 and m[0].distance < .75 * m[1].distance]
		#return sum(m.distance for m in good) / len(good) if good else np.nan
		return 1 - len(good) / len(matches)
