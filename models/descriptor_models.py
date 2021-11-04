import cv2
cv2.setNumThreads(0)  # Making my own parallelisation so I have to disable OpenCV's
from joblib import delayed, Parallel
from matej.collections import flatten
from matej.parallel import tqdm_joblib
import numpy as np
from tqdm import trange

from . import RecognitionModel


class DescriptorModel(RecognitionModel, rgb=False):
	""" Models that compute hand-crafted features with descriptor-based methods. """

	def __init__(self, descriptor='sift', dense=False, *args, **kw):
		#pylint: disable=no-member
		self.descriptor = descriptor.lower()
		self.name = f'd{descriptor.upper()}' if dense else descriptor.upper()
		self.grid_step = dense

		self._args = args
		self._kw = kw
		self.alg = self._create_descriptor()
		self.matcher = cv2.BFMatcher()

	def _create_descriptor(self):
		if self.descriptor == 'sift':
			return cv2.xfeatures2d.SIFT_create(*self._args, **self._kw)
		elif self.descriptor == 'surf':
			return cv2.xfeatures2d.SURF_create(*self._args, **self._kw)
		elif self.descriptor == 'orb':
			return cv2.ORB_create(*self._args, **self._kw)
		else:
			raise ValueError(f"Unsupported descriptor algorithm {self.descriptor}.")

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
		with tqdm_joblib(trange(len(descriptors) - 1, desc=f"Computing {self.name} dist matrix rows", leave=kw.get('tqdm_leave_pbar', True))) as idx:
			upper_triangle = list(flatten(Parallel(n_jobs=-1)(
				delayed(self._dist_row)(descriptors, i)
				for i in idx
			)))
		matrix = np.zeros((len(descriptors), len(descriptors)))  # Diagonal will stay 0 (since those are self-comparisons)
		matrix[np.triu_indices_from(matrix, 1)] = matrix[np.tril_indices_from(matrix, -1)] = upper_triangle  # Fill both triangles (symmetric matrix)
		matrix[np.isnan(matrix)] = 1  # Replace NaNs with maximal distance
		return matrix

	def _dist_row(self, descriptors, i):
		return [self._dist(descriptors[i], des) for des in descriptors[i+1:]]

	def _dist(self, des1, des2):
		if des1 is None or len(des1) == 0 or des2 is None or len(des2) == 0:
			return np.nan
		matches = self.matcher.knnMatch(des1, des2, k=2)
		good = [m[0] for m in matches if len(m) > 1 and m[0].distance < .75 * m[1].distance]
		#return sum(m.distance for m in good) / len(good) if good else np.nan
		return 1 - len(good) / len(matches)

	# Pickling has to ignore the algorithms and matchers, since those can't be pickled
	def __getstate__(self):
		state = self.__dict__.copy()
		del state['alg']
		del state['matcher']
		return state

	def __setstate__(self, newstate):
		self.__dict__.update(newstate)
		self.alg = self._create_descriptor()
		self.matcher = cv2.BFMatcher()
