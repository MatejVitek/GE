from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist


class RecognitionModel(ABC):
	@abstractmethod
	def extract_features(self, img):
		"""
		Process input image(s) and return the corresponding feature vector(s)

		:param img: Input image(s)

		:return Feature vector(s) as numpy array(s)
		"""

	def dist_matrix(self, feature_vectors, metric='cosine'):
		return cdist(feature_vectors, feature_vectors, metric=metric)
