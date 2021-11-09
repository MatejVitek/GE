from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist


class RecognitionModel(ABC):
	@abstractmethod
	def extract_features(self, img, **kw):
		"""
		Process input image(s) and return the corresponding feature vector(s)

		:param img: Input image(s)

		:return Feature vector(s) as numpy array(s)
		"""

	def dist_matrix(self, feature_vectors, **kw):
		return cdist(feature_vectors, feature_vectors, metric=kw.get('metric', 'cosine'))

	def __init_subclass__(cls, rgb=True):
		cls.accepts_rgb_input = rgb
