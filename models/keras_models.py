from keras.models import load_model
import numpy as np
import os

from . import RecognitionModel


class KerasModel(RecognitionModel, supports_pickling=False):
	def __init__(self, model_file):
		self.model = load_model(model_file)

	def extract_features(self, images, **kw):
		return self.model.predict(np.stack(images), workers=len(os.sched_getaffinity(0)), **kw)  # This won't work on Windows, need a different way of determining available CPUs
