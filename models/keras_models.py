from keras.models import load_model
import numpy as np
import os

from . import RecognitionModel


class KerasModel(RecognitionModel):
	def __init__(self, model_file):
		self.model = load_model(model_file)

	def extract_features(self, images):
		return self.model.predict(np.stack(images), workers=len(os.sched_getaffinity(0)))  # This won't work on Windows, need a different way of determining available CPUs
