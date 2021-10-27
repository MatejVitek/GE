import numpy as np
import sklearn.metrics as skmetrics

from evaluation import Evaluation, Metric


class BinarySegmentationEvaluation(Evaluation):
	def __init__(self, *args, **kw):
		super().__init__(IoU('binary', **kw), F(**kw), Precision(**kw), Recall(**kw), *args)


class BinaryIntensitySegmentationEvaluation(BinarySegmentationEvaluation):
	#TODO: For EyeZ probably some of the code from compute._segmentation_metrics_for_sample could be moved here (similar to VerificationMetric.error_rates_from_dist_matrix)
	def __init__(self, *args, **kw):
		super().__init__(AUC(**kw), *args)


class IoU(Metric):
	def __init__(self, average='macro', *args, **kw):
		super().__init__(*args, **kw)
		self.average = average

	def __call__(self, gt, predictions):
		return skmetrics.jaccard_score(gt, predictions, average=self.average, zero_division=0)


class F(Metric):
	def __init__(self, beta=1, *args, **kw):
		super().__init__(*args, **kw)
		self.name = f"F{beta}-score"
		self.beta = beta

	def __call__(self, gt=None, predictions=None, *, precision=None, recall=None):
		if precision is not None and recall is not None:
			# Edge case
			if precision == recall == 0:
				return 0
			return (1 + self.beta ** 2) * precision * recall / (self.beta ** 2 * precision + recall)
		else:
			return skmetrics.fbeta_score(gt, predictions, beta=self.beta, zero_division=0)


class Precision(Metric):
	def __call__(self, *args, **kw):
		return skmetrics.precision_score(*args, zero_division=0, **kw)


class Recall(Metric):
	def __call__(self, *args, **kw):
		return skmetrics.recall_score(*args, zero_division=0, **kw)


class AUC(Metric):
	def __call__(self, gt=None, predictions=None, *, precisions=None, recalls=None):
		if precisions is not None and recalls is not None:
			if len(precisions) < 2 or len(recalls) < 2:
				return 0
			return skmetrics.auc(recalls, precisions)
		else:
			if not np.any(predictions):
				return 0
			return skmetrics.roc_auc_score(gt, predictions)
