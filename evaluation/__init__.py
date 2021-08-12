from abc import ABC, abstractmethod
from matej.collections import ensure_iterable
from matej.math import RunningStats
import re


VAR_RE = re.compile(r'\W+')


class Metric(RunningStats, ABC):
	def __init__(self, *args, **kw):
		super().__init__(type(self).__name__, *args, **kw)

	@abstractmethod
	def __call__(self, *args, **kw):
		pass

	def compute_and_update(self, *args, **kw):
		self.update(self(*args, **kw))

	def plot(self, plt, *args, **kw):
		raise NotImplementedError(f"Plotting is not supported for {self}.")


class Evaluation(dict):
	def __init__(self, metrics, *args):
		super().__init__((metric.name, metric) for metric in ensure_iterable(metrics) + args)

	def __str__(self):
		return "\n".join(map(str, self.metrics))

	def __delattr__(self, name):
		item_name = self._find(name)
		if item_name:
			del self[item_name]
		else:
			raise AttributeError(f"No attribute called: {name}")

	def __getattr__(self, name):
		item_name = self._find(name)
		if item_name:
			return self[item_name]
		else:
			raise AttributeError(f"No attribute called: {name}")

	def __setattr__(self, name, val):
		item_name = self._find(name)
		if item_name:
			self[item_name] = val
		else:
			self[name] = val

	def _find(self, name):
		if name in self:
			return name
		for metric in self:
			if name.lower() == VAR_RE.sub('', metric.lower()):
				return metric
		return None

	@property
	def metrics(self):
		return filter(lambda m: isinstance(m, Metric), self.values())


from .segmentation import BinarySegmentationEvaluation, SegmentationEvaluation
