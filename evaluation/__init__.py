#TODO: For EyeZ project, Metrics should be separated from RunningStats - they should only contain the computation stuff (or should they? The current setup makes it easy to store and print results. I should implement +-*/... operators, though, that work with the .mean attribute)
# New idea: Have a Metric class that only defines the computation. Then subclass it with two subclasses: a) MetricVar class that (computes and) saves all values at initiation (and doesn't allow updates) as a numpy array and b) RunningMetric class that works like the current Metric class.
from abc import ABC, abstractmethod
import itertools as it
from matej.collections import ensure_iterable
from matej.math import RunningStats
import numpy as np
import re


VAR_RE = re.compile(r'\W+')


class Metric(RunningStats, ABC):
	def __init__(self, *args, **kw):
		super().__init__(type(self).__name__, *args, **kw)

	@abstractmethod
	def __call__(self, *args, **kw):
		""" Override this method in subclasses with the actual computation of the metric. """

	def compute_and_update(self, *args, **kw):
		result = self(*args, **kw)
		self.update(result)
		return result


class Evaluation(dict):
	def __init__(self, metrics, *args):
		super().__init__((metric.name, metric) for metric in it.chain(ensure_iterable(metrics), args))

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
		for metric in self.keys():
			if name.lower() == VAR_RE.sub('', metric.lower()):
				return metric
		return None

	def keys(self):
		return filter(lambda k: isinstance(self[k], Metric), super().keys())

	def values(self):
		return filter(lambda v: isinstance(v, Metric), super().values())

	def items(self):
		return filter(lambda i: isinstance(i[1], Metric), super().items())

	def __iter__(self):
		yield from self.values()

	def means(self):
		return {metric.name: metric.mean for metric in self}

	def vars(self):
		return {metric.name: metric.var for metric in self}

	def stds(self):
		return {metric.name: metric.std for metric in self}

	def as_arrays(self, n='all'):
		return {metric.name: np.array(metric.last(n)) for metric in self}


def def_tick_format(x, _):
	return np.format_float_positional(x, precision=3, trim='-')


from .segmentation import BinarySegmentationEvaluation, BinaryIntensitySegmentationEvaluation
