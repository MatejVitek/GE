from matej.collections import shuffle, tzip
import numpy as np
import sklearn.metrics as skmetrics

from evaluation import Evaluation, Metric


class VerificationEvaluation(Evaluation):
	def __init__(self, *args, **kw):
		super().__init__(EER(**kw), VER_AT_FAR(**kw), VER_AT_FAR(.1, **kw), AUC(**kw), *args)
		self.far = None
		self.frr = None
		self.ver = None
		self.threshold = None

	def metrics_from_dist_matrix(self, dist_matrix, r_classes=None, c_classes=None, closest_only=False, *, attempts=None, balance_attempts=False, compute_metrics=True, threshold_points=1000):
		"""
		Compute and return FAR, FRR, VER at various threshold values. Also compute all verification metrics.

		:param array dist_matrix: matrix of distances between samples
		:param Iterable r_classes: row classes (labels)
		:param Iterable c_classes: column classes. If None, will assume the entire distance matrix was passed, and as such
                                   `c_classes` will be the same as `r_classes` and the diagonal of the distance matrix
								   will be disregarded (to avoid comparing a sample to itself).
		:param Iterable closest_only: In each probe attempt only the closest gallery sample for each gallery class will be considered.
		:param Iterable attempts: authentication attempts passed as tuples (row, column, same_class)
		:param bool balance_attempts: whether to balance genuine/impostor attempts (I should delete this one for EyeZ, since it's covered by attempts)
		:param bool compute_metrics: whether verification metrics should be computed and updated in this `Evaluation`.
		:param int threshold_points: number of points to use for the threshold.
		                             If 0, all unique points in the distance matrix will be used (can be very slow).
		"""

		if attempts is None:
			if r_classes is None:
				raise ValueError("Either classes or attempts must be passed.")

			disregard_diagonal = False
			if c_classes is None:
				c_classes = r_classes
				disregard_diagonal = True

			if (len(r_classes), len(c_classes)) != dist_matrix.shape:
				raise ValueError(f"Dimensions of classes {(len(r_classes), len(c_classes))} and dist matrix {dist_matrix.shape} must match.")

			if closest_only:
				classes = set(c_classes)
				# This approach can lead to duplicate comparisons (for example (2, 4) and (4, 2)). Should these be allowed or filtered out?
				attempts = [(                                    # List of tuples of:
						r,                                       # Row number
						min((                                    # Column number with minimum distance to row sample (excluding the diagonal if necessary)
							c
							for c, c_cls in enumerate(c_classes)
							if c_cls == cls
							and not (disregard_diagonal and r == c)
						), key=lambda c: dist_matrix[r,c]),  #pylint: disable=cell-var-from-loop
						r_cls == cls                             # Whether the row class and column class are the same
				) for cls in classes                             # For each distinct column class
				for r, r_cls in enumerate(r_classes)]            # For each row
			else:
				attempts = [
					(r, c, r_classes[r] == c_classes[c])
					for r, c in np.ndindex(dist_matrix.shape)
					if not (disregard_diagonal and r <= c)  # If we're using all comparisons, then the entire lower triangle is just duplicated comparisons
				]

		if threshold_points:
			self.threshold = np.linspace(dist_matrix.min(), dist_matrix.max(), threshold_points)
		else:
			self.threshold = np.unique(dist_matrix)
		# Edge case handling
		self.threshold = np.unique(np.concatenate(([-1e-8], self.threshold, [1.])))

		genuine_distances = dist_matrix[tzip(*((r, c) for r, c, s in attempts if s))]
		impostor_distances = dist_matrix[tzip(*((r, c) for r, c, s in attempts if not s))]

		if balance_attempts:
			genuine_distances, impostor_distances = zip(*zip(shuffle(genuine_distances), shuffle(impostor_distances)))

		self.far = np.array([np.count_nonzero(impostor_distances <= t) / len(impostor_distances) for t in self.threshold])
		self.frr = np.array([np.count_nonzero(genuine_distances > t) / len(genuine_distances) for t in self.threshold])
		self.ver = 1 - self.frr

		if compute_metrics:
			for metric in self:
				args = (self.far, self.frr, self.threshold) if isinstance(metric, EER) else (self.far, self.ver)
				metric.compute_and_update(*args)

		return self.far, self.frr, self.ver, self.threshold


class IdentificationEvaluation(Evaluation):
	def __init__(self, *args, **kw):
		super().__init__(Rank(**kw), Rank(5, **kw), AUCMC(**kw), *args)

	def metrics_from_dist_matrix(self, dist_matrix, classes=None, *, compute_metrics=True, threshold_points=1000):
		"""
		Compute and return Rank-n accuracy for 0 <= n <= n_classes. Also compute all identification metrics.

		:param array dist_matrix: matrix of distances between samples
		:param Iterable classes: classes (labels) of the samples in the matrix rows/columns.
		                         If >1D Iterable with first dimension of size 2,
								 the first contained Iterable will be the row classes (probe),
								 while the second will be considered the column classes (gallery).
		:param bool compute_metrics: whether verification metrics should be computed and updated in this `Evaluation`.
		:param int threshold_points: number of points to use for the threshold.
		                             If 0, all unique points in the distance matrix will be used.
		"""

		attempts = sample_from_dist_matrix(classes)
		if threshold_points:
			self.threshold = np.linspace(dist_matrix.min(), dist_matrix.max(), threshold_points)
		else:
			self.threshold = np.unique(dist_matrix)
		# Edge case handling
		self.threshold = np.unique(np.concatenate(([-1e-8], self.threshold, [1.])))

		same = dist_matrix[[(r, c) for r, c, s in attempts if s]]
		diff = dist_matrix[[(r, c) for r, c, s in attempts if not s]]

		self.far = np.array([np.count_nonzero(diff <= t) / len(diff) for t in self.threshold])
		self.frr = np.array([np.count_nonzero(same > t) / len(same) for t in self.threshold])
		self.ver = 1 - self.frr

		if compute_metrics:
			for metric in self:
				args = (self.far, self.frr, self.threshold) if isinstance(metric, EER) else (self.far, self.ver)
				metric.compute_and_update(*args)

		return self.far, self.frr, self.ver, self.threshold


class EER(Metric):
	def __call__(self, far, frr, threshold):
		# See https://math.stackexchange.com/questions/2987246/finding-the-y-coordinate-of-the-intersection-of-two-functions-when-all-x-coordin
		# and https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line for explanation of below formulas
		try:
			i = np.argwhere(np.diff(np.sign(far - frr))).flatten()[0]
		except IndexError:
			# Edge case where there's no intersection
			return 1

		x = (threshold[i], threshold[i+1])
		y = (far[i], far[i+1], frr[i], frr[i+1])
		return (
			((x[0] * y[1] - x[1] * y[0]) * (y[2] - y[3]) - (x[0] * y[3] - x[1] * y[2]) * (y[0] - y[1])) /
			((x[0] - x[1]) * (-y[0] + y[1] + y[2] - y[3]))
		)


class VER_AT_FAR(Metric):
	def __init__(self, far_point=.01, *args, **kw):
		super().__init__(*args, **kw)
		self.far_point = far_point
		self.name = f"VER@{round(far_point * 100)}%FAR"

	def __call__(self, far, ver):
		try:
			i = np.argwhere(np.diff(np.sign(far - self.far_point))).flatten()[0]
		except IndexError:
			# Edge case where there's no intersection
			return 0
		if self.far_point < .9 and far[i+1] == 1:
			# Edge case where FAR just jumps to 1
			return ver[i]

		alpha = (self.far_point - far[i]) / (far[i+1] - far[i])
		return (1 - alpha) * ver[i] + alpha * ver[i+1]


class AUC(Metric):
	def __call__(self, far, ver):
		return skmetrics.auc(far, ver)


class Rank(Metric):
	def __init__(self, n=1, *args, **kw):
		super().__init__(*args, **kw)
		self.n = n
		self.name = f"Rank-{n}"

	def __call__(self, dist_matrix, classes):
		if self.n <= 0:
			return 0
		
		# return accuracy of self.n min-distance predictions for each row


class AUCMC(Metric):
	def __call__(self, dist_matrix=None, classes=None, *, cmc=None):
		if dist_matrix is not None:
			cmc = self.plot_cmc(dist_matrix, classes)
		return skmetrics.auc(np.linspace(0, 1, len(cmc)), cmc)

	@staticmethod
	def plot_cmc(dist_matrix, classes):
		n_classes = len(set(classes))
		return np.array([Rank(top)(dist_matrix, classes) for top in range(len(n_classes) + 1)])
