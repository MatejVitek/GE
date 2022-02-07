#!/usr/bin/env python3

# Always import these
import os
import sys
from pathlib import Path
from ast import literal_eval
from joblib import delayed, Parallel
from matplotlib.lines import Line2D
from matej import make_module_callable
from matej.collections import dict_product, DotDict, ensure_iterable, flatten, lmap, shuffled, treedict
from matej.parallel import tqdm_joblib
import argparse
from tkinter import *
import tkinter.filedialog as filedialog
from tqdm import tqdm

# Import whatever else is needed
from compute import TRAIN_DATASETS, TEST_DATASETS, Plot  # Plot is needed for pickle loading
from abc import ABC, abstractmethod
from data.sets.mobius import Light
from evaluation import def_tick_format
from evaluation.recognition import *
import itertools as it
import logging
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
import operator as op
import pickle
from statistics import harmonic_mean


# If you get "Cannot connect to X display" errors, run this script as:
# MPLBACKEND=Agg python segmentation_evaluation_plot_and_save_ge.py
# matplotlib should handle this automatically but it can fail when $DISPLAY is set but an X-server is not available (such as when connecting remotely with ssh -Y without running an X-server on the client)


# Constants
ATTR_EXP = 'colour', 'light', 'phone', ('light', 'phone'), 'gaze'  # What to run attribute-based bias experiments on
FIG_EXTS = 'pdf',# 'png', 'svg', 'eps'  # Which formats to save figures to
CMAP = plt.cm.plasma  # Colourmap to use in figures
MARKERS = 'osP*Xv^<>p1234'
ZOOMED_LIMS = .6, .95  # Axis limits of zoomed-in P/R curves
model_complexity = {
	'cgans2020cl': (11.5e6, None),
	'fcn8': (138e6, 15e9),
	'mu-net': (409e3, 180e9),
	'rgb-ss-eye-ms': (22.7e6, None),
	'scleramaskrcnn': (69e6, None),
	'sclerasegnet': (59e6, 17.94e9),
	'sclerau-net2': (3e6, 180e9)
}
for ensemble in ('RGB-SS-Eye-MS+ScleraU-Net2+FCN8', 'RGB-SS-Eye-MS+CGANs2020CL+FCN8+ScleraMaskRCNN', 'RGB-SS-Eye-MS+ScleraU-Net2+FCN8+ScleraSegNet', 'RGB-SS-Eye-MS+ScleraU-Net2+ScleraSegNet', 'ScleraU-Net2+FCN8+ScleraMaskRCNN'):
	model_complexity[ensemble.lower()] = sum(model_complexity[model][0] for model in ensemble.lower().split('+')), None

# Auxiliary stuff
TRAIN_TEST_DICT = {train: [test for test in TEST_DATASETS if test not in train and not (train == 'All' and test == 'SMD')] for train in TRAIN_DATASETS}  # Filter out invalid train-test configurations
TRAIN_TEST = [(train, test) for train, tests in TRAIN_TEST_DICT.items() for test in tests]  # List of all valid train-test configurations
FIG_EXTS = ensure_iterable(FIG_EXTS, True)
colourise = lambda x: zip(x, CMAP(np.linspace(0, 1, len(x))))
fmt = lambda x: np.format_float_positional(x, precision=3, unique=False)
oom = lambda x: 10 ** math.floor(math.log10(x))  # Order of magnitude: oom(0.9) = 0.1, oom(30) = 10
logging.getLogger('matplotlib.backends.backend_ps').addFilter(lambda record: 'PostScript backend' not in record.getMessage())  # Suppress matplotlib warnings about .eps transparency


class Main:
	def __init__(self, *args, **kw):
		# Default values
		self.models = Path(args[0] if len(args) > 0 else kw.get('models', 'Models'))
		self.datasets = Path(args[1] if len(args) > 1 else kw.get('datasets', 'Datasets'))
		self.save = Path(args[2] if len(args) > 2 else kw.get('save', 'Results'))
		self.k = kw.get('k', 5)
		self.discard = kw.get('discard', {'GFCM1', 'GFCM2', 'HSFCM', 'I-FCM', 'RSKFCM'})
		self.plot = kw.get('plot', False)

		# Extra keyword arguments
		self.extra = DotDict(**kw)

	def __str__(self):
		return str(vars(self))

	def __call__(self):
		if not self.models.is_dir():
			raise ValueError(f"{self.models} is not a directory.")
		if not self.datasets.is_dir():
			raise ValueError(f"{self.datasets} is not a directory.")

		self.eval_dir = self.save/'Evaluations'
		self.fig_dir = self.save/'Figures'
		self.latex_dir = self.save/'LaTeX'
		self.pkl_dir = self.save/'Pickles'
		self.eval_dir.mkdir(parents=True, exist_ok=True)
		self.fig_dir.mkdir(parents=True, exist_ok=True)
		self.latex_dir.mkdir(parents=True, exist_ok=True)
		self.pkl_dir.mkdir(parents=True, exist_ok=True)

		plt.rcParams['font.family'] = 'Times New Roman'
		plt.rcParams['font.weight'] = 'normal'
		plt.rcParams['font.size'] = 24

		self._samples = {}
		self._results = treedict()
		for test_dir in self.datasets.iterdir():
			test = test_dir.stem
			with (test_dir/'Samples.pkl').open('rb') as f:
				self._samples[test] = pickle.load(f)
			with (test_dir/'Recognition.pkl').open('rb') as f:
				self._results['dist'][test] = pickle.load(f)

		for model_dir in self.models.iterdir():
			model = model_dir.name
			if model in self.discard:
				continue
			for train, tests in TRAIN_TEST_DICT.items():
				# Read results
				for test in tests:
					try:
						with (model_dir/f'Pickles/Segmentation/{train}_{test}.pkl').open('rb') as f:
							for collection in ('seg', 'pr', 'mean_pr'):
								self._results[collection][model][train][test] = pickle.load(f)
					except IOError:
						print(f"Missing segmentation results for {model} ({train} - {test})")
					try:
						with (model_dir/f'Pickles/Recognition/{train}_{test}.pkl').open('rb') as f:
							self._results['dist'][model][train][test] = pickle.load(f)
					except IOError:
						print(f"Missing recognition results for {model} ({train} - {test})")

				# Compute harmonic means
				try:
					for pb in range(2):
						for metric in next(iter(self._results['seg'][model][train].values()))[pb]:
							self._results['hmean'][model][train][pb][metric] = harmonic_mean([self._results['seg'][model][train][t][pb][metric].mean() for t in tests])
				except (StopIteration, AttributeError):
					print("Cannot compute harmonic mean for non-existent segmentation results")

		print("Sorting models by the harmonic mean of their binary F1-Scores on the evaluation datasets")
		self._sorted_models = sorted(
			filter(lambda model: '+' not in model, self._results['seg']),
			key=lambda model: self._results['hmean'][model]['All'][1]['F1-score'],
			reverse=True
		)

		self._bias_matrices = [[], [], []]

		self._experiment1()
		for attr in ATTR_EXP:
			self._experiment2(attr)
		self._experiment3()
		self._experiment4()
		self._experiment5()
		self._experiment6()

		for strat in range(2):
			with Heatmap("Overall (Stratified)" if strat else "Overall", self.fig_dir, self._bias_matrices[2]) as hmap:
				hmap.plot(np.corrcoef(np.hstack(self._bias_matrices[strat])))

		if self.plot:
			plt.show()

	def _experiment1(self):
		print("Experiment 1: Overall performance", flush=True)
		tests = TRAIN_TEST_DICT['All']

		# Save and latexify overall performances
		with (self.latex_dir/'Performance.txt').open('w', encoding='utf-8') as latex:
			for model in self._sorted_models:
				for test in tests:
					r = self._results['seg'][model]['All'][test]
					mean_std = [{metric: (r[pb][metric].mean(), r[pb][metric].std()) for metric in r[pb]} for pb in range(2)]
					self._save_evals(mean_std, f'{model} - All - {test}')
					self._latexify_evals(mean_std, self._results['hmean'][model]['All'], latex, model, test, tests)

		# Plot overall performances to bar plot and P/R curves
		with Bar('Overall', self.fig_dir, self._sorted_models, len(tests)) as bar:
			for i, (test, bar_colour) in enumerate(colourise(tests)):
				with PR(test, self.fig_dir) as roc:
					for j, (model, roc_colour) in enumerate(colourise(self._sorted_models)):
						mean_plot, lower_std, upper_std = self._results['mean_pr'][model]['All'][test]

						# Compute k-fold std
						results = self._results['seg'][model]['All'][test][1]['F1-score'].copy()
						np.random.shuffle(results)
						folds = np.array_split(results, self.k)
						std = np.std(lmap(np.mean, folds))

						bar.plot(results.mean(), j, i, std=std, label=test, colour=bar_colour)
						#roc.plot(mean_plot, lower_std, upper_std, label=model, colour=roc_colour)
						roc.plot(mean_plot, label=model, colour=roc_colour)

	def _experiment2(self, attrs):
		attrs = ensure_iterable(attrs, True)
		name = ", ".join(f"{attr.title()}s" for attr in attrs)
		print(f"Experiment 2: Bias across different {name}", flush=True)
		models = list(self._sorted_models)
		if 'colour' in attrs:
			models.append('RGB-SS-Eye-MS+ScleraU-Net2+FCN8')
		if 'light' in attrs:
			models.append('RGB-SS-Eye-MS+ScleraU-Net2+FCN8+ScleraSegNet')
		if 'phone' in attrs:
			models.append('RGB-SS-Eye-MS+ScleraU-Net2+ScleraSegNet')

		bias = {}
		f1 = {}
		with (self.latex_dir/f'Bias - {name}.txt').open('w', encoding='utf-8') as latex:
			for model in models:
				for train in TRAIN_TEST_DICT:
					samples = self._samples['MOBIUS']
					f1scores = self._results['seg'][model][train]['MOBIUS'][1]['F1-score']
					possible_values = {attr: {getattr(sample, attr) for sample in samples} for attr in attrs}
					groups = [  # List of 1D arrays. Each 1D array contains per-image F1s for a specific attribute value combination (such as light=natural, phone=iPhone)
						f1scores[[i for i, sample in enumerate(samples) if all(getattr(sample, attr) == current_values[attr] for attr in attrs)]]
						for current_values in dict_product(possible_values)
					]

					b = self._compute_biases(groups)
					self._save_biases(b, f'{model} - {train} - {name}')
					self._latexify_biases(b, latex, model, models, train, list(TRAIN_TEST_DICT))

					# Save results from train config 'All' for later plotting
					if train == 'All':
						bias[model] = b
						f1[model] = f1scores.mean()

		self._plot_biases(bias, f1, models, name)

	def _experiment3(self):
		print("Experiment 3: Bias across different evaluation datasets", flush=True)
		trains = [train for train in TRAIN_TEST_DICT if train != 'All' and 'SMD' not in train]  # Skip models trained on SMD so we can consistently compute bias on 3 evaluation datasets
		models = self._sorted_models + ['RGB-SS-Eye-MS+CGANs2020CL+FCN8+ScleraMaskRCNN']

		bias = treedict()
		with (self.latex_dir/'Bias - Test data.txt').open('w', encoding='utf-8') as latex:
			for model in models:
				for train in trains:
					groups = [self._results['seg'][model][train][test][1]['F1-score'] for test in TRAIN_TEST_DICT[train]]
					bias[train][model] = self._compute_biases(groups)
					self._save_biases(bias[train][model], f'{model} - {train}')
					self._latexify_biases(bias[train][model], latex, model, models, train, trains)

		for train in trains:
			self._plot_biases(bias[train], {model: self._results['hmean'][model][train][1]['F1-score'] for model in self._results['hmean']}, models, f"test data ({train})")

	def _experiment4(self):
		print("Experiment 4: Bias across different training datasets", flush=True)
		tests = [test for test in TEST_DATASETS if test != 'SMD']  # Skip SMD so we can consistently compute bias on 5 training configurations
		models = self._sorted_models + ['ScleraU-Net2+FCN8+ScleraMaskRCNN']

		bias = treedict()
		f1 = treedict()
		with (self.eval_dir/'LaTeX - Bias - Train data.txt').open('w', encoding='utf-8') as latex:
			for model in models:
				for test in tests:
					groups = [self._results['seg'][model][train][test][1]['F1-score'] for train in TRAIN_TEST_DICT]
					bias[test][model] = self._compute_biases(groups)
					f1[test][model] = harmonic_mean(lmap(np.mean, groups))
					self._save_biases(bias[test][model], f'{model} - {test}')
					self._latexify_biases(bias[test][model], latex, model, models, test, test)

		for test in tests:
			self._plot_biases(bias[test], f1[test], models, f"train data ({test})")

	def _experiment5(self):
		print("Experiment 5: Recognition", flush=True)
		train_test_configs = ('All', 'MOBIUS'), ('MASD+SBVPI', 'SMD'), ('All', 'SLD')  # For SMD we can't use the 'All' model, as that was trained on SMD data
		tests = lmap(op.itemgetter(1), train_test_configs)
		models = ['Ground truth'] + self._sorted_models

		saved_evals = treedict()
		for train, test in train_test_configs:
			for model in models:
				results = self._results['dist'][test] if model == 'Ground truth' else self._results['dist'][model][train][test]
				save_f = self.pkl_dir/f'{test} - {model}.pkl'
				if not save_f.is_file():
					with tqdm_joblib(tqdm(total=len(results), desc="Computing verification metrics")):
						evals = Parallel(n_jobs=len(results))(
							delayed(self._evaluate_method)(self._samples[test], dist_matrix, folds=5)
							for dist_matrix in results.values()
						)
					save_f.parent.mkdir(parents=True, exist_ok=True)
					print(f"Saving evals to {save_f}")
					with save_f.open('wb') as f:
						pickle.dump(evals, f)
				else:
					print(f"Loading evals from {save_f}")
					with save_f.open('rb') as f:
						evals = pickle.load(f)

				for method, (eval_, cross_eval) in zip(results, evals):
					self._save_rec(cross_eval, f'Recognition - {model} - {method} - {test}')
					saved_evals[method][model][test] = eval_, cross_eval

		for method in saved_evals:
			with (self.latex_dir/f'Recognition - {method}.txt').open('w', encoding='utf-8') as latex:
				for test in tests:
					for model in models:
						_, cross_eval = saved_evals[method][model][test]
						self._latexify_rec(cross_eval, latex, test, tests, model, models)

		for test in tests:
			for method in saved_evals:
				with \
					ROC(f'{test} - {method}', self.fig_dir, xscale='log') as roc, \
					Histogram(f'{test} - {method}', self.fig_dir, xscale='invlog') as hist:
					for model, colour in colourise(models):
						eval_, _ = saved_evals[method][model][test]
						roc.plot(eval_.far, eval_.ver, label=model, colour=colour)
						hist.plot(eval_.genuine, eval_.impostors, label=model, colour=colour)

	def _experiment6(self):
		print("Experiment 6: Recognition on bias groups", flush=True)
		train_test_configs = ('All', 'MOBIUS'), ('MASD+SBVPI', 'SMD'), ('All', 'SLD')  # For SMD we can't use the 'All' model, as that was trained on SMD data
		tests = lmap(op.itemgetter(1), train_test_configs)
		bias_clusters = {
			'Eye colours': (('ScleraMaskRCNN',), ('CGANs2020CL', 'ScleraU-Net2'), ('RGB-SS-Eye-MS', 'FCN8', 'ScleraSegNet', 'MU-Net')),
			'Evaluation data': (('RGB-SS-Eye-MS', 'CGANs2020CL', 'FCN8', 'ScleraSegNet', 'ScleraMaskRCNN'), ('ScleraU-Net2',), ('MU-Net',)),
			'Lightings': (('ScleraU-Net2',), ('RGB-SS-Eye-MS', 'CGANs2020CL', 'FCN8', 'ScleraSegNet', 'MU-Net'), ('ScleraMaskRCNN',)),
			'Phones': (('RGB-SS-Eye-MS', 'CGANs2020CL', 'ScleraU-Net2', 'MU-Net', 'ScleraMaskRCNN'), ('FCN8', 'ScleraSegNet')),
			'Training data': (('FCN8',), ('RGB-SS-Eye-MS', 'CGANs2020CL', 'ScleraU-Net2', 'ScleraSegNet', 'ScleraMaskRCNN'), ('MU-Net',))
		}

		for bias_type, clusters in bias_clusters.items():
			saved_evals = treedict()
			for train, test in train_test_configs:
				for c, cluster in enumerate(clusters, start=1):
					for model in cluster:
						with (self.pkl_dir/f'{test} - {model}.pkl').open('rb') as f:
							for method, (eval_, cross_eval) in zip(self._results['dist'][test], pickle.load(f)):
								if test not in saved_evals[method][cluster]:
									saved_evals[method][cluster][test] = []
								saved_evals[method][cluster][test].append((cross_eval, eval_.far, eval_.ver, eval_.genuine, eval_.impostors))

					for method in saved_evals:
						cross_eval = Evaluation.from_evals(lmap(op.itemgetter(0), saved_evals[method][cluster][test]))
						far = np.stack(lmap(op.itemgetter(1), saved_evals[method][cluster][test])).mean(axis=0)
						ver = np.stack(lmap(op.itemgetter(2), saved_evals[method][cluster][test])).mean(axis=0)
						genuine = np.hstack(lmap(op.itemgetter(3), saved_evals[method][cluster][test]))
						impostors = np.hstack(lmap(op.itemgetter(4), saved_evals[method][cluster][test]))
						saved_evals[method][cluster][test] = cross_eval, far, ver, genuine, impostors
						self._save_rec(cross_eval, f'Recognition across {bias_type} - Cluster {c} - {method} - {test}')

			for method in saved_evals:
				with (self.latex_dir/f'Recognition across {bias_type} - {method}.txt').open('w', encoding='utf-8') as latex:
					for test in tests:
						for c, cluster in enumerate(clusters, start=1):
							cross_eval, _, _, _, _ = saved_evals[method][cluster][test]
							self._latexify_rec(cross_eval, latex, test, tests, f'Cluster {c}', [f'Cluster {x+1}' for x in range(len(clusters))])

			for test in tests:
				for method in saved_evals:
					with \
						ROC(f'{bias_type} - {test} - {method}', self.fig_dir, xscale='log') as roc, \
						Histogram(f'{bias_type} - {test} - {method}', self.fig_dir, xscale='invlog') as hist:
						for c, (cluster, colour) in enumerate(colourise(clusters), start=1):
							_, far, ver, genuine, impostors = saved_evals[method][cluster][test]
							roc.plot(far, ver, label=f"Cluster {c}", colour=colour)
							hist.plot(genuine, impostors, label=f"Cluster {c}", colour=colour)

	def _save_evals(self, mean_std, name):
		save = self.eval_dir/f'{name}.txt'
		print(f"Saving to {save}")
		with save.open('w', encoding='utf-8') as f:
			for pb, pb_text in enumerate(("Probabilistic", "Binarised")):
				print(pb_text, file=f)
				for metric, (mean, std) in mean_std[pb].items():
					print(f"{metric} (μ ± σ): {mean} ± {std}", file=f)
				print(file=f)

	def _latexify_evals(self, mean_std, hmean, latex, model, test, tests):
		if test == tests[0]:  # First line of model
			latex.write(fr"\multirow{{{len(tests)}}}{{*}}{{{model}}}")
		latex.write(f" & {tests if test == tests[0] else test.ljust(max(map(len, tests[1:])))}")
		for bp, metrics in enumerate((('F1-score', 'Precision', 'Recall', 'IoU'), ('F1-score', 'AUC'))):
			for metric in metrics:
				latex.write(f" & {fmt(mean_std[not bp][metric][0])} & ")
				if test == tests[0]:  # First line of model
					latex.write(fr"\multirow{{{len(tests)}}}{{*}}{{{fmt(hmean[not bp][metric])}}}")
		latex.write(r" \\")
		if test == tests[-1] and model != self._sorted_models[-1]:  # Last line of model but not last line overall
			latex.write(r"\hline")
		latex.write("\n")

	def _compute_biases(self, groups, n_samples=None):
		# By default both stratified (with 100 samples per group) and non-stratified experiments will be run
		if n_samples is None:
			bias = self._compute_biases(groups, 100)
			n_samples = 0
		else:
			bias = {}, {}

		if n_samples:
			groups = [np.random.choice(group, n_samples, False) for group in groups]  # List of 1D arrays of per-image F1s for stratified groups
		group_means = np.array(lmap(np.mean, groups))  # 1D array of per-stratified-group mean F1s
		group_stds = np.array(lmap(np.std, groups))  # 1D array of per-stratified-group σ of F1
		control_groups = [list(it.islice(shuffled(list(flatten(groups))), len(group))) for group in groups]  # 2D list of per-image F1s across control groups
		control_means = np.array(lmap(np.mean, control_groups))  # 1D array of per-control-group mean F1s

		strat = bool(n_samples)  # Stratified?
		bias[strat]['σ'] = group_means.std()  # Scalar σ of F1 means across groups
		bias[strat]['MAD'] = np.mean(np.abs(group_means - group_means.mean()))  # Scalar mean absolute deviation of F1 means across groups
		bias[strat]['CR'] = bias[strat]['σ'] / control_means.std()  # Scalar ratio between σ of F1 means across groups and σ of F1 means across control groups
		bias[strat]['IIR'] = bias[strat]['σ'] / group_stds.mean()  # Scalar ratio between σ of F1 means across groups and the mean σ of F1 within groups

		return bias

	def _save_biases(self, bias, name):
		save = self.eval_dir/f'Bias - {name}.txt'
		print(f"Saving to {save}")
		with save.open('w', encoding='utf-8') as f:
			for s, strat in enumerate(("Total", "Stratified")):
				print(strat, file=f)
				for metric, score in bias[s].items():
					print(f"{metric}: {score}", file=f)
				print(file=f)

	def _latexify_biases(self, bias, latex, model, models, column2, column2_values):
		if column2 == column2_values[0]:  # First line of model
			latex.write(fr"\multirow{{{len(column2_values)}}}{{*}}{{{'Ensemble' if '+' in model else model}}}")
		latex.write(f" & {column2 if column2 == column2_values[0] else column2.ljust(max(map(len, column2_values[1:])))}")
		latex.write("".join(f" & ${fmt(bias[strat][metric])}$" for strat in (False, True) for metric in ('σ', 'MAD', 'CR', 'IIR')))
		latex.write(r" \\")
		if column2 == column2_values[-1] and model != models[-1]:  # Last line of model but not last line overall
			latex.write(r"\hline")
		latex.write("\n")

	def _plot_biases(self, bias, f1, models, fig_suffix):
		metrics = list(next(iter(bias.values()))[0])
		labels = ["Ensemble" if '+' in model else model for model in models]
		for strat in range(2):
			if strat:
				fig_suffix += " (Stratified)"
			with Bar(f'Bias across {fig_suffix}', self.fig_dir, labels, len(metrics), ylabel="Bias") as bar:
				for i, (metric, bar_colour) in enumerate(colourise(metrics)):
					max_bias = {metric: max(bias[model][strat][metric] for model in models) for metric in metrics}
					with Scatter(f'Bias ({metric}) across {fig_suffix}', self.fig_dir, xlabel="F1-score", ylabel=metric) as scatter, \
						Scatter(f'Bias ({metric}) and Size across {fig_suffix}', self.fig_dir, xlabel="F1-score", ylabel=metric) as size:  # Params as circle sizes
						#Scatter(f'Bias ({metric}) and Size across {fig_suffix}', self.fig_dir, xscale='log', xlabel="# Parameters", ylabel="F1-score") as size:  # Params on x axis
						for j, ((model, sc_colour), label, marker) in enumerate(zip(colourise(models), labels, MARKERS)):
							b = bias[model][strat][metric]
							b_normalised_to_01 = b / max_bias[metric]  # Normalise bias to 0-1 range (for marker size in scatter plot)
							b_normalised_to_stdmad = b_normalised_to_01 * max(max_bias['σ'], max_bias['MAD']) if metric not in ('σ', 'MAD') else b  # Normalise metrics other than σ and MAD to the range of these two (so we can plot them on the same graph)
							bar.plot(b_normalised_to_stdmad, j, i, label=metric, colour=bar_colour)
							scatter.plot(f1[model], b, label=label, colour=sc_colour, marker=marker)
							#size.plot(model_complexity[model.lower()][0], f1[model], 150*b_normalised_to_01, label=label, colour=sc_colour, marker=marker)
							size.plot(f1[model], b, .01 * math.sqrt(model_complexity[model.lower()][0]), label=label, colour=sc_colour, marker=marker)
			with Heatmap(fig_suffix, self.fig_dir, metrics) as hmap:
				self._bias_matrices[strat].append(np.array([[bias[model][strat][metric] for model in models] for metric in metrics]))
				hmap.plot(np.corrcoef(self._bias_matrices[strat][-1]))
				self._bias_matrices[2] = metrics

	def _evaluate_method(self, samples, dist_matrix, folds=1):
		# Gallery and probe
		idx = shuffled(range(len(samples)))
		idx = [i for i in idx if not hasattr(samples[i], 'light') or samples[i].light is Light.POOR or samples[i].n == 'bad']  # Filter out poorly lit images
		gp_split = round(.3 * len(idx))
		g_idx = idx[:gp_split]
		p_idx = idx[gp_split:]
		g_classes = [samples[i].label for i in g_idx]
		p_classes = [samples[i].label for i in p_idx]
		d = dist_matrix[np.ix_(g_idx, p_idx)]  # Filter so we only have the gallery rows and probe columns

		# Overall evaluation
		eval_ = VerificationEvaluation()
		eval_.metrics_from_dist_matrix(d, g_classes, p_classes, balance_attempts=True)

		if folds == 1:
			return eval_

		# Cross-evaluation
		cross_eval = VerificationEvaluation()
		for fold in np.array_split(p_idx, folds):
			p_classes = [samples[i].label for i in fold]
			d = dist_matrix[np.ix_(g_idx, fold)]
			cross_eval.metrics_from_dist_matrix(d, g_classes, p_classes, balance_attempts=True)

		return eval_, cross_eval

	def _save_rec(self, ver_eval, name):
		save = self.eval_dir/f'{name}.txt'
		print(f"Saving to {save}")
		with save.open('w', encoding='utf-8') as f:
			print(ver_eval, file=f)

	def _latexify_rec(self, ver_eval, latex, test, tests, model, models):
		if model == models[0]:  # First line of model
			latex.write(fr"\multirow{{{len(models)}}}{{*}}{{{test}}}")
		latex.write(f" & {model if model == models[0] else model.ljust(max(map(len, models[1:])))}")
		for metric in ('EER', 'VER@1%FAR', 'VER@10%FAR', 'AUC'):
			latex.write(fr" & ${fmt(ver_eval[metric].mean)} \pm {fmt(ver_eval[metric].std)}$")
		latex.write(r" \\")
		if model == models[-1] and test != tests[-1]:  # Last line of model but not last line overall
			latex.write(r"\hline")
		latex.write("\n")

	def process_command_line_options(self):
		ap = argparse.ArgumentParser(description="Evaluate segmentation results.")
		ap.add_argument('models', type=Path, nargs='?', default=self.models, help="directory with model information")
		ap.add_argument('datasets', type=Path, nargs='?', default=self.datasets, help="directory with the original dataset pickles")
		ap.add_argument('save', type=Path, nargs='?', default=self.save, help="directory to save figures and evaluations to")
		ap.add_argument('-k', '--folds', type=int, default=self.k, help="number of folds to use for std")
		ap.add_argument('-d', '--discard', action='append', type=str, help="discard model")
		ap.add_argument('-p', '--plot', action='store_true', help="show drawn plots")
		ap.parse_known_args(namespace=self)

		ap = argparse.ArgumentParser(description="Extra keyword arguments.")
		ap.add_argument('-e', '--extra', nargs=2, action='append', help="any extra keyword-value argument pairs")
		ap.parse_known_args(namespace=self.extra)

		if self.extra.extra:
			for key, value in self.extra.extra:
				try:
					self.extra[key] = literal_eval(value)
				except ValueError:
					self.extra[key] = value
			del self.extra['extra']

	def gui(self):
		gui = GUI(self)
		gui.mainloop()
		return gui.ok


class Figure(ABC):
	def __init__(self, name, save_dir, *, fontsize=24):
		self.name = name
		self.dir = save_dir
		self.fontsize = fontsize
		self.fig = None
		self.ax = None

	@abstractmethod
	def __enter__(self, *args, **kw):
		plt.rcParams['font.size'] = self.fontsize
		self.fig, self.ax = plt.subplots(*args, num=str(self.dir/self.name), **kw)
		return self

	def __exit__(self, *args, **kw):
		plt.rcParams['font.size'] = self.fontsize
		self.close()
		plt.close(self.fig)

	@abstractmethod
	def close(self):
		pass

	@abstractmethod
	def plot(self):
		plt.rcParams['font.size'] = self.fontsize

	def save(self, name=None, fig=None):
		if fig is None:
			fig = self.fig
		if name is None:
			name = self.name
		self.dir.mkdir(parents=True, exist_ok=True)
		for ext in FIG_EXTS:
			save = self.dir/f'{name}.{ext}'
			print(f"Saving to {save}")
			fig.savefig(save, bbox_inches='tight')

	@staticmethod
	def _nice_tick_size(min_, max_, min_ticks=3, max_ticks=7):
		diff = max_ - min_
		return min(
			oom(diff) * np.array([.1, .2, .5, 1, 2, 5]),  # Different possible tick sizes
			# REQUIRES PYTHON>=3.8 key=lambda tick_size: (max(0, min_ticks - (n_ticks := diff // tick_size + 1), n_ticks - max_ticks), n_ticks)  # Return the one closest to the requested number of ticks. If several are in the range, return the one with the fewest ticks.
			key=lambda tick_size: (max(0, min_ticks - diff // tick_size - 1, diff // tick_size + 1 - max_ticks), diff // tick_size + 1)
		)


class PR(Figure):
	def __init__(self, name, save_dir, **kw):
		super().__init__(name, save_dir/'PR', **kw)
		self.cmb_fig = None
		self.cmb_ax = None
		self.zoom_ax = None

	def __enter__(self):
		super().__enter__()
		self.cmb_fig, self.cmb_ax = plt.subplots(num=str(self.dir/f'{self.name} Combined'))
		# This .81 has to be the diff of original ylims
		self.zoom_ax = zoomed_inset_axes(self.cmb_ax, .81 / abs(np.diff(ZOOMED_LIMS)[0]), bbox_to_anchor=(1.15, 0, 1, 1), bbox_transform=self.cmb_ax.transAxes, loc='upper left', borderpad=0)
		self.axes = self.ax, self.cmb_ax, self.zoom_ax
		for ax in self.axes:
			ax.grid(which='major', alpha=.5)
			ax.grid(which='minor', alpha=.2)
			ax.xaxis.set_major_formatter(FuncFormatter(def_tick_format))
			ax.yaxis.set_major_formatter(FuncFormatter(def_tick_format))
			ax.margins(0)
			ax.set_xlabel("Recall")
			ax.set_ylabel("Precision")
		self.zoom_ax.set_xlabel(None)
		self.zoom_ax.set_ylabel(None)
		self.fig.tight_layout(pad=0)
		#self.cmb_fig.tight_layout(pad=0)
		return self

	def close(self, *args, **kw):
		self.ax.set_xlim(.2, 1.01)
		self.ax.set_ylim(0, 1.01)
		self.ax.xaxis.set_major_locator(MultipleLocator(.2))
		#self.ax.xaxis.set_minor_locator(MultipleLocator(.1))
		self.ax.yaxis.set_major_locator(MultipleLocator(.2))
		#self.ax.yaxis.set_minor_locator(MultipleLocator(.1))
		self.save(f'{self.name} (No Legend)')

		_, labels = self.ax.get_legend_handles_labels()
		if labels:
			ncol = (len(labels) - 1) // 10 + 1
			legend = self.ax.legend(bbox_to_anchor=(1.02, .5), ncol=ncol, loc='center left', borderaxespad=0)
			self.save()

		self.ax.set_xlim(*ZOOMED_LIMS)
		self.ax.set_ylim(*ZOOMED_LIMS)
		self.ax.xaxis.set_major_locator(MultipleLocator(.1))
		self.ax.xaxis.set_minor_locator(MultipleLocator(.05))
		self.ax.yaxis.set_major_locator(MultipleLocator(.1))
		self.ax.yaxis.set_minor_locator(MultipleLocator(.05))
		#self.save(f'{self.name} (Zoomed)')

		if labels:
			legend.remove()
			#self.save(f'{self.name} (Zoomed, No Legend)')

		self.cmb_ax.set_xlim(.2, 1.01)
		self.cmb_ax.set_ylim(.2, 1.01)
		self.cmb_ax.xaxis.set_major_locator(MultipleLocator(.2))
		self.cmb_ax.yaxis.set_major_locator(MultipleLocator(.2))
		mark_inset(self.cmb_ax, self.zoom_ax, loc1=2, loc2=3, ec='0.5')
		self.zoom_ax.set_xlim(*ZOOMED_LIMS)
		self.zoom_ax.set_ylim(*ZOOMED_LIMS)
		self.zoom_ax.xaxis.set_major_locator(MultipleLocator(.1))
		self.zoom_ax.xaxis.set_minor_locator(MultipleLocator(.05))
		self.zoom_ax.yaxis.set_major_locator(MultipleLocator(.1))
		self.zoom_ax.yaxis.set_minor_locator(MultipleLocator(.05))
		self.save(f'{self.name} (Combined, No Legend)', self.cmb_fig)

		if labels:
			self.cmb_ax.legend(bbox_to_anchor=(2.2, .5), loc='center left', ncol=ncol, columnspacing=.5, borderaxespad=0)
			self.save(f'{self.name} (Combined)', self.cmb_fig)

		plt.close(self.cmb_fig)

	def plot(self, mean_plot, lower_std=None, upper_std=None, *, label=None, colour=None, bin_only=False):
		super().plot()
		for ax in self.axes:
			if not bin_only:
				ax.plot(mean_plot.recall, mean_plot.precision, label=label, linewidth=2, color=colour)
				for std in lower_std, upper_std:
					if std is not None:
						ax.plot(std.recall, std.precision, ':', linewidth=1, color=colour)
				ax.plot(*mean_plot.f1_point, 'o', markersize=12, color=colour)
				ax.plot(*mean_plot.bin_point, 'o', markersize=12, color=colour, markerfacecolor='none')
			else:
				ax.plot(*mean_plot.bin_point, 'o', label=label, markersize=12, color=colour, markerfacecolor='none')

class Bar(Figure):
	def __init__(self, name, save_dir, groups, n=1, *, ylabel="F1-score", fontsize=30, margin=.2):
		super().__init__(name, save_dir/'Bar', fontsize=fontsize)
		self.groups = groups
		self.m = len(groups)
		self.n = n
		self.ylabel = ylabel
		self.margin = margin
		self.width = (1 - self.margin) / self.n
		self.min = None
		self.max = None

	def __enter__(self):
		super().__enter__(figsize=(15, 5))
		self.ax.grid(axis='y', which='major', alpha=.5)
		self.ax.grid(axis='y', which='minor', alpha=.2)
		self.ax.yaxis.set_major_formatter(FuncFormatter(def_tick_format))
		self.ax.margins(0)
		self.ax.set_ylabel(self.ylabel)
		self.fig.tight_layout(pad=0)
		self.min = float('inf')
		self.max = float('-inf')
		return self

	def close(self, *args, **kw):
		handles, labels = self.ax.get_legend_handles_labels()
		if labels:
			by_label = dict(zip(labels, handles))  # Remove duplicate labels
			for attempt in range(4, 1, -1):
				if len(by_label) % attempt == 0:
					ncol = attempt
					break
			else:
				ncol = 3
			self.ax.legend(by_label.values(), by_label.keys(), ncol=ncol, bbox_to_anchor=(.02, 1.02, .96, .1), loc='lower left', mode='expand', borderaxespad=0)
		self.ax.set_xticks(np.arange(self.m) + (self.margin + self.n * self.width) / 2)
		#self.ax.set_xticklabels(self.groups, rotation=60, ha='right', rotation_mode='anchor')  # Rotated x labels
		self.ax.set_xticklabels([group if g % 2 else f"\n{group}" for g, group in enumerate(self.groups)])  # 2-row x labels
		ymin = self.min if self.min != float('inf') else 0
		ymax = self.max if self.max != float('-inf') else 1
		ytick_size = self._nice_tick_size(ymin, ymax)
		self.ax.set_ylim(max(0, ymin - ytick_size), ymax + ytick_size)
		self.ax.yaxis.set_major_locator(MultipleLocator(ytick_size))
		self.ax.yaxis.set_minor_locator(MultipleLocator(ytick_size / 2))
		self.save()
		self.ax.tick_params(axis='x', bottom=False, labelbottom=False)
		#self.save(f'{self.name} (No Labels)')

	def plot(self, val, group=0, index=0, *, std=None, label=None, colour=None):
		super().plot()
		plt.rcParams['font.size'] = 10
		err_w = np.clip(self.width * 10, 2, 5)
		self.ax.bar(
			group + self.margin / 2 + index * self.width,
			val,
			yerr=std,
			error_kw=dict(lw=err_w, capsize=1.5 * err_w, capthick=.5 * err_w),
			width=self.width,
			align='edge',
			label=label,
			color=colour
		)
		self.min = min(self.min, val - std if std else val)
		self.max = max(self.max, val + std if std else val)


class Scatter(Figure):
	def __init__(self, name, save_dir, *, xlabel="F1-score", ylabel="Bias", xscale='linear', fontsize=28):
		super().__init__(name, save_dir/'Scatter', fontsize=fontsize)
		self.xscale = xscale
		self.xmin = self.xmax = self.ymin = self.ymax = None
		self.xlabel = xlabel
		self.ylabel = ylabel

	def __enter__(self):
		super().__enter__()
		self.ax.grid(axis='y', which='major', alpha=.5)
		self.ax.grid(axis='y', which='minor', alpha=.2)
		self.ax.set_xscale(self.xscale)
		self.ax.yaxis.set_major_formatter(FuncFormatter(def_tick_format))
		self.ax.margins(0)
		self.ax.set_xlabel(self.xlabel)
		self.ax.set_ylabel(self.ylabel)
		self.fig.tight_layout(pad=0)
		self.xmin = self.ymin = float('inf')
		self.xmax = self.ymax = float('-inf')
		return self

	def close(self, *args, **kw):
		xmin = self.xmin if self.xmin != float('inf') else 0
		xmax = self.xmax if self.xmax != float('-inf') else 1
		ymin = self.ymin if self.ymin != float('inf') else 0
		ymax = self.ymax if self.ymax != float('-inf') else 1
		if self.xscale == 'log':
			self.ax.set_xlim(max(1, oom(.1 * xmin)), oom(10 * xmax))
		else:
			xtick_size = self._nice_tick_size(xmin, xmax)
			self.ax.set_xlim(max(0, xmin - xtick_size), xmax + xtick_size)
			self.ax.xaxis.set_major_locator(MultipleLocator(xtick_size))
			self.ax.xaxis.set_minor_locator(MultipleLocator(xtick_size / 2))
		ytick_size = self._nice_tick_size(ymin, ymax)
		self.ax.set_ylim(max(0, ymin - ytick_size), ymax + ytick_size)
		self.ax.yaxis.set_major_locator(MultipleLocator(ytick_size))
		self.ax.yaxis.set_minor_locator(MultipleLocator(ytick_size / 2))
		self.save(f'{self.name} (No Legend)')
		if self.ax.get_legend_handles_labels()[0]:
			self.ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
		self.save()

	def plot(self, x, y, size=None, *, label=None, colour=None, marker=None):
		super().plot()
		markersize = 8
		if size:
			self.ax.plot(x, y, 'o', markersize=size, color=(*colour[:3], .2))
		self.ax.plot(x, y, marker, markersize=markersize, label=label, color=colour)

		self.xmin = min(self.xmin, x)
		self.xmax = max(self.xmax, x)
		self.ymin = min(self.ymin, y)
		self.ymax = max(self.ymax, y)


class ROC(Figure):
	def __init__(self, name, save_dir, *, xlabel="FAR", ylabel="Verification", xscale='linear', fontsize=20):
		super().__init__(name, save_dir/'ROC', fontsize=fontsize)
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.xscale = xscale

	def __enter__(self):
		super().__enter__()
		self.ax.grid(which='major', alpha=.5)
		self.ax.grid(which='minor', alpha=.2)
		self.ax.set_xscale(self.xscale)
		self.ax.xaxis.set_major_formatter(FuncFormatter(def_tick_format))
		self.ax.yaxis.set_major_formatter(FuncFormatter(def_tick_format))
		self.ax.margins(0)
		self.ax.set_xlabel(self.xlabel)
		self.ax.set_ylabel(self.ylabel)
		self.fig.tight_layout(pad=0)
		return self

	def close(self, *args, **kw):
		if self.xscale == 'log':
			self.ax.set_xlim(10e-5, 1.01)
			self.ax.set_xticks(np.logspace(-5, 0, 6))
		else:
			self.ax.set_xlim(0, 1.01)
			self.ax.xaxis.set_major_locator(MultipleLocator(.2))
			#self.ax.xaxis.set_minor_locator(MultipleLocator(.1))
		self.ax.set_ylim(0, 1.01)
		self.ax.yaxis.set_major_locator(MultipleLocator(.2))
		#self.ax.yaxis.set_minor_locator(MultipleLocator(.1))
		self.save(f'{self.name} (No Legend)')

		_, labels = self.ax.get_legend_handles_labels()
		if labels:
			ncol = (len(labels) - 1) // 10 + 1
			self.ax.legend(bbox_to_anchor=(1.02, .5), ncol=ncol, loc='center left', borderaxespad=0)
			self.save()

	def plot(self, x, y, *, label=None, colour=None):
		super().plot()
		self.ax.plot(x, y, label=label, linewidth=2, color=colour)


class CMC(Figure):
	def __init__(self, name, save_dir, *, fontsize=20):
		super().__init__(name, save_dir/'CMC', fontsize=fontsize)

	def __enter__(self):
		super().__enter__()
		self.ax.grid(which='major', alpha=.5)
		self.ax.grid(which='minor', alpha=.2)
		self.ax.xaxis.set_major_formatter(FuncFormatter(def_tick_format))
		self.ax.yaxis.set_major_formatter(FuncFormatter(def_tick_format))
		self.ax.margins(0)
		self.ax.set_xlabel("n")
		self.ax.set_ylabel("Rank-n accuracy")
		self.fig.tight_layout(pad=0)
		return self

	def close(self, *args, **kw):
		self.ax.set_xlim(0, 1.01)
		self.ax.set_ylim(0, 1.01)
		self.ax.xaxis.set_major_locator(MultipleLocator(.2))
		#self.ax.xaxis.set_minor_locator(MultipleLocator(.1))
		self.ax.yaxis.set_major_locator(MultipleLocator(.2))
		#self.ax.yaxis.set_minor_locator(MultipleLocator(.1))
		self.save(f'{self.name} (No Legend)')

		_, labels = self.ax.get_legend_handles_labels()
		if labels:
			ncol = (len(labels) - 1) // 10 + 1
			self.ax.legend(bbox_to_anchor=(1.02, .5), ncol=ncol, loc='center left', borderaxespad=0)
			self.save()

	def plot(self, cmc, *, label=None, colour=None):
		super().plot()
		self.ax.plot(np.arange(len(cmc)), cmc, label=label, linewidth=2, color=colour)


class Histogram(Figure):
	def __init__(self, name, save_dir, *, xlabel="Distance", ylabel="Frequency", xscale='linear', fontsize=20):
		super().__init__(name, save_dir/'Histogram', fontsize=fontsize)
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.xscale = xscale

	def __enter__(self):
		super().__enter__()
		self.ax.grid(which='major', alpha=.5)
		self.ax.grid(which='minor', alpha=.2)
		if self.xscale == 'invlog':
			self.ax.set_xscale('function', functions=(lambda x: 10 ** x, np.log10))
		else:
			self.ax.set_xscale(self.xscale)
		self.ax.xaxis.set_major_formatter(FuncFormatter(def_tick_format))
		self.ax.yaxis.set_major_formatter(FuncFormatter(def_tick_format))
		self.ax.margins(0)
		self.ax.set_xlabel(self.xlabel)
		self.ax.set_ylabel(self.ylabel)
		self.fig.tight_layout(pad=0)
		return self

	def close(self, *args, **kw):
		if self.xscale == 'log':
			self.ax.set_xlim(10e-5, 1.01)
			self.ax.set_xticks(np.logspace(-5, 0, 6))
		else:
			self.ax.set_xlim(0, 1.01)
			if self.xscale == 'invlog':
				self.ax.set_xticks([0, .5, .8, .9, .95, 1])
			else:
				self.ax.xaxis.set_major_locator(MultipleLocator(.2))
				#self.ax.xaxis.set_minor_locator(MultipleLocator(.1))
		self.save(f'{self.name} (No Legend)')

		gen_imp_legend = plt.legend(handles=[Line2D([0], [0], lw=2, ls='-', color='grey', label="Genuine"), Line2D([0], [0], lw=2, ls='--', color='grey', label="Impostors")], ncol=2, loc='upper center')
		self.ax.add_artist(gen_imp_legend)
		self.ax.legend(bbox_to_anchor=(.5, 1.02), ncol=2, loc='lower center', borderaxespad=0)
		self.save()

	def plot(self, genuine, impostors, *, label=None, colour=None, n_points=100):
		super().plot()
		for dist, l, ls in ((genuine, label, '-'), (impostors, None, '--')):
			y, x = np.histogram(dist, n_points)
			x = .5 * (x[:-1] + x[1:])
			self.ax.plot(x, y, label=l, linewidth=1, linestyle=ls, color=colour)


class Heatmap(Figure):
	def __init__(self, name, save_dir, labels, *args, **kw):
		super().__init__(name, save_dir/'Heatmap', *args, **kw)
		self.labels = labels

	def __enter__(self):
		super().__enter__()
		self.ax.margins(0)
		self.fig.tight_layout(pad=0)
		return self

	def close(self, *args, **kw):
		self.ax.set_xticks(range(len(self.labels)))
		self.ax.set_yticks(range(len(self.labels)))
		self.ax.set_xticklabels(self.labels)
		self.ax.set_yticklabels(self.labels)
		self.save()

	def plot(self, matrix):
		super().plot()
		p = self.ax.matshow(matrix, vmin=-1, vmax=1, cmap='bwr')
		self.fig.colorbar(p)


class GUI(Tk):
	def __init__(self, argspace, *args, **kw):
		super().__init__(*args, **kw)
		self.args = argspace
		self.ok = False

		self.frame = Frame(self)
		self.frame.pack(fill=BOTH, expand=YES)

		# In grid(), column default is 0, but row default is first empty row.
		row = 0
		self.models_lbl = Label(self.frame, text="Models:")
		self.models_lbl.grid(column=0, row=row, sticky='w')
		self.models_txt = Entry(self.frame, width=60)
		self.models_txt.insert(END, self.args.models)
		self.models_txt.grid(column=1, columnspan=3, row=row)
		self.models_btn = Button(self.frame, text="Browse", command=self.browse_models)
		self.models_btn.grid(column=4, row=row)

		row += 1
		self.datasets_lbl = Label(self.frame, text="Datasets:")
		self.datasets_lbl.grid(column=0, row=row, sticky='w')
		self.datasets_txt = Entry(self.frame, width=60)
		self.datasets_txt.insert(END, self.args.datasets)
		self.datasets_txt.grid(column=1, columnspan=3, row=row)
		self.datasets_btn = Button(self.frame, text="Browse", command=self.browse_datasets)
		self.datasets_btn.grid(column=4, row=row)

		row += 1
		self.save_lbl = Label(self.frame, text="Save to:")
		self.save_lbl.grid(column=0, row=row, sticky='w')
		self.save_txt = Entry(self.frame, width=60)
		self.save_txt.insert(END, self.args.save)
		self.save_txt.grid(column=1, columnspan=3, row=row)
		self.save_btn = Button(self.frame, text="Browse", command=self.browse_save)
		self.save_btn.grid(column=4, row=row)

		row += 1
		self.k_lbl = Label(self.frame, text="Folds:")
		self.k_lbl.grid(column=0, row=row, stick='w')
		self.k_spn = Spinbox(self.frame, from_=1, to=100)
		self.k_spn.delete(0, END)
		self.k_spn.insert(END, self.k)
		self.k_spn.grid(column=1, row=row)

		row += 1
		self.chk_frame = Frame(self.frame)
		self.chk_frame.grid(row=row, columnspan=3, sticky='w')
		self.plot_var = BooleanVar()
		self.plot_var.set(self.args.plot)
		self.plot_chk = Checkbutton(self.chk_frame, text="Show plots", variable = self.plot_var)
		self.plot_chk.grid(sticky='w')

		row += 1
		self.extra_frame = ExtraFrame(self.frame)
		self.extra_frame.grid(row=row, columnspan=3, sticky='w')

		row += 1
		self.ok_btn = Button(self.frame, text="OK", command=self.confirm)
		self.ok_btn.grid(column=1, row=row)
		self.ok_btn.focus()

	def browse_models(self):
		self._browse_dir(self.models_txt)

	def browse_save(self):
		self._browse_dir(self.datasets_txt)

	def browse_save(self):
		self._browse_dir(self.save_txt)

	def _browse_dir(self, target_txt):
		init_dir = target_txt.get()
		while not os.path.isdir(init_dir):
			init_dir = os.path.dirname(init_dir)

		new_entry = filedialog.askdirectory(parent=self, initialdir=init_dir)
		if new_entry:
			_set_entry_text(target_txt, new_entry)

	def _browse_file(self, target_txt, exts=None):
		init_dir = os.path.dirname(target_txt.get())
		while not os.path.isdir(init_dir):
			init_dir = os.path.dirname(init_dir)

		if exts:
			new_entry = filedialog.askopenfilename(parent=self, filetypes=exts, initialdir=init_dir)
		else:
			new_entry = filedialog.askopenfilename(parent=self, initialdir=init_dir)

		if new_entry:
			_set_entry_text(target_txt, new_entry)

	def confirm(self):
		self.args.models = Path(self.models_txt.get())
		self.args.datasets = Path(self.datasets_txt.get())
		self.args.save = Path(self.save_txt.get())
		self.args.plot = self.plot_var.get()

		for kw in self.extra_frame.pairs:
			key, value = kw.key_txt.get(), kw.value_txt.get()
			if key:
				try:
					self.args.extra[key] = literal_eval(value)
				except ValueError:
					self.args.extra[key] = value

		self.ok = True
		self.destroy()


class ExtraFrame(Frame):
	def __init__(self, *args, **kw):
		super().__init__(*args, **kw)
		self.pairs = []

		self.key_lbl = Label(self, width=30, text="Key", anchor='w')
		self.value_lbl = Label(self, width=30, text="Value", anchor='w')

		self.add_btn = Button(self, text="+", command=self.add_pair)
		self.add_btn.grid()

	def add_pair(self):
		pair_frame = KWFrame(self, pady=2)
		self.pairs.append(pair_frame)
		pair_frame.grid(row=len(self.pairs), columnspan=3)
		self.update_labels_and_button()

	def update_labels_and_button(self):
		if self.pairs:
			self.key_lbl.grid(column=0, row=0, sticky='w')
			self.value_lbl.grid(column=1, row=0, sticky='w')
		else:
			self.key_lbl.grid_remove()
			self.value_lbl.grid_remove()
		self.add_btn.grid(row=len(self.pairs) + 1)


class KWFrame(Frame):
	def __init__(self, *args, **kw):
		super().__init__(*args, **kw)

		self.key_txt = Entry(self, width=30)
		self.key_txt.grid(column=0, row=0)

		self.value_txt = Entry(self, width=30)
		self.value_txt.grid(column=1, row=0)

		self.remove_btn = Button(self, text="-", command=self.remove)
		self.remove_btn.grid(column=2, row=0)

	def remove(self):
		i = self.master.pairs.index(self)
		del self.master.pairs[i]
		for pair in self.master.pairs[i:]:
			pair.grid(row=pair.grid_info()['row'] - 1)
		self.master.update_labels_and_button()
		self.destroy()


def _set_entry_text(entry, txt):
	entry.delete(0, END)
	entry.insert(END, txt)


if __name__ == '__main__':
	main = Main()

	# If CLI arguments, read them
	if len(sys.argv) > 1:
		main.process_command_line_options()

	# Otherwise get them from a GUI
	else:
		if not main.gui():
			# If GUI was cancelled, exit
			sys.exit(0)

	main()

else:
	# Make module callable (python>=3.5)
	def _main(*args, **kw):
		Main(*args, **kw)()
	make_module_callable(__name__, _main)
