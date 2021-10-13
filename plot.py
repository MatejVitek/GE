#!/usr/bin/env python3

# Always import these
import os
import sys
from pathlib import Path
from ast import literal_eval
from matej import make_module_callable
from matej.collections import DotDict, dzip, ensure_iterable, flatten, lmap, shuffled, treedict
import argparse
from tkinter import *
import tkinter.filedialog as filedialog

# Import whatever else is needed
from compute import TRAIN_DATASETS, TEST_DATASETS, Plot  # Plot is needed for pickle loading
from abc import ABC, abstractmethod
from evaluation.segmentation import *
from evaluation import def_tick_format
import itertools as it
import logging
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
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

# Auxiliary stuff
TRAIN_TEST_DICT = {train: [test for test in TEST_DATASETS if test not in train and not (train == 'All' and test == 'SMD')] for train in TRAIN_DATASETS}  # Filter out invalid train-test configurations
TRAIN_TEST = [(train, test) for train, tests in TRAIN_TEST_DICT.items() for test in tests]  # List of all valid train-test configurations
FIG_EXTS = ensure_iterable(FIG_EXTS, True)
colourise = lambda x: zip(x, CMAP(np.linspace(0, 1, len(x))))
fmt = lambda x: np.format_float_positional(x, precision=3, unique=False)
oom = lambda x: 10 ** math.floor(math.log10(x))  # Order of magnitude: oom(0.9) = 0.1, oom(30) = 10
dict_product = lambda d: (dzip(d, x) for x in it.product(*d.values()))  # {a: [1, 2], b: [3, 4]} --> [{a: 1, b: 3}, {a: 1, b: 4}, {a: 2, b: 3}, {a: 2, b: 4}]
logging.getLogger('matplotlib.backends.backend_ps').addFilter(lambda record: 'PostScript backend' not in record.getMessage())  # Suppress matplotlib warnings about .eps transparency


class Main:
	def __init__(self, *args, **kw):
		# Default values
		root = Path('')
		self.models = Path(args[0] if len(args) > 0 else kw.get('models', root/'Models'))
		self.save = Path(args[1] if len(args) > 1 else kw.get('save', 'Results'))
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

		self.eval_dir = self.save/'Evaluations'
		self.fig_dir = self.save/'Figures'
		self.eval_dir.mkdir(parents=True, exist_ok=True)
		self.fig_dir.mkdir(parents=True, exist_ok=True)

		plt.rcParams['font.family'] = 'Times New Roman'  # This doesn't work without MS fonts installed
		plt.rcParams['font.weight'] = 'normal'
		plt.rcParams['font.size'] = 24

		self._samples = treedict()
		self._results = treedict()
		self._plots = treedict()
		self._mean_plots = treedict()
		self._hmeans = treedict()
		for model_dir in self.models.iterdir():
			model = model_dir.name
			if model in self.discard:
				continue
			for train, tests in TRAIN_TEST_DICT.items():
				# Read results
				for test in tests:
					for collection, result in zip((self._samples, self._results, self._plots, self._mean_plots), self._load(model_dir, f'{train}_{test}')):
						collection[model][train][test] = result

				# Compute harmonic means
				for pb in range(2):
					for metric in next(iter(self._results[model][train].values()))[pb]:
						self._hmeans[model][train][pb][metric] = harmonic_mean([self._results[model][train][t][pb][metric].mean() for t in tests])

		print("Sorting models by the harmonic mean of their binary F1-Scores on the evaluation datasets")
		self._sorted_models = sorted(self._results.keys(), key=lambda model: self._hmeans[model]['All'][1]['F1-score'], reverse=True)

		self._experiment1()
		for attr in ATTR_EXP:
			self._experiment2(attr)
		self._experiment3()
		self._experiment4()

		if self.plot:
			plt.show()

	def _experiment1(self):
		print("Experiment 1: Overall performance")
		tests = TRAIN_TEST_DICT['All']

		# Save and latexify overall performances
		with (self.eval_dir/'LaTeX - Performance.txt').open('w', encoding='utf-8') as latex:
			for model in self._sorted_models:
				for test in tests:
					r = self._results[model]['All'][test]
					mean_std = [{metric: (r[pb][metric].mean(), r[pb][metric].std()) for metric in r[pb]} for pb in range(2)]
					self._save_evals(mean_std, f'{model} - All - {test}')
					self._latexify_evals(mean_std, self._hmeans[model]['All'], model, test, tests, latex)

		# Plot overall performances to bar plot and P/R curves
		with Bar('Overall', self.fig_dir, self._sorted_models, len(tests)) as bar:
			for i, (test, bar_colour) in enumerate(colourise(tests)):
				with ROC(test, self.fig_dir) as roc:
					for j, (model, roc_colour) in enumerate(colourise(self._sorted_models)):
						mean_plot, lower_std, upper_std = self._mean_plots[model]['All'][test]

						# Compute k-fold std
						results = self._results[model]['All'][test][1]['F1-score'].copy()
						np.random.shuffle(results)
						folds = np.array_split(results, self.k)
						std = np.std(lmap(np.mean, folds))

						bar.plot(results.mean(), j, i, std=std, label=test, colour=bar_colour)
						#roc.plot(mean_plot, lower_std, upper_std, label=model, colour=roc_colour)
						roc.plot(mean_plot, label=model, colour=roc_colour)

	def _experiment2(self, attrs):
		attrs = ensure_iterable(attrs, True)
		name = ", ".join(f"{attr.title()}s" for attr in attrs)
		print(f"Experiment 2: Bias across different {name}")

		# Compute, save, and latexify biases
		bias = {}
		f1 = {}
		with (self.eval_dir/f'LaTeX - Bias - {name}.txt').open('w', encoding='utf-8') as latex:
			for model in self._sorted_models:
				for train in TRAIN_TEST_DICT:
					samples = self._samples[model][train]['MOBIUS']
					f1scores = self._results[model][train]['MOBIUS'][1]['F1-score']
					possible_values = {attr: {getattr(sample, attr) for sample in samples} for attr in attrs}
					groups = [  # List of 1D arrays. Each 1D array contains per-image F1s for a specific attribute value combination (such as light=natural, phone=iPhone)
						f1scores[[i for i, sample in enumerate(samples) if all(getattr(sample, attr) == current_values[attr] for attr in attrs)]]
						for current_values in dict_product(possible_values)
					]

					b = self._compute_biases(groups)
					self._save_biases(b, f'{model} - {train} - {name}')
					self._latexify_biases(b, latex, model, train, list(TRAIN_TEST_DICT))

					# Save results from train config 'All' for later plotting
					if train == 'All':
						bias[model] = b
						f1[model] = f1scores.mean()

		# Plot biases
		self._plot_biases(bias, f1, name)

	def _experiment3(self):
		print("Experiment 3: Bias across different evaluation datasets")
		trains = [train for train in TRAIN_TEST_DICT if train != 'All' and 'SMD' not in train]  # Skip models trained on SMD so we can consistently compute bias on 3 evaluation datasets

		# Compute, save, and latexify biases
		bias = treedict()
		with (self.eval_dir/'LaTeX - Bias - Test data.txt').open('w', encoding='utf-8') as latex:
			for model in self._sorted_models:
				for train in trains:
					groups = [self._results[model][train][test][1]['F1-score'] for test in TRAIN_TEST_DICT[train]]
					bias[model][train] = self._compute_biases(groups)
					self._save_biases(bias[model][train], f'{model} - {train}')
					self._latexify_biases(bias[model][train], latex, model, train, trains)

		# Plot biases
		for train in trains:
			self._plot_biases({model: bias[model][train] for model in bias}, {model: self._hmeans[model][train][1]['F1-score'] for model in self._hmeans}, f"test data ({train})")

	def _experiment4(self):
		print("Experiment 4: Bias across different training datasets")
		tests = [test for test in TEST_DATASETS if test != 'SMD']  # Skip SMD so we can consistently compute bias on 5 training configurations

		# Compute, save, and latexify biases
		bias = treedict()
		f1 = treedict()
		with (self.eval_dir/'LaTeX - Bias - Train data.txt').open('w', encoding='utf-8') as latex:
			for model in self._sorted_models:
				for test in tests:
					groups = [self._results[model][train][test][1]['F1-score'] for train in TRAIN_TEST_DICT]
					bias[model][test] = self._compute_biases(groups)
					f1[model][test] = harmonic_mean(lmap(np.mean, groups))
					self._save_biases(bias[model][test], f'{model} - {test}')
					self._latexify_biases(bias[model][test], latex, model, test, test)

		# Plot biases
		for test in tests:
			self._plot_biases({model: bias[model][test] for model in bias}, {model: f1[model][test] for model in f1}, f"train data ({test})")

	def _load(self, model_dir, name):
		f = model_dir/f'Pickles/{name}.pkl'
		if not f.is_file():
			raise ValueError(f"{f} does not exist")
		print(f"Loading data from {f}")
		with open(f, 'rb') as f:
			return [pickle.load(f) for _ in range(4)]

	def _save_evals(self, mean_std, name):
		save = self.eval_dir/f'{name}.txt'
		print(f"Saving to {save}")
		with save.open('w', encoding='utf-8') as f:
			for pb, pb_text in enumerate(("Probabilistic", "Binarised")):
				print(pb_text, file=f)
				for metric, (mean, std) in mean_std[pb].items():
					print(f"{metric} (μ ± σ): {mean} ± {std}", file=f)
				print(file=f)

	def _latexify_evals(self, mean_std, hmean, model, test, tests, latex):
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
		bias[strat]['Fisher'] = bias[strat]['σ'] / control_means.std()  # Scalar ratio between σ of F1 means across groups and σ of F1 means across control groups
		bias[strat]['Vito'] = bias[strat]['σ'] / group_stds.mean()  # Scalar ratio between σ of F1 means across groups and the mean σ of F1 within groups

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

	def _latexify_biases(self, bias, latex, model, column2, column2_values):
		if column2 == column2_values[0]:  # First line of model
			latex.write(fr"\multirow{{{len(column2_values)}}}{{*}}{{{model}}}")
		latex.write(f" & {column2 if column2 == column2_values[0] else column2.ljust(max(map(len, column2_values[1:])))}")
		latex.write("".join(f" & ${fmt(bias[strat][metric])}$" for strat in (False, True) for metric in ('σ', 'MAD', 'Fisher', 'Vito')))
		latex.write(r" \\")
		if column2 == column2_values[-1] and model != self._sorted_models[-1]:  # Last line of model but not last line overall
			latex.write(r"\hline")
		latex.write("\n")

	def _plot_biases(self, bias, f1, fig_suffix):
		metrics = list(next(iter(bias.values()))[0])
		for strat in range(2):
			if strat:
				fig_suffix += " (Stratified)"
			with Bar(f'Bias across {fig_suffix}', self.fig_dir, self._sorted_models, len(metrics), ylabel="Bias") as bar:
				for i, (metric, bar_colour) in enumerate(colourise(metrics)):
					max_bias = {metric: max(bias[model][strat][metric] for model in self._sorted_models) for metric in metrics}
					with Scatter(f'Bias ({metric}) across {fig_suffix}', self.fig_dir, xlabel="F1-score", ylabel=metric) as scatter, \
						Scatter(f'Bias ({metric}) and Size across {fig_suffix}', self.fig_dir, xscale='log') as size:
						for j, ((model, sc_colour), marker) in enumerate(zip(colourise(self._sorted_models), MARKERS)):
							b = bias[model][strat][metric]
							b_normalised_to_01 = b / max_bias[metric]  # Normalise bias to 0-1 range (for marker size in scatter plot)
							b_normalised_to_stdmad = (b_normalised_to_01 * max(max_bias['σ'], max_bias['MAD'])) if metric not in ('σ', 'MAD') else b  # Normalise metrics other than σ and MAD to the range of these two (so we can plot them on the same graph)
							bar.plot(b_normalised_to_stdmad, j, i, label=metric, colour=bar_colour)
							scatter.plot(f1[model], b, label=model, colour=sc_colour, marker=marker)
							size.plot(model_complexity[model.lower()][0], f1[model], 150*b_normalised_to_01, label=model, colour=sc_colour, marker=marker)

	def process_command_line_options(self):
		ap = argparse.ArgumentParser(description="Evaluate segmentation results.")
		ap.add_argument('models', type=Path, nargs='?', default=self.models, help="directory with model information")
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
	def __init__(self, name, save_dir, fontsize=24):
		self.name = name
		self.dir = save_dir
		self.fontsize = fontsize
		self.fig = None
		self.ax = None

	@abstractmethod
	def __enter__(self, *args, **kw):
		plt.rcParams['font.size'] = self.fontsize
		self.fig, self.ax = plt.subplots(*args, num=self.name, **kw)
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
		for ext in FIG_EXTS:
			save = self.dir/f'{name}.{ext}'
			print(f"Saving to {save}")
			fig.savefig(save, bbox_inches='tight')

	@staticmethod
	def _nice_tick_size(min_, max_, min_ticks=3, max_ticks=7):
		diff = max_ - min_
		return min(
			oom(diff) * np.array([.1, .2, .5, 1, 2, 5]),  # Different possible tick sizes
			key=lambda tick_size: (max(0, min_ticks - (n_ticks := diff // tick_size + 1), n_ticks - max_ticks), n_ticks)  # Return the one closest to the requested number of ticks. If several are in the range, return the one with the fewest ticks.
		)


class ROC(Figure):
	def __init__(self, name, save_dir, fontsize=20):
		super().__init__(f'ROC - {name}', save_dir, fontsize)
		self.cmb_fig = None
		self.cmb_ax = None
		self.zoom_ax = None

	def __enter__(self):
		super().__enter__()
		self.cmb_fig, self.cmb_ax = plt.subplots(num=f'{self.name} Combined')
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
		#self.save(f'{self.name} (No Legend)')

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
		#self.save(f'{self.name} (Combined, No Legend)', self.cmb_fig)

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
	def __init__(self, name, save_dir, groups, n=1, ylabel="F1-score", fontsize=30, margin=.2):
		super().__init__(f'Bar - {name}', save_dir, fontsize)
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
	def __init__(self, name, save_dir, xlabel="# Parameters", ylabel="F1-score", xscale='linear', fontsize=28):
		super().__init__(f'Scatter - {name}', save_dir, fontsize)
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
		#self.save(f'{self.name} (No Legend)')
		if self.ax.get_legend_handles_labels()[0]:
			self.ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
		self.save()

	def plot(self, x, y, size=None, *, label=None, colour=None, marker=None):
		super().plot()
		markersize = 8
		if size:
			self.ax.plot(x, y, 'o', markersize=size, color=(*colour[:3], .25))
		self.ax.plot(x, y, marker, markersize=markersize, label=label, color=colour)

		self.xmin = min(self.xmin, x)
		self.xmax = max(self.xmax, x)
		self.ymin = min(self.ymin, y)
		self.ymax = max(self.ymax, y)


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
