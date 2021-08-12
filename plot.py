#!/usr/bin/env python3

# Always import these
import os
import sys
from pathlib import Path
from ast import literal_eval
from matej import make_module_callable
from matej.collections import DotDict, dzip, ensure_iterable, lmap, shuffle, treedict
from matej.parallel import tqdm_joblib
import argparse
from tkinter import *
from tkinter.colorchooser import askcolor
import tkinter.filedialog as filedialog
from joblib.parallel import Parallel, delayed
from tqdm import tqdm

# Import whatever else is needed
from compute import ATTR_EXP, TRAIN_DATASETS, TEST_DATASETS, Plot  # Plot is needed for pickle loading
from abc import ABC, abstractmethod
from evaluation.segmentation import *
from evaluation.plot import def_tick_format
import itertools
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
import pickle
from random import sample
import logging


# If you get "Cannot connect to X display" errors, run this script as:
# MPLBACKEND=Agg python segmentation_evaluation_plot_and_save_ge.py
# matplotlib should handle this automatically but it can fail when $DISPLAY is set but an X-server is not available (such as when connecting remotely with ssh -Y without running an X-server on the client)


# Constants
FIG_EXTS = 'png', 'pdf', 'svg', 'eps'
CMAP = plt.cm.plasma
ZOOMED_LIMS = .6, .95
model_complexity = {
	'mu-net': (409e3, 180e9),
	'rgb-ss-eye-ms': (22.6e6, None),
	'sclerasegnet': (59e6, 17.94e9),
	'sclerau-net2': (3e6, 180e9)
}

# Auxiliary stuff
TRAIN_TEST_DICT = {train: [test for test in TEST_DATASETS if test not in train and not (train == 'All' and test == 'SMD')] for train in TRAIN_DATASETS}  # Filter out invalid train-test configurations
TRAIN_TEST = [(train, test) for train, tests in TRAIN_TEST_DICT.items() for test in tests]  # List of all valid train-test configurations
FIG_EXTS = ensure_iterable(FIG_EXTS, True)
fmt = lambda x: np.format_float_positional(x, precision=3, unique=False)
logging.getLogger('matplotlib.backends.backend_ps').addFilter(lambda record: 'PostScript backend' not in record.getMessage())  # Suppress matplotlib warnings about .eps transparency


class Main:
	def __init__(self, *args, **kw):
		# Default values
		root = Path('')
		self.models = Path(args[0] if len(args) > 0 else kw.get('models', root/'Models'))
		self.save = Path(args[1] if len(args) > 1 else kw.get('save', 'Results'))
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
		#self.bin_only = smap(str.lower, ensure_iterable(self.extra.get('no_roc', ('ScleraMaskRCNN')), True))

		plt.rcParams['font.family'] = 'Times New Roman'
		plt.rcParams['font.weight'] = 'normal'
		plt.rcParams['font.size'] = 24

		print("Sorting models by their mean binary F1-Score")
		self._sorted_models = sorted(os.listdir(self.models), key=lambda model: np.array([self._load(model, f'{train}_{test}')[1].f1score.mean for train in TRAIN_DATASETS for test in TEST_DATASETS]).mean(), reverse=True)

		self._results = treedict()
		for model in self._sorted_models:
			for f in (self.models/model/'Pickles').iterdir():
				train, test, *attr = f.stem.split('_')
				if attr:
					attr_name, attr_val = attr[0].split('=')
					self._results[model][train][test][attr_name][attr_val] = self._load(model, f.stem)
				else:
					self._results[model][train][test]['overall'] = self._load(model, f.stem)

		self._experiment1()
		for attr in ATTR_EXP:
			self._experiment2(attr)
		self._experiment3()

		if self.plot:
			plt.show()

	def _experiment1(self):
		print("Experiment 1: Overall performance")
		with Bar('Overall', self.fig_dir, self._sorted_models, len(TRAIN_TEST)) as bar, (self.eval_dir/'LaTeX - Performance.txt').open('w', encoding='utf-8') as latex:
			for i, model in enumerate(self._sorted_models):
				colours = dzip(TRAIN_TEST_DICT, CMAP(np.linspace(0, 1, len(TRAIN_TEST_DICT))))
				for j, (train, test) in enumerate(TRAIN_TEST):
					pred_eval, bin_eval = self._results[model][train][test]['overall'][:2]
					self._save_evals(pred_eval, bin_eval, model, train, test, latex)
					bar.plot(bin_eval, i, j, colour=colours[train], label=f"{train}-{test}")

	def _experiment2(self, attr):
		attr = attr.title()
		print(f"Experiment 2: Bias across different {attr}s")
		with (self.eval_dir/'LaTeX - Bias - Colour.txt').open('w', encoding='utf-8') as latex:
			for model in self._sorted_models:
				for train, test in TRAIN_TEST:
					if attr not in self._results[model][train][test]:
						continue
					bias = self._compute_biases(self._results[model][train][test][attr].values())
					self._save_biases(bias, f'{model} - {train} - {attr}')
					self._latexify_biases(bias, model, train, latex)

	def _experiment3(self):
		print("Experiment 3: Bias across different evaluation datasets")
		with (self.eval_dir/'LaTeX - Bias - Test data.txt').open('w', encoding='utf-8') as latex:
			for model in self._sorted_models:
				train_datasets = [train for train in TRAIN_DATASETS if train != 'All' and 'SMD' not in train]  # Skip models trained on SMD
				for train in train_datasets:
					bias = self._compute_biases([test['overall'][:2] for test in self._results[model][train].values()])
					self._save_biases(bias, f'{model} - {train}')
					self._latexify_biases(bias, model, train, latex, train_datasets=train_datasets)

	""" def _experiment3(self):
		print("Experiment 3: Performance across complexities")
		models_with_size = [model for model in self._sorted_models if model.lower() in model_complexity]
		models_with_both = [model for model in models_with_size if model_complexity[model.lower()][1] is not None]
		size_colours = CMAP(np.linspace(0, 1, len(models_with_size)))
		both_colours = CMAP(np.linspace(0, 1, len(models_with_both)))

		with Scatter('Size', self.fig_dir) as size, Scatter('Both', self.fig_dir) as both:
			for model, colour in zip(models_with_size, size_colours):
				bin_eval = self._load(model, 'Overall')[1]
				size.plot(model_complexity[model.lower()][0], bin_eval.f1score.mean, label=model, colour=colour)
				both.plot(model_complexity[model.lower()][0], bin_eval.f1score.mean, model_complexity[model.lower()][1], label=model, colour=colour)
		with Scatter('Full Only', self.fig_dir) as full:
			for model, colour in zip(models_with_both, both_colours):
				bin_eval = self._load(model, 'Overall')[1]
				full.plot(model_complexity[model.lower()][0], bin_eval.f1score.mean, model_complexity[model.lower()][1], label=model, colour=colour) """

	def _load(self, model, name):
		name = self.models/model/f'Pickles/{name}.pkl'
		if not name.is_file():
			raise ValueError(f"{name} does not exist")
		print(f"Loading data from {name}")
		with open(name, 'rb') as f:
			return pickle.load(f) + pickle.load(f)

	def _save_evals(self, pred_eval, bin_eval, model, train, test, latex=None):
		save = self.eval_dir/f'{model} - {train} - {test}.txt'
		print(f"Saving to {save}")
		with save.open('w', encoding='utf-8') as f:
			print("Probabilistic", file=f)
			print(pred_eval, file=f)
			print(file=f)
			print("Binarised", file=f)
			print(bin_eval, file=f)
		if latex:
			self._latexify_evals(pred_eval, bin_eval, model, train, test, latex)

	def _latexify_evals(self, pred_eval, bin_eval, model, train, test, latex):
		if (train, test) == TRAIN_TEST[0]:  # First line of model
			latex.write(fr"\multirow{{{len(TRAIN_TEST)}}}{{*}}{{{model}}}")
		latex.write(" &")
		if test == TRAIN_TEST_DICT[train][0]:  # First line of model+train
			latex.write(fr" \multirow{{{len(TRAIN_TEST_DICT[train])}}}{{*}}{{{train}}}")
		latex.write(f" & {test}")
		for pb_eval, metrics in ((bin_eval, ('F1-score', 'Precision', 'Recall', 'IoU')), (pred_eval, ('F1-score', 'AUC'))):
			for metric in metrics:
				latex.write(fr" & ${fmt(pb_eval[metric].mean)} \pm {fmt(pb_eval[metric].std)}$ &")
				if test == TRAIN_TEST_DICT[train][0]:  # First line of model+train
					mean = np.mean([self._results[model][train][t]['overall'][pb_eval is bin_eval][metric].mean for t in TRAIN_TEST_DICT[train]])
					latex.write(fr" \multirow{{{len(TRAIN_TEST_DICT[train])}}}{{*}}{{{fmt(mean)}}}")
		latex.write(" \\\\")
		if test == TRAIN_TEST_DICT[train][-1] and train != list(TRAIN_TEST_DICT)[-1]:  # Last line of model+train but not of model
			latex.write(r"\cmidrule(lr){2-15}")
		elif train == list(TRAIN_TEST_DICT)[-1] and model != self._sorted_models[-1]:  # Last line of model but not last line overall
			latex.write(r"\hline")
		print(file=latex)

	def _compute_biases(self, pb_evals, n_samples=None):
		# By default both stratified (with 100 samples per group) and non-stratified experiments will be run
		if n_samples is None:
			bias = self._compute_biases(pb_evals, 100)
			n_samples = 0
		else:
			bias = treedict()

		for pb in range(2):
			groups = [pb_eval[pb].f1score.last('all') for pb_eval in pb_evals]  # 2D list of per-image F1s across different groups
			total_eval = iter(shuffle(list(itertools.chain.from_iterable(groups))))  # 1D list (actually iterator) of all per-image F1s
			if n_samples:
				groups = [sample(group, n_samples) for group in groups]  # 2D list of per-image F1s across stratified groups
				group_means = np.array(lmap(np.mean, groups))  # 1D array of per-stratified-group mean F1s
				group_stds = np.array(lmap(np.std, groups))  # 1D array of per-stratified-group σ of F1
			else:
				group_means = np.array([pb_eval[pb].f1score.mean for pb_eval in pb_evals])  # 1D array of per-group mean F1s
				group_stds = np.array([pb_eval[pb].f1score.std for pb_eval in pb_evals])  # 1D array of per-group σ of F1
			control_groups = [list(itertools.islice(total_eval, len(group))) for group in groups]  # 2D list of per-image F1s across control groups
			control_means = np.array(lmap(np.mean, control_groups))  # 1D array of per-control-group mean F1s

			strat = bool(n_samples)  # Stratified?
			bias[strat][pb]['σ'] = group_means.std()  # Scalar σ of F1 means across groups
			bias[strat][pb]['MAD'] = np.mean(np.abs(group_means - group_means.mean()))  # Scalar mean absolute deviation of F1 means across groups
			bias[strat][pb]['Fisher'] = bias[strat][pb]['σ'] / control_means.std()  # Scalar ratio between σ of F1 means across groups and σ of F1 means across control groups
			bias[strat][pb]['Vito'] = bias[strat][pb]['σ'] / group_stds.mean()  # Scalar ratio between σ of F1 means across groups and the mean σ of F1 within groups

		return bias

	def _save_biases(self, bias, name):
		for strat, name in enumerate((name, f'{name} - Stratified')):
			save = self.eval_dir/f'Bias - {name}.txt'
			print(f"Saving to {save}")
			with save.open('w', encoding='utf-8') as f:
				for i, pb in enumerate(("Probabilistic", "Binarised")):
					print(pb, file=f)
					for name, score in bias[strat][i].items():
						print(f"{name}: {score}", file=f)
					print(file=f)

	def _latexify_biases(self, bias, model, train, latex, train_datasets=TRAIN_DATASETS):
		if train == train_datasets[0]:  # First line of model
			latex.write(fr"\multirow{{{len(train_datasets)}}}{{*}}{{{model}}}")
		latex.write(f" & {train if train == train_datasets[0] else train.ljust(max(map(len, train_datasets)))}")
		for pb in (1, 0):  # 1 = probabilistic, 0 = binarised
			for metric in ('σ', 'MAD', 'Fisher', 'Vito'):
				latex.write("".join(f" & ${fmt(bias[strat][pb][metric])}$" for strat in (False, True)))
		latex.write(" \\\\")
		if train == train_datasets[-1] and model != self._sorted_models[-1]:  # Last line of model but not last line overall
			latex.write(r"\hline")
		print(file=latex)

	def process_command_line_options(self):
		ap = argparse.ArgumentParser(description="Evaluate segmentation results.")
		ap.add_argument('models', type=Path, nargs='?', default=self.models, help="directory with model information")
		ap.add_argument('save', type=Path, nargs='?', default=self.save, help="directory to save figures and evaluations to")
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


class ROC(Figure):
	def __init__(self, name, save_dir, fontsize=20):
		super().__init__(f'{name} ROC', save_dir, fontsize)
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
		self.save(f'{self.name} (Zoomed)')

		if labels:
			legend.remove()
			self.save(f'{self.name} (Zoomed, No Legend)')

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
	def __init__(self, name, save_dir, groups, n=1, fontsize=30, margin=.2):
		super().__init__(f'{name} bar', save_dir, fontsize)
		self.groups = groups
		self.m = len(groups)
		self.n = n
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
		self.ax.set_ylabel("F1-Score")
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
		ymin = max(self.min - .01, 0) if self.min != float('inf') else 0
		ymax = self.max + .01 if self.max != float('-inf') else 1.01
		self.ax.set_ylim(ymin, ymax)
		self.ax.set_xticks(np.arange(self.m) + (self.margin + self.n * self.width) / 2)
		self.ax.set_xticklabels(self.groups, rotation=60, ha='right', rotation_mode='anchor')
		if ymax - ymin >= .35:
			self.ax.yaxis.set_major_locator(MultipleLocator(.1))
			self.ax.yaxis.set_minor_locator(MultipleLocator(.05))
		else:
			self.ax.yaxis.set_major_locator(MultipleLocator(.05))
			self.ax.yaxis.set_minor_locator(MultipleLocator(.025))
		self.save()
		self.ax.tick_params(axis='x', bottom=False, labelbottom=False)
		self.save(f'{self.name} (No Labels)')

	def plot(self, evaluation, group=0, index=0, *, label=None, colour=None):
		super().plot()
		plt.rcParams['font.size'] = 10
		err_w = np.clip(self.width * 10, 2, 5)
		self.ax.bar(
			group + self.margin / 2 + index * self.width,
			evaluation.f1score.mean,
			yerr=evaluation.f1score.std,
			error_kw=dict(lw=err_w, capsize=1.5 * err_w, capthick=.5 * err_w),
			width=self.width,
			align='edge',
			label=label,
			color=colour
		)
		self.min = min(self.min, evaluation.f1score.mean - evaluation.f1score.std)
		self.max = max(self.max, evaluation.f1score.mean + evaluation.f1score.std)


class Scatter(Figure):
	def __init__(self, name, save_dir, fontsize=28):
		super().__init__(f'{name} scatter', save_dir, fontsize)
		self.min = None
		self.max = None

	def __enter__(self):
		super().__enter__()
		self.ax.grid(axis='y', which='major', alpha=.5)
		self.ax.grid(axis='y', which='minor', alpha=.2)
		self.ax.set_xscale('log')
		self.ax.yaxis.set_major_formatter(FuncFormatter(def_tick_format))
		self.ax.margins(0)
		self.ax.set_xlabel("# Parameters")
		self.ax.set_ylabel("F1-Score")
		self.fig.tight_layout(pad=0)
		self.min = float('inf')
		self.max = float('-inf')
		return self

	def close(self, *args, **kw):
		ymin = max(self.min - .1, 0) if self.min != float('inf') else 0
		ymax = self.max + .1 if self.max != float('-inf') else 1.01
		self.ax.set_xlim(1, 3e8)
		self.ax.set_ylim(ymin, ymax)
		if ymax - ymin >= .35:
			self.ax.yaxis.set_major_locator(MultipleLocator(.1))
			self.ax.yaxis.set_minor_locator(MultipleLocator(.05))
		else:
			self.ax.yaxis.set_major_locator(MultipleLocator(.05))
			self.ax.yaxis.set_minor_locator(MultipleLocator(.025))
		self.save(f'{self.name} (No Legend)')
		if self.ax.get_legend_handles_labels()[0]:
			self.ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
		self.save()

	def plot(self, x, y, size=None, *, label=None, colour=None):
		super().plot()
		markersize = 8
		flopsize = 0
		if size:
			#flopsize = 2e-4 * math.sqrt(size)
			flopsize = 2 * math.log(size)
			self.ax.plot(x, y, 'o', markersize=flopsize, color=(*colour[:3], .3))
		self.ax.plot(x, y, 'o', markersize=markersize, label=label, color=colour)
		self.min = min(self.min, y)
		self.max = max(self.max, y)


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
