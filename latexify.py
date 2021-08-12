#!/usr/bin/env python3
from matej.collections import treedict
import numpy as np
from pathlib import Path


root = Path('/root/GE/Results/Evaluations')


performance = treedict()
for f in filter(lambda f: not f.stem.startswith('Bias'), root.iterdir()):
	model, train, test = f.stem.split(' - ')

	with f.open('r', encoding='utf-8') as in_file:
		pb = None
		for line in in_file.readlines():
			line = line.strip()
			if line in {'Probabilistic', 'Binarised'} or line == '':
				if line != '':
					pb = line
				continue
			metric, value = line.split(': ')
			metric = metric.rstrip(' (μ ± σ)')
			mean, std = map(float, value.split(' ± '))
			performance[model][train][test][pb][metric] = mean, std
			performance[model][train]['mean'][pb][metric] = np.array([performance[model][train][t][pb][metric][0] for t in performance[model][train] if t != 'mean']).mean()

bias = treedict()
for f in root.glob('Bias*'):
	_, model, train, *experiment = f.stem.split(' - ')
	if (stratified := 'Stratified' in experiment):
		experiment.remove('Stratified')
	if not experiment:
		experiment = "Test data"
	else:
		experiment = experiment[0]

	with f.open('r', encoding='utf-8') as in_file:
		pb = None
		for line in in_file.readlines():
			line = line.strip()
			if line in {'Probabilistic', 'Binarised'} or line == '':
				if line != '':
					pb = line
				continue
			metric, value = line.split(': ')
			bias[experiment][model][train][pb][metric][stratified] = float(value)

sorted_models = sorted(performance, key=lambda model: np.array([performance[model][train]['mean']['Binarised']['F1-score'] for train in performance[model]]).mean(), reverse=True)
fmt = lambda x: np.format_float_positional(x, precision=3, unique=False)

print("Overall performance:")
for model in sorted_models:
	print(fr"\multirow{{{sum(len(tests) for tests in performance[model].values())}}}{{*}}{{{model}}}", end="")
	for train, tests in performance[model].items():
		tests = {test: score for test, score in tests.items() if test != 'mean'}  # means are recorded inline, so remove them from iteration
		print(fr" & \multirow{{{len(tests)}}}{{*}}{{{train}}}", end="")
		for test, score in tests.items():
			# If not first line for current model+train, we need to add an extra &
			if test != list(tests)[0]:
				print(" &", end="")
			print(f" & {test}", end="")
			for pb, metrics in (('Binarised', ('F1-score', 'Precision', 'Recall', 'IoU')), ('Probabilistic', ('F1-score', 'AUC'))):
				for metric in metrics:
					print(" & $" + r" \pm ".join(map(fmt, score[pb][metric])) + "$", end=" &")
					if test == next(iter(tests)):  # if first line for current model+train
						print(fr" \multirow{{{len(tests)}}}{{*}}{{{fmt(performance[model][train]['mean'][pb][metric])}}}", end="")
			print(" \\\\", end="")
			# If not last line for current model+train, we can end the line here
			if test != list(tests)[-1]:
				print()
		# If not last line for current model, we make a cmidrule
		if train != list(performance[model])[-1]:
			print(r"\cmidrule(lr){2-15}")
	# If not last line overall, we add an hline
	if model != sorted_models[-1]:
		print(r"\hline")
	else:
		print()

for exp in bias:
	print()
	print(f"Bias across {exp}:")
	for model in sorted_models:
		print(fr"\multirow{{{len(bias[exp][model])}}}{{*}}{{{model}}}", end="")
		for train, score in bias[exp][model].items():
			print(f" & {train if train == next(iter(bias[exp][model])) else train.ljust(max(map(len, bias[exp][model])))}", end="")
			for pb in ('Binarised', 'Probabilistic'):
				for metric in ('σ', 'MAD', 'Fisher'):
					print(f" & ${fmt(score[pb][metric][False])}$ & ${fmt(score[pb][metric][True])}$", end="")
			print(" \\\\", end="")
			# If not last line for current model, we can end the line here
			if train != list(bias[exp][model])[-1]:
				print()
		# If not last line overall, we add an hline
		if model != sorted_models[-1]:
			print(r"\hline")
		else:
			print()
