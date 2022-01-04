#!/usr/bin/env python3

# Always import these
from abc import abstractmethod
import argparse
from ast import literal_eval
from functools import partial
from matej import make_module_callable, StoreDictPairs
from matej.collections import DotDict, ensure_iterable
from matej.gui.tkinter import ToolTip
from pathlib import Path
import sys
import textwrap
from tkinter import *
import tkinter.filedialog as filedialog

# Import whatever else is needed
from joblib.parallel import Parallel, delayed
from matej.parallel import tqdm_joblib
import numpy as np
from PIL import Image
from tqdm import tqdm


# f = /asdf/vsdaj/iosad.goi
# fname = iosad.goi
# bname = iosad
# ext = .goi
# dir = /asdf/vsdaj(/)


# Constants
IMG_EXTS = '.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'


class Main(DotDict):
	def __init__(self):
		#TEMPLATE: Program description. Either a string or a tuple of two strings (one short, one longer)
		self._description = "Combine predictions"

		#TEMPLATE: Automatic arguments. These are added to the argument parser and GUI automatically.
		self._automatic = dict(
			# Path arguments
			# argument=PathArg(default value, help, *filetypes)
			# Regarding filetypes: if directory, leave empty; if file but you don't want to give an extension list, use None; otherwise see https://stackoverflow.com/a/44403840/5769814
			models=PathArg('Models', "Directory with models"),

			# Boolean arguments
			# argument=BoolArg(default value, help, *possible additional argument flags (--argument-name will be added automatically at the start if not present))
			# Note that this removes the ability to combine flags like -of (you have to use -o -f instead)

			# Choice arguments
			# argument=ChoiceArg(default value, (choice1, choice2, ...), help, *argument flags, choice_descriptions=(), type=None)
			# If type not passed (or None), will try to infer (int -> float -> str)

			# Single number arguments
			# argument=NumberArg(default value, (min, max, step), help, *argument flags, type=None, gui_type='slider')
			# If type not passed (or None), will try to infer (int -> float). gui_type can be 'slider' or 'spinbox'
		)
		# Add all the above to argument list
		for attr, arg in self._automatic.items():
			self[attr] = arg.default

		#TEMPLATE: Other arguments. You'll have to manually add these to the argument parser and GUI.
		# self.complicated_argument = ["Default", "Value"]

		#TEMPLATE: Which arguments can be passed as varargs (paths are added by default)
		self._varargs = [attr for attr, arg in self._automatic.items() if isinstance(arg, PathArg)]
		# self._varargs.append(argument_name)

		#TEMPLATE: Other instance attributes you want to declare (these should start with _ so they don't get mixed up with the arguments)
		# self._result = None

		# Extra keyword arguments passed in with -e flag or in extra Key-Value frames in the GUI
		self.extra = DotDict()

	def args(self):
		return {k: v for k, v in self.items() if not k.startswith('_')}

	def __str__(self):
		return str(self.args())

	# Support for *-unpacking arguments
	def __iter__(self):
		return iter([self[attr] for attr in self._varargs])

	# Support for **-unpacking arguments
	def keys(self):
		return self.args().keys()

	##############################
	# Main code                  #
	##############################

	def __call__(self):
		for ensemble in (
			(('RGB-SS-Eye-MS', .726), ('ScleraU-Net2', .74), ('FCN8', .8)),  # Eye colour  (can try all MOBIUS clusters with all attributes tbh)
			(('RGB-SS-Eye-MS', .778), ('CGANs2020CL', .765), ('FCN8', .741), ('ScleraMaskRCNN', .535)),  # Evaluation data
			(('RGB-SS-Eye-MS', .726), ('ScleraU-Net2', .74), ('FCN8', .8), ('ScleraSegNet', .748)),  # Lighting
			(('RGB-SS-Eye-MS', .726), ('ScleraU-Net2', .74), ('ScleraSegNet', .748)),  # Phone
			(('ScleraU-Net2', .742), ('FCN8', .741), ('ScleraMaskRCNN', .535)),  # Training data
		):
			ensemble_dir = self.models/"+".join(model for model, _ in ensemble)
			ensemble_dir.mkdir(parents=True, exist_ok=True)
			filelist_dir = self.models/ensemble[0][0]
			filelist = [f for f in filelist_dir.rglob('*.png') if 'Binarised' not in str(f)]

			with tqdm_joblib(tqdm(filelist, desc=f"Combining {ensemble}")):
				Parallel(n_jobs=-1)(
					delayed(self._process_image)(f.relative_to(filelist_dir), ensemble, ensemble_dir)
					for f in filelist
				)

	def _process_image(self, relative_f, ensemble, ensemble_dir):
		pred_f = ensemble_dir/relative_f
		pred_f.parent.mkdir(parents=True, exist_ok=True)
		bin_f = Path(str(pred_f).replace('Predictions', 'Binarised'))
		bin_f.parent.mkdir(parents=True, exist_ok=True)

		models, weights = zip(*ensemble)
		images = [np.array(Image.open(self.models/model/relative_f).convert('L').resize((480, 360)), dtype=float) for model in models]
		combined = np.average(images, axis=0, weights=weights).astype(np.uint8)

		pred = Image.fromarray(combined)
		bin_ = Image.fromarray(combined >= 128)

		pred.save(pred_f)
		bin_.save(bin_f)

	##############################
	# End of main code           #
	##############################

	# Make this class callable with arguments too without cluttering the __call__ method
	def run(self, *args, **kw):
		# Update with varargs
		kw.update(dict(zip(self._varargs, map(Path, args))))

		# Update with passed parameters
		for arg, value in list(kw.items()):
			if arg in self:
				self[arg] = value
				del kw[arg]
		self.extra.update(kw)

		return self()

	#TEMPLATE: CLI arguments (don't need to touch this if you didn't add manual arguments)
	def process_command_line_options(self):
		# Use long description (or the first one if there's only 1)
		ap = argparse.ArgumentParser(description=(ensure_iterable(self._description, True) * 2)[1], formatter_class=_CustomArgFormatter)

		for attr, arg in self._automatic.items():
			if isinstance(arg, PathArg):
				arg.flags = attr
			else:
				name_flag = '--' + attr.replace("_", "-")
				if name_flag not in arg.flags:
					arg.flags = (name_flag, *arg.flags)
			arg.add_to_ap(ap)

		#TEMPLATE: Add your manual arguments here
		# ap.add_argument('-c', '--complicated-argument', nargs=2, help="this is a complicated argument")

		ap.add_argument('--gui', '--force-gui', dest='_force_gui', action='store_true', help="force GUI (while keeping existing arguments)")
		ap.add_argument('-e', '--extra', nargs='+', action=StoreDictPairs, help="any extra keyword-value argument pairs")
		ap.parse_args(namespace=self)

		if self._force_gui:
			return self.gui()
		return True

	def gui(self):
		gui = GUI(self)
		gui.mainloop()
		return gui.ok


class GUI(Tk):
	#TEMPLATE: GUI for arguments (don't need to touch this if you didn't add manual arguments)
	def __init__(self, argspace, *args, **kw):
		super().__init__(*args, **kw)
		self.args = argspace
		self.ok = False

		self.title(ensure_iterable(self.args._description, True)[0])
		frame = Frame(self)
		frame.pack(fill=BOTH, expand=YES)

		self.auto_vars = {}
		path_frame = Frame(frame)
		path_frame.grid_columnconfigure(1, weight=1)
		path_row = 0
		chk_frame = Frame(frame)
		rad_frame = Frame(frame)
		rad_frame.grid_columnconfigure(1, weight=1)
		rad_row = 0
		sld_frame = Frame(frame)
		sld_frame.grid_columnconfigure(1, weight=1)
		sld_row = 0
		spin_frame = Frame(frame)
		spin_row = 0
		for attr, arg in self.args._automatic.items():
			if isinstance(arg, PathArg):
				self.auto_vars[attr] = PathVar(value=self.args[attr])
				lbl = Label(path_frame, text=f"{arg.name if arg.name else attr.title().replace('_', ' ')}:")
				lbl.grid(row=path_row, column=0, sticky='w')
				txt = Entry(path_frame, width=60, textvariable=self.auto_vars[attr])
				txt.grid(row=path_row, column=1)
				command = partial(self._browse_file, self.auto_vars[attr], arg.exts) if arg.exts else partial(self._browse_dir, self.auto_vars[attr])
				btn = Button(path_frame, text="Browse", command=command)
				btn.grid(row=path_row, column=2)
				ToolTip(lbl, arg.help)
				ToolTip(txt, arg.help)
				ToolTip(btn, arg.help)
				path_row += 1

			elif isinstance(arg, ChoiceArg):
				var_type = IntVar if arg.type is int else DoubleVar if arg.type is float else StringVar
				self.auto_vars[attr] = var_type(value=self.args[attr])
				text = arg.name if arg.name else arg.help if len(arg.help) <= 20 else attr.title().replace('_', ' ')
				text += ":"
				lbl = Label(rad_frame, text=text)
				lbl.grid(row=rad_row, column=0, sticky='w')
				current_frame = Frame(rad_frame)
				for choice in arg.choices:
					rad = Radiobutton(current_frame, text=str(choice), variable=self.auto_vars[attr], value=choice)
					rad.pack(side=LEFT)
				current_frame.grid(row=rad_row, column=1, sticky='w')
				rad_row += 1

			elif isinstance(arg, BoolArg):
				self.auto_vars[attr] = BooleanVar(value=self.args[attr])
				text = arg.name if arg.name else arg.help if len(arg.help) <= 80 else attr.title().replace('_', ' ')
				chk = Checkbutton(chk_frame, text=text, variable=self.auto_vars[attr], anchor='w')
				if text in {arg.name, attr.title().replace('_', ' ')}:
					ToolTip(chk, arg.help)
				chk.pack(fill=X, expand=YES)

			elif isinstance(arg, NumberArg):
				var_type = IntVar if arg.type is int else DoubleVar
				self.auto_vars[attr] = var_type(value=self.args[attr])
				text = f"{arg.name if arg.name else attr.title().replace('_', ' ')}:"
				if arg.gui_type.lower() == 'spinbox':
					lbl = Label(spin_frame, text=text)
					lbl.grid(row=spin_row, column=0, sticky='w')
					spin = Spinbox(spin_frame, from_=arg.min, to=arg.max, increment=arg.step, textvariable=self.auto_vars[attr])
					spin.grid(row=spin_row, column=1, sticky='w')
					spin_row += 1
				elif arg.gui_type.lower() in {'slider', 'scale'}:
					lbl = Label(sld_frame, text=text)
					lbl.grid(row=sld_row, column=0, sticky='w')
					sld = Scale(sld_frame, from_=arg.min, to=arg.max, resolution=arg.step, variable=self.auto_vars[attr], orient=HORIZONTAL)
					sld.grid(row=sld_row, column=1, sticky='we')
					sld_row += 1

			else:
				raise ValueError("Unknown argument type in automatic GUI construction")

		path_frame.pack(fill=X, expand=YES)
		rad_frame.pack(fill=X, expand=YES)
		chk_frame.pack(fill=X, expand=YES)
		sld_frame.pack(fill=X, expand=YES)
		spin_frame.pack(fill=X, expand=YES)

		#TEMPLATE: Add manual arguments here (can add them to the above auto frames as well)

		self.extra_frame = ExtraFrame(frame)
		for key, value in self.args.extra.items():
			self.extra_frame.add_pair(key, str(value))
		self.extra_frame.pack(fill=X, expand=YES)

		ok_btn = Button(frame, text="OK", command=self.confirm)
		ok_btn.pack()
		ok_btn.focus()

	def _browse_dir(self, target_var):
		init_dir = target_var.get()
		while not init_dir.is_dir():
			init_dir = init_dir.parent

		new_entry = filedialog.askdirectory(parent=self, initialdir=init_dir)
		if new_entry:
			target_var.set(new_entry)

	def _browse_file(self, target_var, exts=None):
		init_dir = target_var.get().parent
		while not init_dir.is_dir():
			init_dir = init_dir.parent

		if exts:
			new_entry = filedialog.askopenfilename(parent=self, filetypes=exts, initialdir=init_dir)
		else:
			new_entry = filedialog.askopenfilename(parent=self, initialdir=init_dir)

		if new_entry:
			target_var.set(new_entry)

	#TEMPLATE: Reading arguments from GUI (don't need to touch this if you didn't add manual arguments)
	def confirm(self):
		for attr, var in self.auto_vars.items():
			self.args[attr] = var.get()

		self.args.extra.clear()
		for kw in self.extra_frame.pairs:
			key, value = kw.key_txt.get(), kw.value_txt.get()
			if key:
				d = self.args if key in self.args else self.args.extra
				try:
					d[key] = literal_eval(value)
				except ValueError:
					d[key] = value

		self.ok = True
		self.destroy()


class PathVar(Variable):
	def __init__(self, *args, value=None, **kw):
		super().__init__(*args, value=Path(value), **kw)

	def get(self):
		return Path(super().get())

	def set(self, value):
		return super().set(Path(value))


class ExtraFrame(Frame):
	def __init__(self, *args, **kw):
		super().__init__(*args, **kw)
		self.pairs = []

		self.key_lbl = Label(self, width=30, text="Key", anchor='w')
		self.value_lbl = Label(self, width=30, text="Value", anchor='w')

		self.add_btn = Button(self, text="+", command=self.add_pair)
		self.add_btn.grid()
		ToolTip(self.add_btn, "Add a new key-value pair")

	def add_pair(self, key="", value=""):
		pair_frame = KWFrame(self, pady=2, key=key, value=value)
		self.pairs.append(pair_frame)
		pair_frame.grid(row=len(self.pairs), columnspan=3, sticky='w')
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
	def __init__(self, *args, key="", value="", **kw):
		super().__init__(*args, **kw)

		self.key_txt = Entry(self, width=30)
		self.key_txt.insert(0, key)
		self.key_txt.grid(column=0, row=0, sticky='w')

		self.value_txt = Entry(self, width=30)
		self.value_txt.insert(0, value)
		self.value_txt.grid(column=1, row=0, sticky='w')

		remove_btn = Button(self, text="-", command=self.remove)
		remove_btn.grid(column=2, row=0)
		ToolTip(remove_btn, "Remove this key-value pair")

	def remove(self):
		i = self.master.pairs.index(self)
		del self.master.pairs[i]
		for pair in self.master.pairs[i:]:
			pair.grid(row=pair.grid_info()['row'] - 1)
		self.master.update_labels_and_button()
		self.destroy()


# Custom formatter that respects \n characters
class _CustomArgFormatter(argparse.RawTextHelpFormatter):
	def _split_lines(self, text, width):
		text = super()._split_lines(text, width)
		new_text = []

		# loop through all the lines to create the correct wrapping for each line segment.
		for line in text:
			if not line:
				# this would be a new line.
				new_text.append(line)
				continue

			# wrap the line's help segment which preserves new lines but ensures line lengths are honored
			new_text.extend(textwrap.wrap(line, width))

		return new_text

# Wrappers for automatic arguments
class _AutoArg:  # Can't use ABC because it breaks pickling
	def __init__(self, default, help, *flags, gui_name=None):
		self.default = default
		self.help = help
		self.flags = flags
		self.name = gui_name

	@abstractmethod
	def add_to_ap(self, ap):
		pass


class PathArg(_AutoArg):
	def __init__(self, default, help, *extensions, **kw):
		super().__init__(default, help, **kw)
		self.exts = extensions

	def add_to_ap(self, ap):
		ap.add_argument(self.flags, type=Path, nargs='?', default=self.default, help=self.help.lower())


class BoolArg(_AutoArg):
	def add_to_ap(self, ap):
		name = next((arg for arg in self.flags if arg[:2] == '--'), self.flags[0]).strip('-')
		help = self.help.lower()

		no_f = lambda arg: '--no-' + arg.strip('-') if arg[:2] == '--' else None
		str_to_bool = lambda s: s.lower() in {'true', 'yes', 't', 'y', '1'}

		group = ap.add_mutually_exclusive_group()
		group.add_argument(*self.flags, dest=name, nargs='?', default=self.default, const=True, type=str_to_bool, help=help)
		group.add_argument(*filter(None, map(no_f, self.flags)), dest=name, action='store_false', help="do not " + help)


class ChoiceArg(_AutoArg):
	def __init__(self, default, choices, help, *flags, choice_descriptions=(), type=None, **kw):
		super().__init__(default, help, *flags, **kw)
		self.choices = choices
		self.choice_descriptions = choice_descriptions
		self.type = type

		if self.type is None:
			try:
				self.type = int if all(int(x) == x for x in self.choices) else float
			except ValueError:
				if all(isinstance(x, str) for x in self.choices):
					self.type = str

	def add_to_ap(self, ap):
		help = self.help.lower()
		if self.choice_descriptions:
			longest = max(len(str(choice)) for choice in self.choices)
			help += "\n" + "\n".join(f"\t{choice:>{longest}}: {description}" for choice, description in zip(self.choices, self.choice_descriptions)) + "\n"
		ap.add_argument(*self.flags, type=self.type, default=self.default, choices=self.choices, help=help)


class NumberArg(_AutoArg):
	def __init__(self, default, range, help, *flags, type=None, gui_type='slider', **kw):
		super().__init__(default, help, *flags, **kw)
		self.type = type
		if self.type is None:
			self.type = int if all(int(x) == x for x in range) else float
		self.min, self.max = range[:2]
		self.step = range[2] if len(range) > 2 else 1 if self.type is int else .01
		self.range = self.min, self.max, self.step
		self.gui_type = gui_type

	def _type(self, x):
		x = self.type(x)
		if not self.min <= x <= self.max:
			raise ValueError(f"{self.flags[0]} {x} should be between {self.min} and {self.max}")
		return x

	def add_to_ap(self, ap):
		ap.add_argument(*self.flags, type=self._type, default=self.default, help=self.help.lower() + f" [{self.min} <= x <= {self.max}]")


if __name__ == '__main__':
	main = Main()

	# If CLI arguments, read them
	if len(sys.argv) > 1:
		if not main.process_command_line_options():
			# If CLI arguments weren't processed successfully, exit
			sys.exit(2)

	# Otherwise get them from a GUI
	else:
		if not main.gui():
			# If GUI was cancelled, exit
			sys.exit(1)

	main()

else:
	# Make module callable (python>=3.5)
	make_module_callable(__name__, Main().run)