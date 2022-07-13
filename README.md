# About
This code was used to produce the evaluation results for the paper ["Exploring Bias in Sclera Segmentation Models: A Group Evaluation Approach"](https://sclera.fri.uni-lj.si/publications.html#TIFS_2022).

It was adapted from the [evaluation code from SSBC 2020](https://github.com/MatejVitek/SSBC). See the instructions in that repository for more information on adapting and running this codebase.

# Requirements
Our [utility libraries](https://sclera.fri.uni-lj.si/code.html#Libraries) are required to make this project work. The additional required python packages are:

	joblib
	matplotlib
	numpy
	pillow
	scikit-learn
	scipy
	tqdm
	opencv-python~=3.4.2

# Running the code
This project is a stripped-down version of [our Toolbox](https://sclera.fri.uni-lj.si/code.html#Toolbox). The project's functionality is divided into two parts—computation (slow and memory-intensive) and plotting (fast and efficient).

## Computation
The computation part takes as input the sclera masks output by your model(s), and the ground truth information. It computes the precision/recall/... information and saves it to `.pkl` files ("pickles"). All this is handled by the script `compute.py`. To run the script, use the following syntax:

	python compute.py "/path/to/model/results" "/path/to/ground/truth"

The directory `/path/to/model/results/` should contain a separate subdirectory for each of your segmentation models. If you are only computing results for a single segmentation model, use only one subdirectory. Inside each of these model subdirectories should be different folders for different training configurations. In our paper we used 5 different training configurations: `All`, `MASD+SBVPI`, `MASD+SMD`, `SBVPI`, and `SMD`. Inside each of these should be a folder for each testing dataset: `MOBIUS`, `SLD`, and `SMD`. Finally, inside each of these, there should be two folders: `Predictions` (which contains greyscale probabilistic sclera masks output by your model before thresholding) and `Binarised` (which contains the binary black & white sclera masks obtained after thresholding). See below for a sample tree structure with two models called Segmentor and Segmentator:

	/path/to/model/results/
	├── Segmentor/
	│   ├── All/
	│   │   ├── MOBIUS/
	│   │   │   ├── Predictions/
	│   │   │   │   ├── 1_1i_Ll_1.png
	│   │   │   │   ├── 1_1i_Ll_2.png
	│   │   │   │   ├── 1_1i_Lr_1.png
	│   │   │   │   └── ...
	│   │   │   └── Binarised/
	│   │   │       ├── 1_1i_Ll_1.png
	│   │   │       ├── 1_1i_Ll_2.png
	│   │   │       ├── 1_1i_Lr_1.png
	│   │   │       └── ...
	│   │   ├── SLD/
	│   │   │   ├── Predictions/
	│   │   │   │   ├── 1 (1).png
	│   │   │   │   └── ...
	│   │   │   └── Binarised/
	│   │   │       ├── 1 (1).png
	│   │   │       └── ...
	│   │   └── SMD/
	│   │       └── ...
	│   ├── MASD+SBVPI/
	│   │   ├── MOBIUS/
	│   │   │   └── ...
	│   │   ├── SLD/
	│   │   │   └── ...
	│   │   └── SMD/
	│   │       └── ...
	│   ├── MASD+SMD/
	│   │   └── ...
	│   ├── SBVPI/
	│   │   └── ...
	│   └── SMD/
	│       └── ...
	└── Segmentator/
	    └── ...

The directory `/path/to/ground/truth/` should contain the ground truth information bundled with the evaluation datasets. It should contain a subdirectory for each evaluation dataset (in our case `MOBIUS`, `SLD`, `SMD`). Each of these should contain the subfolder `Images` (containing the raw RGB images of the dataset) and `Masks` (containing the black-and-white sclera masks). All the files in `/path/to/ground/truth/` should have a corresponding entry in each of the `Predictions` and `Binarised` folders of all of your models in `/path/to/model/results/`, otherwise the evaluation cannot be executed fairly and an error will be reported.

Other arguments are not required, but can be useful. For a full list see the output of `compute.py --help`. For instance, a smaller image size (`-r`) can lead to less memory usage but will be less reliable in the calculations. The script can also be run without arguments, in which case it will launch as a simple GUI application.

## Plotting and evaluation
The plotting and quantitative evaluation is handled by `plot.py`. This script takes as input the `.pkl` files produced by `compute.py` and creates and saves various different plot figures and textual quantitative evaluations. To run the script, use the following syntax:

	python plot.py "/path/to/model/results" "/path/to/ground/truth" "/path/to/save/to"

The directory `/path/to/model/results/` should contain a subdirectory for each of your segmentation model(s). This subdirectory should contain the folder `Pickles` where the `.pkl` files produced by `compute.py` are located in the `Recognition` and `Segmentation` subdirectories. The `/path/to/ground/truth` should contain a subdirectory for each evaluation dataset, each of which should contain a `Samples.pkl` file, detailing the ordering of the samples in the dataset that was used in the computation. This tree structure will be produced automatically if you use the same `/path/to/model/results/` and `/path/to/ground/truth` for both scripts, but should be respected if you decide to use different directories.

If you want your model to be included in the generated scatterplot of model complexities as well, you will also have to add your model to `model_complexity` in `plot.py`. An entry in `model_complexity` consists of the model name (which should be lowercase but otherwise identical to the name of your model folder) and a tuple of: the number of trainable parameters (you can use 1 if your solution has no trainable parameters) and floating point operations (FLOP) required for a single forward pass (inference). See the existing entries for a template of how to add your own.

The plots and quantitative evaluations will be saved to `/path/to/save/to`.
