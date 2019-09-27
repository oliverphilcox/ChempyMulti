[![DOI](https://zenodo.org/badge/162767887.svg)](https://zenodo.org/badge/latestdoi/162767887)

# ChempyMulti - Multi-Star Bayesian Inference with Chempy
This is an updated version of the [*Chempy*](http://github.com/jan-rybizki/Chempy) software, including yield table scoring as described in Philcox, Rybizki & Gutke (2018, *ApJ*, [arXiv](https://arxiv.org/abs/1712.05686)), and multi-star Bayesian inference as described in Philcox & Rybizki (2019, submitted to *ApJ*, [arXiv](https://arxiv.org/abs/1909.00812)). This replaces the [*ChempyScoring*](https://github.com/oliverphilcox/ChempyScoring) package.

*Chempy* is a flexible one-zone open-box chemical evolution model, incorporating abundance fitting and stellar feedback calculations. We provide routines for parameter optimization for simulations and observational data and yield table scoring. 

## Tutorials
We provide the following tutorials on basic *Chempy* usage as well as yield table scoring and multi-parameter inference:
- [Inferring Galactic Parameters using Chemical Abundances from Multiple Stars](https://github.com/oliverphilcox/ChempyMulti/blob/master/Multi-Star%20Inference%20with%20Chempy%20-%20Tutorial.ipynb). This includes description of the new *Chempy* routines to compute the ISM abundances at any given simulation time, as well as a detailed guide to running Hamiltonian Monte Carlo inference using *Chempy*, neural networks and PyMC3 for mock or observational data. Analysis is based on the Philcox & Rybizki (2019, submitted to *ApJ*, [arXiv](https://arxiv.org/abs/1909.00812)) paper. 
- [Scoring Yield Tables and Choosing Parameters for Hydrodynamical Simulations using Proto-solar Abundances](https://github.com/oliverphilcox/ChempyMulti/blob/master/Scoring%20%26%20Parameter%20Choice%20Tutorial.ipynb). This describes how to score yield tables based on the MCMC methods of Philcox, Rybizki & Gutke (2018, *ApJ*, [arXiv](https://arxiv.org/abs/1712.05686)), including discussion of the neural network implementation and adding new yield tables to *Chempy*.
- [General *Chempy* Tutorials](https://github.com/jan-rybizki/Chempy/tree/master/tutorials). These describe the basic functionality of *Chempy* including running the simulation, performing inference using MCMC and producing SSP enrichment tables for hydrodynamical simulations. This is based on the Rybizki et al. (2017, *A&A*, [arXiv](https://arxiv.org/abs/1702.08729)) paper.

## Installation

```
pip install git+https://github.com/oliverphilcox/ChempyMulti.git
```
*Chempy* should run with the latest python 2 and python 3 versions.
Its dependencies are: [Numpy](http://numpy.scipy.org/), [SciPy](http://www.scipy.org/), [matplotlib](http://matplotlib.sourceforge.net/), [multiprocessing](https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing),  [corner](http://corner.readthedocs.io/en/latest/) (for MCMC plots) and [tqdm](https://pypi.python.org/pypi/tqdm) (for timing multiprocessing). For **yield table scoring** we additionally require [emcee](http://dan.iel.fm/emcee/current/) (for the Ensemble MCMC inference), [PyTorch](http://pytorch.org/) (for Neural Networks) and [scikit-monaco](https://pypi.python.org/pypi/scikit-monaco) (for Monte Carlo integration). For **multi-star inference** we instead require [scikit-learn](https://scikit-learn.org/stable/) (for Neural Networks) and [PyMC3](https://docs.pymc.io/) (for HMC inference).

These are all pip installable and you can also get part of it with [Anaconda](https://www.continuum.io/downloads).

### Installation without admin rights:
You can install *ChempyMulti* into a folder where you have write access:
```
pip install --install-option='--prefix=~/extra_package/' git+https://github.com/oliverphilcox/ChempyMulti.git
```
Then you have to add the `site-packages/` folder which will be one of the newly created subfolders in `extra_package/` into the ```PYTHONPATH``` variable, e.g.:
```
export PYTHONPATH=~/extra_package/lib/python3.6/site-packages/:$PYTHONPATH
```
If you want this to be permanent, you can add the last line to your `.bashrc`.


## Authors
- Jan Rybizki (MPIA, rybizki@mpia.de) - *Original Chempy*
- Oliver Philcox (Princeton, ohep2@alumni.cam.ac.uk) - *Yield table scoring + Multi-Star Inference*

## Collaborators
- Hans-Walter Rix (MPIA)
- Andreas Just (ZAH)
- Morgan Fouesneau (MPIA)
- Nathan Sandford (UC Berkeley)

## Links
- Philcox & Rybizki (2019, submitted to *ApJ*, [arXiv](https://arxiv.org/abs/1909.00812)) - Multi-Star Inference
- Philcox, Rybizki & Gutcke (2018, ApJ, [arXiv](https://arxiv.org/abs/1712.05686), [Zenodo](https://zenodo.org/record/1247336)) - Yield Table Scoring 
- Rybizki, Just & Rix (A&A, [arXiv](http://arxiv.org/abs/1702.08729), [ASCL](http://ascl.net/1702.011)) - The *Chempy* model
- An early version of *Chempy* is presented in chapter 4 of Jan's [phd thesis](http://nbn-resolving.de/urn:nbn:de:bsz:16-heidok-199349).

## Getting Started
To run tutorials interactively first clone the repository with
```
git clone https://github.com/oliverphilcox/ChempyScoring.git
```
Then you can ```jupyter notebook``` from within the tutorial folder (it will run if you have installed *ChempyMulti*). 
If you did not install ChempyScoring you can still run the tutorial but need to point to the files in the Chempy folder. Basically you have to ```cd ../ChempyScoring/``` and then replace each ```from Chempy import ...``` with ```from . import ...```.

You can also have a look at the *preliminary* Chempy [documentation](http://www.mpia.de/homes/rybizki/html/index.html) which gives an overview over the Chempy classes and functions, although this does not yet provide support for the Scoring, Neural Network or Multi-Star implementations.

## Attribution
Please cite the initial Chempy [paper](https://arxiv.org/abs/1702.08729), the scoring table and neural network [paper](https://arxiv.org/abs/1712.05686) and the multi-star [paper](https://arxiv.org/abs/1909.00812) when using the code in your research.
 
