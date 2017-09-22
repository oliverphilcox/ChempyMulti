# ChempyScoring - Evolution Modelling + Yield Table Scoring
Flexible one-zone open-box chemical evolution modeling. Abundance fitting and stellar feedback calculation. Parameter optimization for simulations and yield table scoring. This is the code described in Philcox & Rybizki (2017, submitted)

## Installation

```
pip install git+https://github.com/oliverphilcox/ChempyScoring.git
```
Chempy should run with the latest python 2 and python 3 version.
Its dependencies are: [Numpy](http://numpy.scipy.org/), [SciPy](http://www.scipy.org/), [matplotlib](http://matplotlib.sourceforge.net/), [multiprocessing](https://docs.python.org/2/library/multiprocessing.html#module-multiprocessing), [emcee](http://dan.iel.fm/emcee/current/) (for the MCMC), [corner](http://corner.readthedocs.io/en/latest/) (for the MCMC plots), [PyTorch](http://pytorch.org/) (for Neural Networks), [scikit-monaco](https://pypi.python.org/pypi/scikit-monaco) (for Monte Carlo integration), and [tqdm](https://pypi.python.org/pypi/tqdm) (for timing multiprocessing). They are all pip installable and you can also get part of it with [Anaconda](https://www.continuum.io/downloads).

### Installation without admin rights:
You can install *ChempyScoring* into a folder where you have write access:
```
pip install --install-option='--prefix=~/extra_package/' git+https://github.com/oliverphilcox/ChempyScoring.git
```
Then you have to add the `site-packages/` folder which will be one of the newly created subfolders in `extra_package/` into the ```PYTHONPATH``` variable, e.g.:
```
export PYTHONPATH=~/extra_package/lib/python3.6/site-packages/:$PYTHONPATH
```
If you want this to be permanent, you can add the last line to your `.bashrc`.


## Authors
- Jan Rybizki (MPIA, rybizki@mpia.de) - *Original Chempy*
- Oliver Philcox (IoA, ohep2@cam.ac.uk) - *Yield table scoring*

## Collaborators
- Hans-Walter Rix (MPIA)
- Andreas Just (ZAH)
- Morgan Fouesneau (MPIA)

## Links
- Philcox & Rybizki 2017 (will be on arXiV shortly)
- <a href="http://arxiv.org/abs/1702.08729"><img src="http://img.shields.io/badge/arXiv-1702.08729-orange.svg?style=flat" alt="arxiv:1702.08729" /></a>
- <a href="http://ascl.net/1702.011"><img src="https://img.shields.io/badge/ascl-1702.011-blue.svg?colorB=262255" alt="ascl:1702.011" /></a>
- An early version of Chempy is presented in chapter 4 of Jan's [phd thesis](http://nbn-resolving.de/urn:nbn:de:bsz:16-heidok-199349).

## Getting started
The jupyter [tutorial](https://github.com/oliverphilcox/ChempyScoring/blob/master/Scoring%20%26%20Parameter%20Choice%20Tutorial.ipynb) shows the usage of the software for computing yield table scores and best posterior parameters. The original Chempy [tutorial](https://github.com/oliverphilcox/ChempyScoring/old_tutorials) illustrate the basic usage of Chempy and basic concepts of galactic chemical evolution modeling. Both can be inspected in the github repository or you can run them interactively on your local machine. 

To run them interactively first clone the repository with
```
git clone https://github.com/oliverphilcox/ChempyScoring.git
```
Then you can ```jupyter notebook``` from within the tutorial folder (it will run if you have installed *ChempyScoring*). 
If you did not install ChempyScoring you can still run the tutorial but need to point to the files in the Chempy folder. Basically you have to ```cd ../ChempyScoring/``` and then replace each ```from Chempy import ...``` with ```from . import ...```.

You can also have a look at the *preliminary* Chempy [documentaion](http://www.mpia.de/homes/rybizki/html/index.html) which gives an overview over the Chempy classes and functions, although this does not yet provide support for the Scoring or Neural Network implementations.

## Attribution
Please cite the initial Chempy [paper](https://arxiv.org/abs/1702.08729) and scoring table updates (Philcox & Rybizki 2017, submitted) when using the code in your research (so far only arxiv link, will be updated).
