import numpy as np
from Chempy.parameter import ModelParameters
a=ModelParameters()
from Chempy.cem_function import extract_parameters_and_priors
from Chempy.wrapper import Chempy

log10beta=1.01
alphaIMF=-2.45
log10N_1a=-2.89
log10SFE=-0.45
log10SFRpeak=-0.45
xout=0.52

params=[log10beta,alphaIMF,log10N_1a,log10SFE,log10SFRpeak,xout]
a2=extract_parameters_and_priors(params,a)
cube,abun=Chempy(a2)

import pickle
pickle.dump(abun,open("C04_9elements_abundances.pkl","wb"))
pickle.dump(cube,open("C04_9elements_cube.pkl","wb"))
