# First load in TNG dataset
import numpy as np
positions=np.load('/data/ohep2/MPIAData/SimData/100MSun/posteriorPDF.npy')

draws=positions[:100,:]

cubes=[]
abuns=[]

def runner(i):
	print('Starting %s' %i)
	from Chempy.parameter import ModelParameters
    	a=ModelParameters()
    	from Chempy.cem_function import extract_parameters_and_priors
    	from Chempy.wrapper import Chempy
    	a2=extract_parameters_and_priors(draws[i],a)
    	_,abun=Chempy(a2)
    	return abun
    	
import multiprocessing as mp
p=mp.Pool()
import tqdm
LEN=len(draws)
abuns=tqdm.tqdm(p.imap_unordered(runner,range(LEN)),total=LEN)

meds=np.median(positions,axis=0)
from Chempy.parameter import ModelParameters
a=ModelParameters()
from Chempy.cem_function import extract_parameters_and_priors
from Chempy.wrapper import Chempy
a2=extract_parameters_and_priors(meds,a)
cubeM,abunM=Chempy(a2)

np.savez('TNG100FreeFreeData.npz',median=abunM,abundances=abuns)
