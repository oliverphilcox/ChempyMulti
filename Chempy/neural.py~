from Chempy.parameter import ModelParameters
from Chempy.cem_function import posterior_function_returning_predictions
import numpy as np
import os

def training_data():
	""" Function to create neural network training dataset for Chempy data. 

	We calculate a list of 5 trial values for each parameter about the prior, and create an array of all combinations.
	Trial values are chosen to be uniform in the Gaussian probability space (16.7,33.3, 50 percentile etc.)
	Gaussian widths (stored in parameter.py) are chosen to fully explore required parameter space
	Trial values normalized to unit Gaussians are created for neural network input	
	
	Outputs (stored in Neural/ as npy files):
		training_norm_grid - Input training dataset saved as normalized unit Gaussians
		training_param_grid - Input training dataset in parameter space
		training_abundances - Output training abundance data
		
	"""

	from scipy.stats import norm as gaussian
		
	# FOR TESTING	
	import warnings
	warnings.filterwarnings("ignore")

	a = ModelParameters()
	
	N = a.training_size # No. data points per parameter
	widths = a.training_widths # Gaussian widths for parameters	
	
	# Create 1d grid of data points equally spaced in probability space 
	prob = np.linspace(1/(N+1), 1-1/(N+1), N)
	grids = [gaussian.ppf(prob) for _ in range(len(a.p0))] # Normalize to unit Gaussian
	norm_grid = np.array(np.meshgrid(*grids)).T.reshape(-1,N+1)
	
	# Create grid in parameter space
	param_grid = [item*widths+a.p0 for item in norm_grid]

	# Save grids
	directory = 'Neural/'
	if not os.path.exists(directory):
		os.makedirs(directory)
		
	np.save(directory+'training_norm_grid.npy',norm_grid)
	np.save(directory+'training_param_grid.npy',param_grid)
	
	## Create abundance output
	training_abundances = []
	for i,item in enumerate(param_grid):
		abundances,_ = posterior_function_returning_predictions((item,a))
		training_abundances.append(abundances)
		if i%100 == 0:
			print("Calculating abundance set %d of %d" %(i,len(param_grid)))
              
	# Save abundance table
	np.save(directory+'training_abundances.npy', training_abundances)

	return 0
	
def verification_and_testing():
	""" This will create the verification and testing data-sets for use with the neural network.
	The data-sets are created randomly from the Gaussian prior distribution
	
	Outputs (saved as .npy files in the Neural/ folder):
		verif_param_grid - Verification parameter data
		verif_abundances - Verification dataset abundances
		test_param_grid - Test parameter data
		test_abundances - Test dataset abundances
		
	"""

	# FOR TESTING	
	import warnings
	warnings.filterwarnings("ignore")
		
	a = ModelParameters()
	names = ['verif','test'] # Two datasets
	
	for i,name in enumerate(names): # Create two identically distributed datasets
		length = a.verif_test_sizes[i]
		param_grid = []
		
		# Distribute data with prior widths
		for _ in range(length):
			param_grid.append(np.random.normal(size = len(a.p0), loc = a.p0,
									scale = a.test_widths))
			np.save("Neural/"+name+"_param_grid.npy",param_grid)
    
		model_abundances = []
		for j,jtem in enumerate(param_grid):
			abundances,_ = posterior_function_returning_predictions((jtem,a))
			model_abundances.append(abundances)
			if j%100 == 0:
				print("Calculating %s abundance set %d of %d" %(name,j,length))
 		
 		# Save abundance table
		np.save("Neural/"+name+"_abundances.npy",model_abundances)
		
	return 0


