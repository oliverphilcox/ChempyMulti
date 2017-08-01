def training_data():
	""" Function to create neural network training dataset for Chempy data
	
	Output is stored in the Neural/training_abundances.npy file for later use"""

	from Chempy.parameter import ModelParameters
	from Chempy.cem_function import posterior_function_returning_predictions
	import numpy as np
	from scipy.stats import norm
	import os
	
	# FOR TESTING	
	import warnings
	warnings.filterwarnings("ignore")

	a = ModelParameters()
	
	## This calculates a list of 5 trial values for each parameter around the prior value, as an array of 6 lists which will be combined
	# Set the desired Gaussian sigma values in the widths parameter (values > prior sigma are used to fully explore parameter space)
	# Parameter values are chosen that are evenly distributed in the Gaussian probability space (e.g. 16.7, 33, 50 etc. percentile points)
	grid1d = np.zeros((a.training_size,len(a.p0)))
	widths = a.neural_widths
	for i in range(a.training_size):
		prob = (i+1)/(a.training_size+1)
		grid1d[i]=  norm.ppf(prob,loc = a.p0, scale = widths)
	grid1d = grid1d.T 
	
	## Now combine lists to make list of all possible combinations (a.training_size^len(a.p0) = 5^6 ~ 10,000)
	grid = np.array(np.meshgrid(grid1d[0],grid1d[1],grid1d[2],grid1d[3],grid1d[4],grid1d[5])).T.reshape(-1,6)
	
	directory = 'Neural/'	
	if not os.path.exists(directory):
		os.makedirs(directory)
	np.save(directory+'training_grid.npy',grid) # Save for future use
	
	# USING Karakas 10 yields for now
	#grid = grid[:6] # USE THIS FOR TESTING
	training_abundances = np.zeros((len(grid),22)) # 22 is number of traceable elements - automate this??
	for i,item in enumerate(grid):
		abundances,_ = posterior_function_returning_predictions((item,a)) # Send new parameters from grid
		training_abundances[i] = abundances
		if i%100 == 0:
			print(i) # For testing
        
	np.save('Neural/training_abundances.npy', training_abundances)
	return 0