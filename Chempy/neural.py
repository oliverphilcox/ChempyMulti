def training_data():
	""" Function to create neural network training dataset for Chempy data
	
	Output is stored in the Neural/training_abundances.npy file for later use"""

	from Chempy.parameter import ModelParameters
	from Chempy.cem_function import posterior_function_returning_predictions
	import numpy as np
	from scipy.stats import norm
	
	# FOR TESTING	
	import warnings
	warnings.filterwarnings("ignore")

	a = ModelParameters()
	
	## This calculates a list of 7 trial values for each parameter around the prior value, as an array of 6 lists which will be combined
	# Set the desired Gaussian sigma values in the widths parameter (values > prior sigma are used to fully explore parameter space)
	# Parameter values are chosen that are evenly distributed in the Gaussian probability space (e.g. 12.5, 25, 37.5 etc. percentile points)
	test_size = 7
	grid1d = np.zeros((test_size,len(a.p0)))
	widths = np.array([0.6,0.3,0.3,0.3,0.2,0.2])
	for i in range(test_size):
		prob = (i+1)/(test_size+1)
		grid1d[i]=  norm.ppf(prob,loc = a.p0, scale = widths)
	grid1d = grid1d.T 
	
	## Now combine lists to make list of all possible combinations (test_size^len(a.p0) = 7^6 ~ 10,000)
	grid = np.array(np.meshgrid(grid1d[0],grid1d[1],grid1d[2],grid1d[3],grid1d[4],grid1d[5])).T.reshape(-1,6)
	np.save('Neural/training_grid.npy',grid) # Save for future use
	
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