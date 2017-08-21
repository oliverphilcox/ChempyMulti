import numpy as np
import os
	
def Hogg_score():
	"""This function will compute the cross-validation abundances for each of the 22 elements,
	using the best parameter choice for each. 
	This computes the likelihood contribution, and saves each separately.
	"""
	from Chempy.parameter import ModelParameters
	import importlib
	import fileinput
	import sys   
	import os
	import multiprocessing as mp
	import tqdm
	from Chempy.wrapper import single_star_optimization
	from Chempy.plot_mcmc import restructure_chain
	from Chempy.cem_function import posterior_function_mcmc_quick
	from scipy.stats import norm
	from .score_function import preload_params_mcmc
	import matplotlib.pyplot as plt
	p = mp.Pool()
	 
	## Code to rewrite parameter file for each element in turn, so as to run MCMC for 21/22 elements only
	# This is definitely not a good implementation (involves rewriting entire parameter file),
	# But other steps are far slower
	
	# Initialise arrays
	element_mean = []
	element_sigma = []
	overall_score = 1.
	factors = []
	
	# Starting elements (copied from original parameter file)
	elements_to_trace = ['Al', 'Ar', 'B', 'Be', 'C', 'Ca', 'Cl', 'Co', 'Cr', 'Cu', 'F', 'Fe', 'Ga', 'Ge', 'H', 'He', 'K', 'Li', 'Mg', 'Mn', 'N', 'Na', 'Ne', 'Ni', 'O', 'P', 'S', 'Sc', 'Si', 'Ti', 'V', 'Zn']
	orig = "\telements_to_trace = "+str(elements_to_trace) # Original element string

	# Calculate required Chempy elements
	preload = preload_params_mcmc()
	elements_init = np.copy(preload.elements)
	np.save('Scores/Hogg_elements.npy',elements_init)
     
	# Create new parameter names
	newstr = []
	for i,el in enumerate(elements_init):
		if el !='Zn':
			newstr.append(orig.replace("'"+str(el)+"', ",""))
		else:
			newstr.append(orig.replace("'"+str(el)+"', ",""))
	for index in range(len(elements_init)): # Iterate over removed element
		for line in fileinput.input("Chempy/parameter.py", inplace=True):
			if "\telements_to_trace" in line:
				#print(newstr[index])
				print(line,end='') # TO TEST
			else:
				print(line,end='')
		fileinput.close()
		del sys.modules['Chempy.parameter']
		from Chempy.parameter import ModelParameters
		a = ModelParameters()
		del sys.modules['Chempy.score_function']
		from .score_function import preload_params_mcmc 
		preload = preload_params_mcmc()
		##############
		
		# Run MCMC with 27/28 elements. 
		print('Running MCMC iteration %d of %d' %(index+1,len(elements_init)))
		#print(a.elements_to_trace)
		single_star_optimization()
		
		# Create the posterior PDF and load it 
		restructure_chain('mcmc/')
		positions = np.load('mcmc/posteriorPDF.npy') # Posterior parameter PDF
		
		##############
		
		for line in fileinput.input("Chempy/parameter.py", inplace=True):
			if "\telements_to_trace" in line:
				print(orig)
			else:
				print(line,end='')
		fileinput.close()
		del sys.modules['Chempy.parameter']
		from Chempy.parameter import ModelParameters
		del sys.modules['Chempy.score_function']
		from .score_function import preload_params_mcmc 
		a = ModelParameters()
		preload = preload_params_mcmc()
		
		##############
		
		# This uses all 28 elements again for predictions
				
		# Multiprocess and calculate elemental predictions for each parameter set

		from .score_function import element_predictor
		indices = np.ones(len(positions))*index
		abundance = list(tqdm.tqdm(p.imap_unordered(element_predictor,zip(positions,indices)),total=len(positions)))
		#p.close()
		#p.join()		
			
		
		abundance = np.array(abundance)
		mean,sigma = norm.fit(abundance)
		print(mean)
		print(sigma)
		
		element_mean.append(mean)
		element_sigma.append(sigma)
		#a.plot_hist=True
		if a.plot_hist == True:
			plt.clf()
			plt.hist(abundance, bins=40, normed=True, alpha=0.6, color='g')
			#abundance = np.array(abundance) # Unmask array
			# Plot the PDF.
			xmin, xmax = plt.xlim()
			x = np.linspace(xmin, xmax, 100)
			p = norm.pdf(x, mean, sigma)
			plt.plot(x, p, c='k', linewidth=2)
			title = 'Plot of element %d abundance' %(index)
			plt.title(title)
			plt.xlabel('[X/Fe] abundance')
			plt.ylabel('Relative frequency')
		
		total_err = np.sqrt((preload.star_error_list[index])**2 + sigma**2)
		likelihood_factor = norm.pdf(mean,loc=preload.star_abundance_list[index],scale=total_err)
		overall_score *= likelihood_factor
		factors.append(likelihood_factor)
		print("Likelihood contribution from %dth element is %.8f" %(index+1,likelihood_factor))
		print(overall_score)
	
	np.savez('Scores/Hogg_beta_elements'+str(a.beta_param)+'.npz',
				elements=elements_init,
				likelihood_factors=factors,
				element_mean = element_mean,
				element_sigma = element_sigma)	
	return overall_score

def element_predictor(args):
	params,index = args	
	index = int(index)
	preload = preload_params_mcmc()
	from .cem_function import posterior_function_mcmc_quick
	_,all_abun = posterior_function_mcmc_quick(params,preload.elements,preload)
	output = all_abun[index]
	return output


    
class preload_params_mcmc():
	"""This calculates and stores useful quantities that would be calculated multiple times otherwise for the mcmc run.
	Definitions can be called from this file
	"""
	import numpy as np
	from Chempy.parameter import ModelParameters
	from scipy.stats import beta
	a = ModelParameters()

	elements_to_trace = list(a.elements_to_trace)
	elements_to_trace.append('Zcorona')
	elements_to_trace.append('SNratio')
	
	# Neural network coeffs
	coeffs = np.load('Neural/neural_model.npz')
	w_array_0 = coeffs['w_array_0']
	w_array_1 = coeffs['w_array_1']
	b_array_0 = coeffs['b_array_0']
	b_array_1 = coeffs['b_array_1']
	coeffs.close()
	
	# Beta function calculations
	model_errors = np.linspace(a.flat_model_error_prior[0],a.flat_model_error_prior[1],a.flat_model_error_prior[2])
	error_weight = beta.pdf(model_errors, a = a.beta_error_distribution[1], b = a.beta_error_distribution[2])
	error_weight/= sum(error_weight)

	wildcard = np.load('Chempy/input/stars/'+a.stellar_identifier+'.npy')	
	
	elements = []
	star_abundance_list = []
	star_error_list = []
	for i,item in enumerate(a.elements_to_trace):
		if item in list(wildcard.dtype.names):
			elements.append(item)
			star_abundance_list.append(float(wildcard[item][0]))
			star_error_list.append(float(wildcard[item][1]))
	star_abundance_list = np.hstack(star_abundance_list)
	star_error_list = np.hstack(star_error_list)
	elements = np.hstack(elements)
	
	err=[]
	for i,item in enumerate(model_errors):
		error_temp = np.ones(len(elements))*item
		err.append(np.sqrt(np.multiply(error_temp[:,None],error_temp[:,None]).T + np.multiply(star_error_list,star_error_list)).T)
	
def Bayes_score():
	"""
	This calculates the Bayes factor score for a specific yield set and choice of error parameter, as defined in parameter file.
	First MCMC is run to determine the centre of the parameter space and then integration is performed.
	This needs a trained neural network in the Neural/ folder.
	
	Output is Bayes score and predicted (1 sigma) error.
	"""
	from .parameter import ModelParameters
	from .cem_function import posterior_function_mcmc_quick
	from .score_function import preload_params_mcmc
	from .plot_mcmc import restructure_chain
	from .wrapper import single_star_optimization
	from scipy.stats import multivariate_normal as scinorm
	from numpy.random import multivariate_normal as numnorm
	from skmonaco import mcimport
	import time
	
	# Load model parameters
	a = ModelParameters()
	preload = preload_params_mcmc()
	init_time = time.time()
	
	# Compute posterior + load median values - this automatically uses the neural network!!
	print('After %.3f seconds, finding posterior parameter values' %(time.time()-init_time))
	single_star_optimization()
	restructure_chain('mcmc/')
	positions = np.load('mcmc/posteriorPDF.npy')
	init_param = []
	for j in range(len(a.p0)):
		init_param.append(np.percentile(positions[:,j],50))
	print('After %.3f seconds, initial parameters are:' %(time.time()-init_time),init_param)
	
	# Function to compute posterior (needs a trained neural network)
	def posterior(theta):
		a = ModelParameters()
		post,_ = posterior_function_mcmc_quick(theta,a,preload)
		posterior = np.exp(post)
		return posterior
	
	# Read prior sigma from file	
	sigma = [] # Read prior sigma from parameter file
	for i,param_name in enumerate(a.to_optimize):
		sigma.append(a.priors.get(param_name)[1])
	sigma = np.array(sigma)
	
	# Compute covariance matrix
	print('After %.3f seconds, computing covariance matrix' %(time.time()-init_time))
	positions = np.load('mcmc/posteriorPDF.npy')
	cov_matrix = np.zeros((len(a.p0),len(a.p0)))
	for i in range(len(a.p0)):
		for j in range(len(a.p0)):
			cov_matrix[i,j] = np.cov((positions[:,i],positions[:,j]))[1,0]

	def gauss_factor(theta):
		# Returns gaussian fit to data
		return scinorm.pdf(theta,mean=np.array(init_param),cov=cov_matrix)

	def posterior_mod(theta):
		# Returns flattened posterior
		return posterior(theta)/gauss_factor(theta)

	def dist(size):
	# Distribution function for mcmc sampling
		mean = np.array(init_param)
		return numnorm(mean,cov_matrix,size=size)
		
	print('After %.3f seconds, starting parameter-space integration for beta = %.3f' %(time.time()-init_time, a.beta_param))
	integral,integral_err = mcimport(posterior_mod,a.int_samples,dist)

	print('After %.3f seconds, integration is complete' %(time.time()-init_time))
	np.save('Scores/integral_'+str(a.beta_param)+'.npy',integral)
	np.save('Scores/integral_err_'+str(a.beta_param)+'.npy',integral_err)
	
	return integral,integral_err	
	
def Bayes_wrapper():
	"""
	This function calculates the Bayes score as a function of the beta-function parameter (defined in parameter.py)
	It is not currently parallelized (parallelization is done in MCMC already, but not in the integration).
	
	Output scores are saved and labelled in the Scores/ folder as an .npz file
	"""
	import time
	import fileinput
	import sys
	from .parameter import ModelParameters
	directory = 'Scores/'
	if not os.path.exists(directory):
		os.makedirs(directory)

	a = ModelParameters()
	beta_params = a.list_of_beta_params
	score = []
	score_err = []
	init_time = time.time()
	
	for i in range(len(beta_params)): # Iterate over beta parameters
		print("Calculating value %d of %d after %.3f seconds" %(i+1,len(beta_params),time.time()-init_time))
		# Modify beta value
		for line in fileinput.input("Chempy/parameter.py", inplace=True):
			if "\tbeta_param" in line:
				print("\tbeta_param = %.5f" %beta_params[i])
			else:
				print(line,end='')
		fileinput.close()		
		# Reimport model parameters for new beta 
		del sys.modules['Chempy.parameter']
		del sys.modules['Chempy.score_function']
		from .parameter import ModelParameters
		from .score_function import preload_params_mcmc 
		preload = preload_params_mcmc() # Must be reloaded since this depends on beta parameter
		a = ModelParameters()
		
		# Calculate Bayes score
		integral,integral_err = Bayes_score()
		score.append(integral)
		score_err.append(integral_err)
	
	# Save output as npz array
	np.savez("Scores/Bayes_score - "+str(a.yield_table_name_sn2)+", "+str(a.yield_table_name_agb)+", "+str(a.yield_table_name_1a)+".npz",
				beta_param=beta_params,
				score=score,
				score_err=score_err)
				
	return beta_params,score,score_err
	
def Hogg_wrapper():
	"""
	This function calculates the Hogg score as a function of the beta-function parameter (defined in parameter.py)
	It is not currently parallelized (parallelization is done in MCMC already and for element predictions).
	"""
	import time
	import fileinput
	import sys
	from .parameter import ModelParameters
	from .score_function import Hogg_score
	
	a = ModelParameters()
	beta_params = a.list_of_beta_params
	Hogg_score_list = []
	init_time = time.time()
	
	for i in range(len(beta_params)): # Iterate over beta parameters
		print("Calculating value %d of %d after %.3f seconds" %(i+1,len(beta_params),time.time()-init_time))
		# Modify beta value
		for line in fileinput.input("Chempy/parameter.py", inplace=True):
			if "\tbeta_param" in line:
				print("\tbeta_param = %.5f" %beta_params[i])
			else:
				print(line,end='')
		fileinput.close()		
		# Reimport model parameters for new beta 
		del sys.modules['Chempy.parameter']
		del sys.modules['Chempy.score_function']
		from Chempy.score_function import preload_params_mcmc
		from Chempy.parameter import ModelParameters
		a = ModelParameters()
		
		# Calculate Bayes score
		score = Hogg_score()
		Hogg_score_list.append(score)
	
	# Save output as npz array
	np.savez("Scores/Hogg_score - "+str(a.yield_table_name_sn2)+", "+str(a.yield_table_name_agb)+", "+str(a.yield_table_name_1a)+".npz",
				beta_param=beta_params,
				score=Hogg_score_list)
				
	return beta_params,Hogg_score_list
	
	