def Hogg_scoring(index):
	"""This function will compute the cross-validation abundances for each of the 22 elements,
	using the best parameter choice for each. Abundances are saved in Hogg/abundnace[INDEX].npy,
	with order used in elements.npy"""
	import numpy as np
	from Chempy.parameter import ModelParameters
	import importlib
	import fileinput
	import sys   
	import os
	from Chempy.wrapper import multi_star_optimization
	from Chempy.plot_mcmc import restructure_chain
	from Chempy.cem_function import posterior_function_returning_predictions

	directory = "Scores/Hogg_"+str(index)+"/"
	if not os.path.exists(directory):
		os.makedirs(directory)
    
	## Code to rewrite parameter file for each element in turn, so as to run MCMC for 21/22 elements only
	# This is definitely not a good implementation (involves rewriting entire parameter file),
	# But other steps are far slower

	# Starting elements (copied from original parameter file)
	elements_to_trace = ['Al', 'Ar', 'B', 'Be', 'C', 'Ca', 'Cl', 'Co', 'Cr', 'Cu', 'F', 'Fe', 'Ga', 'Ge', 'H', 'He', 'K', 'Li', 'Mg', 'Mn', 'N', 'Na', 'Ne', 'Ni', 'O', 'P', 'S', 'Sc', 'Si', 'Ti', 'V', 'Zn']
	orig = "\telements_to_trace = "+str(elements_to_trace) # Original element string

	# Calculate required Chempy elements
	sol_dat = np.load("Chempy/input/stars/Proto-sun.npy")
	el_names = []
	for el in elements_to_trace:
		if el in sol_dat.dtype.names:
			el_names.append(el)
	np.save(directory+"elements.npy",np.array(el_names))
     
	# Create new parameter names
	newstr = []
	for i,el in enumerate(el_names):
		if el !='Zn':
			newstr.append(orig.replace("'"+str(el)+"', ",""))
		else:
			newstr.append(orig.replace("'"+str(el)+"', ",""))

	for i in range(len(el_names)): # Iterate over removed element
		for line in fileinput.input("Chempy/parameter.py", inplace=True):
			if "\telements_to_trace" in line:
				print(newstr[i])
			else:
				print(line,end='')
		del sys.modules['Chempy.parameter']
		from Chempy.parameter import ModelParameters
		a = ModelParameters()
		##############
		# MCMC using 21 elements only goes here
		print('Running MCMC iteration %d of %d' %(i+1,len(el_names)))
		multi_star_optimization()
		restructure_chain('mcmc/')
		positions = np.load('mcmc/posteriorPDF.npy')
		median = []
		up = []
		low = []
		for j in range(len(a.p0)):
			median.append(np.percentile(positions[:,j],50))
			low.append(np.percentile(positions[:,j],15.865))
			up.append(np.percentile(positions[:,j],100-15.865))
		np.save(directory+'median%d.npy' %(i),np.array(median))
		np.save(directory+'low%d.npy' %(i),np.array(low))
		np.save(directory+'up%d.npy' %(i),np.array(up))
		##############
		for line in fileinput.input("Chempy/parameter.py", inplace=True):
			if "\telements_to_trace" in line:
				print(orig)
			else:
				print(line,end='')
		del sys.modules['Chempy.parameter']
		from Chempy.parameter import ModelParameters
		a = ModelParameters()
		##############
		# Code needing all 22 elements goes here
		param = median
		abundances,names = posterior_function_returning_predictions((param,a))
		for n,name in enumerate(names):
			if name == el_names[i]:
				required_abundance = abundances[n]
		np.save(directory+'abundance%d.npy' %(i),required_abundance)
	return None

	 
def Hogg_wrapper():
	import fileinput
	import sys
	from Chempy.parameter import ModelParameters
	from Chempy.score_function import Hogg_scoring
	import time
	init_time = time.time()	
	for index in [1., 1.2, 1.4, 1.7, 2., 2.5, 3., 4., 5., 6., 8., 10., 12., 15., 18., 22., 26., 30., 40., 50.]:
		for line in fileinput.input("Chempy/parameter.py", inplace=True):
			if "\tbeta_param =" in line:
				print("\tbeta_param = %.2f" %(index))
			else:
				print(line,end='')
		del sys.modules['Chempy.parameter']
		Hogg_scoring(index)  
		
	return None
    
    
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
	
			
