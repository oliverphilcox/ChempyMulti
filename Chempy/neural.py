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

def create_network():
	""" Function to create and train the neural network - this overwrites any previous network.
	
	Outputs:
		epochs - Training epoch number (outputted each 10 epochs)
		losslog - loss value for each 10 epochs
		
		Neural/neural_model.npz - Saved .npz file with model weights
	"""
	
	import torch
	from torch.autograd import Variable

	# Load parameters
	a = ModelParameters()
	n_train = a.training_size**len(a.p0) # No. data points in training set
	
	# Load pre-processed training data
	tr_input = np.load('Neural/training_norm_grid.npy')
	tr_output = np.load('Neural/training_abundances.npy')	
	
	# Calculate the model dimensions
	dim_in = tr_input.shape[1]
	dim_out = tr_output.shape[1]

	# Convert to torch variables
	tr_input = Variable(torch.from_numpy(tr_input)).type(torch.FloatTensor)
	tr_output = Variable(torch.from_numpy(tr_output), requires_grad=False).type(torch.FloatTensor)
	
	# Set up the neural network, with one hidden layer
	model = [] # Remove any previous network
	
	model = torch.nn.Sequential(
				torch.nn.Linear(dim_in,a.neurons),
				torch.nn.ReLU(),
				torch.nn.Linear(a.neurons,dim_out)
				)
	loss_fn = torch.nn.L1Loss(size_average=True)
	
	# Use Adam optimizer with learning rate specified in parameter.py
	optimizer = torch.optim.Adam(model.parameters(), lr = a.learning_rate)
	
	# For loss records
	losslog = []
	epoch = []
	
	# Train neural network
	for i in range(a.epochs):
		pred_output = model(tr_input)
		loss = loss_fn(pred_output, tr_output)
		optimizer.zero_grad() # Initially zero gradient
		loss.backward() # Backpropagation
		optimizer.step() # Update via Adam 
		
		# Output loss
		if i % 10 ==0:
			losslog.append(loss.data[0])
			epoch.append(i)
		if i % 100==0:
			print("Training epoch %d of %d complete" %(i,a.epochs))
		
	# Convert weights to numpy arrays	
	model_numpy = []
	for param in model.parameters():
		model_numpy.append(param.data.numpy())
		
	np.savez("Neural/neural_model.npz",
				w_array_0=model_numpy[0],
				b_array_0=model_numpy[1],
				w_array_1=model_numpy[2],
				b_array_1=model_numpy[3])
				
	return epoch, losslog


def neural_output(test_input):
	""" This calculates the neural network predicted output for a trained network.
	
	Inputs:
		test_input - Array containing unnormalized parameter values
		(coeffs - loaded automatically from Neural/neural_model.npz)
	
	Output:
		Neural network abundance prediction
	"""
	
	a = ModelParameters()	
	
	# Load in model coefficients
	coeffs = np.load('Neural/neural_model.npz')
	w_array_0 = coeffs['w_array_0']
	w_array_1 = coeffs['w_array_1']
	b_array_0 = coeffs['b_array_0']
	b_array_1 = coeffs['b_array_1']
		
	# Normalize data for input into network
	norm_data = (test_input - a.p0)/np.array(a.training_widths)
	
	# Calculate neural network output
	hidden = np.maximum(0,np.dot(w_array_0,norm_data)+b_array_0)
	output = np.dot(w_array_1, hidden)+b_array_1
	
	return output
	

	
	