from .parameter import ModelParameters
from .cem_function import posterior_function_returning_predictions
import numpy as np
import os
import matplotlib.pyplot as plt
a = ModelParameters()

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
	import multiprocessing as mp

	# FOR TESTING
	import warnings
	warnings.filterwarnings("ignore")

	a = ModelParameters()

	N = a.training_size # No. data points per parameter
	widths = a.training_widths # Gaussian widths for parameters

	# Create 1d grid of data points equally spaced in probability space
	prob = np.linspace(1/(N+1), 1-1/(N+1), N)
	grids = [gaussian.ppf(prob) for _ in range(len(a.p0))] # Normalize to unit Gaussian
	norm_grid = np.array(np.meshgrid(*grids)).T.reshape(-1,len(a.p0))

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

	return None

def verification_and_testing():
	""" This will create the verification and testing data-sets for use with the neural network.
	The data-sets are created randomly from the Gaussian prior distribution, within the bounds set in the parameter file

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

	lower = np.zeros(len(a.p0))
	upper = np.zeros(len(a.p0))

	# Set upper/lower bounds in parameter space
	for i,param_name in enumerate(a.to_optimize):
		lower[i], upper[i] = a.constraints.get(param_name)

	for j ,name in enumerate(names): # Create both test sets
		param_grid = []
		model_abundances = []
		for k in range(a.verif_test_sizes[j]):
			param = np.ones(len(a.p0))*np.inf # To ensure initial value is not in range
			for i in range(len(a.p0)):
				param[i] = np.inf
				while param[i] > upper[i] or param[i] < lower[i]: # Continue until param is in correct range
					param[i] = np.random.normal(loc=a.p0[i],scale=a.test_widths[i])
			param_grid.append(param)
			abundances,_ = posterior_function_returning_predictions((param,a))
			model_abundances.append(abundances)
			if k%100==0 :
				print("Calculating %s abundance set %d of %d" %(name,k,a.verif_test_sizes[j]))
		np.save("Neural/"+name+"_param_grid.npy",param_grid)
		np.save("Neural/"+name+"_abundances.npy",model_abundances)

	return None

def create_network(learning_rate=a.learning_rate,Plot=True):
	""" Function to create and train the neural network - this overwrites any previous network.
	Inputs:
		learning_rate of model (default is in parameter.py)
		Plot - whether to plot loss curve against epoch
	Outputs:
		epochs - Training epoch number (outputted each 10 epochs)
		losslog - loss value for each 10 epochs
		Plot of loss against epoch (if Plot=True)

		Neural/neural_model.npz - Saved .npz file with model weights
	"""

	import torch
	from torch.autograd import Variable
	#from torch.optim import lr_scheduler

	# Load parameters
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
				torch.nn.Tanh(),
				torch.nn.Linear(a.neurons,dim_out)
				)
	loss_fn = torch.nn.L1Loss(size_average=True)

	# Use Adam optimizer with learning rate specified in parameter.py
	optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
	#scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

	# For loss records
	losslog = []
	epoch = []

	# Train neural network
	for i in range(a.epochs):
		pred_output = model(tr_input)
		loss = loss_fn(pred_output, tr_output)
		optimizer.zero_grad() # Initially zero gradient
		loss.backward() # Backpropagation
		optimizer.step() # Update via optimizer
		#scheduler.step(loss)

		# Output loss
		if i % 3 ==0:
			losslog.append(loss.data[0])
			epoch.append(i)
		if i % 1000==0:
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

	if Plot==True:
		plt.plot(epoch,losslog,label=learning_rate)
		plt.ylabel("L1 Loss value")
		plt.xlabel("Epoch")
		plt.title("Loss plot")
		plt.legend()
		plt.show()
		plt.savefig('Neural/lossplot')

	return epoch, losslog


def neural_output(test_input):
	""" This calculates the neural network predicted output for a trained network.

	Inputs:
		test_input - Array containing unnormalized parameter values
		(coeffs - loaded automatically from Neural/neural_model.npz)

	Output:
		Neural network abundance prediction
	"""
	from Chempy.cem_function import posterior_function_returning_predictions

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
	hidden1 = np.tanh(np.array(np.dot(w_array_0,norm_data)+b_array_0))
	output = np.dot(w_array_1, hidden1)+b_array_1

	return output

def neural_errors(dataset):
	""" Calculate median absolute error between Chempy and neural network for each set of parameters.
	Input is the name of the dataset: 'verif' or 'test'
	"""

	# Load abundances
	model_abundances = np.load('Neural/'+dataset+'_abundances.npy')
	params = np.load('Neural/'+dataset+'_param_grid.npy')

	# Calculate absolute model error
	error=[]
	for i in range(len(params)):
		predicted_abundances = neural_output(params[i])
		error.append(np.absolute(predicted_abundances-model_abundances[i]))

	return np.median(error,axis=1)

def calculate_errors(dataset):
	""" Calculate summed absolute error between Chempy and neural network across all elements for each set of parameters.
	Input is the name of the dataset: 'verif' or 'test'
	Output is mean and standard deviation of errors across all elements and maximum element error
	"""

	# Load abundances
	model_abundances = np.load('Neural/'+dataset+'_abundances.npy')
	params = np.load('Neural/'+dataset+'_param_grid.npy')

	# Calculate absolute model error
	error=[]
	for i in range(len(params)):
		predicted_abundances = neural_output(params[i])
		error.append(np.absolute(predicted_abundances-model_abundances[i]))

	return np.mean(error,axis=1),np.std(error,axis=1),np.amax(error,axis=1)

def neural_corner_plot(dataset):
	""" This function plots a corner plot of the model parameters, with colors showing the median neural network error.
	The error corresponding to max color can be changed by the color_max parameter.

	Input:
		Name of dataset ('verif' or 'test')

	Output:
		corner_parameter_plot.png saved in Neural/ directory
	"""

	from matplotlib import cm

	a=ModelParameters()

	# Load datasets
	data_tr = np.load('Neural/training_param_grid.npy')
	data_v = np.load('Neural/'+dataset+'_param_grid.npy')

	param_error = neural_errors(dataset)

	# Initialize plot
	plt.clf()
	text_size = 12
	plt.rc('font', family='serif',size = text_size)
	plt.rc('xtick', labelsize=text_size)
	plt.rc('ytick', labelsize=text_size)
	plt.rc('axes', labelsize=text_size, lw=1.0)
	plt.rc('lines', linewidth = 1)
	plt.rcParams['ytick.major.pad']='8'
	plt.rcParams['text.latex.preamble']=[r"\usepackage{libertine}"]
	params = {'text.usetex' : True,
          'font.family' : 'libertine',
          'text.latex.unicode': True,
          }
	plt.rcParams.update(params)
	parameter_names = [r'$\alpha_\mathrm{IMF}$',r'$\log_{10}(\mathrm{N_{Ia}})$',
                   r'$\log_{10}(\tau_\mathrm{Ia})$',r'$\log_{10}(\mathrm{SFE})$',
                   r'$\log_{10}(\mathrm{SFR_{peak}})$',r'x_{out}']


	# Plot settings
	fig,axes = plt.subplots(nrows = len(a.p0), ncols = len(a.p0),figsize=(14.69,8.0),dpi=300)
	alpha = 0.5
	lw=2 # Linewidth
	left = 0.1 # Left side of subplots
	right = 0.8 # Right side
	bottom = 0.075
	top = 0.97
	wspace = 0.0 # blankspace width between subplots
	hspace = 0.0 # blankspace height between subplots
	color_max = a.color_max
	plt.subplots_adjust(left=left,bottom=bottom,right=right,top=top,wspace=wspace,hspace=hspace)

	# Create plot
	for i in range(len(a.p0)):
		for j in range(len(a.p0)):
			axes[i,j].locator_params(nbins=4)
			if j==1:
				axes[i,j].locator_params(nbins=4)
			if i==j:
				counts,edges = np.histogram(np.asarray(data_v[:,j]),bins=10)
				max_count = float(np.max(counts))
				counts = np.divide(counts,max_count)
				median = np.zeros(len(edges)-1)
				for k in range(len(edges)-1):
					choice = np.logical_and(np.greater(data_v[:,j],edges[k]),np.less(data_v[:,j],edges[k+1]))
					error=np.extract(choice,param_error)
					if len(error) != 0:
						median[k] = np.median(error)
				colors = cm.coolwarm(median/color_max)
				axes[i,j].bar(left = edges[:-1], height=counts, width = edges[1]-edges[0],
									color=colors,alpha=alpha, linewidth=0)
				axes[i,j].set_xlim(min(data_v[:,j]),max(data_v[:,j]))
				axes[i,j].set_ylim(0,1.05)
				if j !=0:
					plt.setp(axes[i,j].get_yticklabels(), visible=False)
				axes[i,j].vlines(np.percentile(data_v[:,j],15.865),axes[i,j].get_ylim()[0],axes[i,j].get_ylim()[1], color = 'k',alpha=alpha,linewidth = lw,linestyle = 'dashed')
				axes[i,j].vlines(np.percentile(data_v[:,j],100-15.865),axes[i,j].get_ylim()[0],axes[i,j].get_ylim()[1], color = 'k',alpha=alpha,linewidth = lw,linestyle = 'dashed')
				axes[i,j].vlines(np.percentile(data_v[:,j],50),axes[i,j].get_ylim()[0],axes[i,j].get_ylim()[1], color = 'k',alpha=alpha,linewidth = lw)
			if i>j:
				if j !=0:
					plt.setp(axes[i,j].get_yticklabels(), visible=False)
				P1 = axes[i,j].scatter(data_v[:,j],data_v[:,i],marker='x',alpha=0.3,
												c=param_error,vmin=0,vmax=color_max,cmap=cm.coolwarm,s=3)
				P2 = axes[i,j].scatter(data_tr[:,j],data_tr[:,i],c='k',marker='+',s=80)
				axes[i,j].set_xlim(min(data_tr[:,j])-0.1,max(data_tr[:,j])+0.1)
				axes[i,j].set_ylim(min(data_tr[:,i])-0.1,max(data_tr[:,i])+0.1)
			if j>i:
				axes[i,j].axis('off')
			if i == len(a.p0)-1:
				axes[i,j].set_xlabel(parameter_names[j])
			if j ==0:
				axes[i,j].set_ylabel(parameter_names[i])
			if i==2 and j == 1:
				cplot = axes[i,j].scatter(data_v[:,j],data_v[:,i],marker='.',alpha=0.3,
													c=param_error,vmin=0,vmax=color_max,cmap=cm.coolwarm,s=3)
				axes[i,j].set_xlim(min(data_tr[:,j])-0.1,max(data_tr[:,j])+0.1)
				axes[i,j].set_ylim(min(data_tr[:,i])-0.1,max(data_tr[:,i])+0.1)
	cax=fig.add_axes([0.82,0.06,0.02,0.9])
	plt.colorbar(cplot,cax=cax)

	plt.show()
	fig.savefig('Neural/'+dataset+'_corner_parameter_plot.png',bbox_inches='tight')

	return None



def max_err_corner_plot(dataset):
	""" This function plots a corner plot of the model parameters, with colors showing the max neural network error across all elements.
	The error corresponding to max color can be changed by the color_max parameter.

	Input:
		Name of dataset ('verif' or 'test')

	Output:
		corner_parameter_plot.png saved in Neural/ directory
	"""

	from matplotlib import cm

	a=ModelParameters()

	# Load datasets
	data_tr = np.load('Neural/training_param_grid.npy')
	data_v = np.load('Neural/'+dataset+'_param_grid.npy')

	_,_,param_error = calculate_errors(dataset) # Finds MAX error across all elements

	# Initialize plot
	plt.clf()
	text_size = 12
	plt.rc('font', family='serif',size = text_size)
	plt.rc('xtick', labelsize=text_size)
	plt.rc('ytick', labelsize=text_size)
	plt.rc('axes', labelsize=text_size, lw=1.0)
	plt.rc('lines', linewidth = 1)
	plt.rcParams['ytick.major.pad']='8'
	plt.rcParams['text.latex.preamble']=[r"\usepackage{libertine}"]
	params = {'text.usetex' : True,
          'font.family' : 'libertine',
          'text.latex.unicode': True,
          }
	plt.rcParams.update(params)
	parameter_names = [r'$\alpha_\mathrm{IMF}$',r'$\log_{10}(\mathrm{N_{Ia}})$',
                   r'$\log_{10}(\tau_\mathrm{Ia})$',r'$\log_{10}(\mathrm{SFE})$',
                   r'$\log_{10}(\mathrm{SFR_{peak}})$',r'x_{out}']


	# Plot settings
	fig,axes = plt.subplots(nrows = len(a.p0), ncols = len(a.p0),figsize=(14.69,8.0),dpi=300)
	alpha = 0.5
	lw=2 # Linewidth
	left = 0.1 # Left side of subplots
	right = 0.8 # Right side
	bottom = 0.075
	top = 0.97
	wspace = 0.0 # blankspace width between subplots
	hspace = 0.0 # blankspace height between subplots
	color_max = 0.06
	plt.subplots_adjust(left=left,bottom=bottom,right=right,top=top,wspace=wspace,hspace=hspace)

	# Create plot
	for i in range(len(a.p0)):
		for j in range(len(a.p0)):
			axes[i,j].locator_params(nbins=4)
			if j==1:
				axes[i,j].locator_params(nbins=4)
			if i==j:
				counts,edges = np.histogram(np.asarray(data_v[:,j]),bins=10)
				max_count = float(np.max(counts))
				counts = np.divide(counts,max_count)
				median = np.zeros(len(edges)-1)
				for k in range(len(edges)-1):
					choice = np.logical_and(np.greater(data_v[:,j],edges[k]),np.less(data_v[:,j],edges[k+1]))
					error=np.extract(choice,param_error)
					if len(error) != 0:
						median[k] = np.median(error)
				colors = cm.coolwarm(median/color_max)
				axes[i,j].bar(left = edges[:-1], height=counts, width = edges[1]-edges[0],
									color=colors,alpha=alpha, linewidth=0)
				axes[i,j].set_xlim(min(data_v[:,j]),max(data_v[:,j]))
				axes[i,j].set_ylim(0,1.05)
				if j !=0:
					plt.setp(axes[i,j].get_yticklabels(), visible=False)
				axes[i,j].vlines(np.percentile(data_v[:,j],15.865),axes[i,j].get_ylim()[0],axes[i,j].get_ylim()[1], color = 'k',alpha=alpha,linewidth = lw,linestyle = 'dashed')
				axes[i,j].vlines(np.percentile(data_v[:,j],100-15.865),axes[i,j].get_ylim()[0],axes[i,j].get_ylim()[1], color = 'k',alpha=alpha,linewidth = lw,linestyle = 'dashed')
				axes[i,j].vlines(np.percentile(data_v[:,j],50),axes[i,j].get_ylim()[0],axes[i,j].get_ylim()[1], color = 'k',alpha=alpha,linewidth = lw)
			if i>j:
				if j !=0:
					plt.setp(axes[i,j].get_yticklabels(), visible=False)
				P1 = axes[i,j].scatter(data_v[:,j],data_v[:,i],marker='x',alpha=0.3,
												c=param_error,vmin=0,vmax=color_max,cmap=cm.coolwarm,s=3)
				P2 = axes[i,j].scatter(data_tr[:,j],data_tr[:,i],c='k',marker='+',s=80)
				axes[i,j].set_xlim(min(data_tr[:,j])-0.1,max(data_tr[:,j])+0.1)
				axes[i,j].set_ylim(min(data_tr[:,i])-0.1,max(data_tr[:,i])+0.1)
			if j>i:
				axes[i,j].axis('off')
			if i == len(a.p0)-1:
				axes[i,j].set_xlabel(parameter_names[j])
			if j ==0:
				axes[i,j].set_ylabel(parameter_names[i])
			if i==2 and j == 1:
				cplot = axes[i,j].scatter(data_v[:,j],data_v[:,i],marker='.',alpha=0.3,
													c=param_error,vmin=0,vmax=color_max,cmap=cm.coolwarm,s=3)
				axes[i,j].set_xlim(min(data_tr[:,j])-0.1,max(data_tr[:,j])+0.1)
				axes[i,j].set_ylim(min(data_tr[:,i])-0.1,max(data_tr[:,i])+0.1)
	cax=fig.add_axes([0.82,0.06,0.02,0.9])
	plt.colorbar(cplot,cax=cax)

	plt.show()
	fig.savefig('Neural/'+dataset+'max_err_corner_parameter_plot.png',bbox_inches='tight')

	return None

def neural_output_int(test_input,a,b):
	""" This calculates the neural network predicted output for a trained network.
	This is cut down for speed.

	Inputs:
		test_input - Array containing unnormalized parameter values
		(coeffs - loaded automatically from Neural/neural_model.npz)

	Output:
		Neural network abundance prediction
	"""

	#a = ModelParameters()

	# Load in model coefficients
	#coeffs = np.load('Neural/neural_model.npz')

	# Normalize data for input into network
	norm_data = (test_input - a.p0)/np.array(a.training_widths)

	# Calculate neural network output
	hidden1 = np.tanh(np.array(np.dot(b.w_array_0,norm_data)+b.b_array_0))
	output = np.dot(b.w_array_1, hidden1)+b.b_array_1

	return output
	

def test_dataset(width,size):
	"""
	Create test dataset for fixed gaussian width.
	The data points are randomly distributed along a uniform distribution with fixed width in parameter space.
	
	Input: width of test dataset
	"""
	
	import warnings
	warnings.filterwarnings("ignore")
	
	directory = 'SingleElement/'
	if not os.path.exists(directory):
		os.makedirs(directory)

	a = ModelParameters()
	
	sigma = []
	for i,param_name in enumerate(a.to_optimize):
		sigma.append(a.priors.get(param_name)[1])

	param_grid = []
	abundance_grid = []
	for i in range(size):
		if i%10==0:
			print('Calculating sample %d of %d' %(i+1,size))
		param = np.random.uniform(a.p0-width*np.array(sigma),a.p0+width*np.array(sigma))
		pred,_ = posterior_function_returning_predictions((param,a))
		param_grid.append(list(param))
		abundance_grid.append(list(pred))
       
	np.save('SingleElement/'+str(width)+'_sigma_param_grid.npy',param_grid)
	np.save('SingleElement/'+str(width)+'_sigma_abundances.npy',abundance_grid)

	return None	
    
