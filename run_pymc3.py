## Run the PyMC3 inference for a given input dataset and parameters

import numpy as np
import pymc3 as pm
import pymc3.math as ma
import theano.tensor as tt
import time as ttime
import os,sys,json
from Chempy.parameter import ModelParameters
from configparser import ConfigParser

###########################
# Read in parameter file
if len(sys.argv)!=2:
	print("Please supply parameter file")
	sys.exit()

Config = ConfigParser()
Config.read(sys.argv[1])

neural_model = Config.get('input','neural_model')
mock_data_file = Config.get('input','mock_data_file')
outfile = Config.get('input','outfile')

all_n = json.loads(Config['inference']['all_n'])
max_stars = max(all_n)
elem_err = Config.getboolean('inference','elem_err')
max_iteration = Config.getint('inference','max_iteration')

chains = Config.getint('sampler','chains')
cores = Config.getint('sampler','cores')
tune = Config.getint('sampler','tune')
n_init = Config.getint('sampler','n_init')
n_samples = Config.getint('sampler','n_samples')

######################

os.chdir('/home/oliverphilcox/ChempyMulti/')
a=ModelParameters()

# Load in the neural network weights
model_numpy=np.load(neural_model)
w_array_0=np.matrix(model_numpy["w0"])
b_array_0=np.matrix(model_numpy["b0"])
w_array_1=np.matrix(model_numpy["w1"])
b_array_1=np.matrix(model_numpy["b1"])

# Load standardization parameters
input_mean=model_numpy.f.in_mean
input_std=model_numpy.f.in_std
output_mean=model_numpy.f.out_mean
output_std=model_numpy.f.out_std

# Load mock data
mock_data=np.load(mock_data_file)
true_Times = mock_data.f.true_time
all_els = mock_data.f.elements
mock_data.close()

# Define priors
Lambda_prior_mean = a.p0[:2]
Theta_prior_mean = a.p0[2:]
Lambda_prior_width = [0.3,0.3]
Theta_prior_width = [0.3,0.1,0.1]

# Now standardize
std_Lambda_prior_mean = (Lambda_prior_mean-input_mean[:2])/input_std[:2]
std_Lambda_prior_width = (Lambda_prior_width)/input_std[:2]
std_Theta_prior_mean = (Theta_prior_mean-input_mean[2:5])/input_std[2:5]
std_Theta_prior_width = (Theta_prior_width)/input_std[2:5]

# Define critical theta edge:
log_SFR_crit = 0.29402
std_log_SFR_crit = (log_SFR_crit-input_mean[3])/input_std[3]

# Define bounds on age to stop predicting out of parameter space:
min_time,max_time = [1.,13.8]
std_min_time,std_max_time=[(time-input_mean[-1])/input_std[-1] for time in [min_time,max_time]]

def n_star_inference(n_stars,iteration,elem_err=False,n_init=20000,n_samples=1000,max_stars=100):    
    ## Define which stars to use
    these_stars = np.arange(max_stars)[iteration*n_stars:(iteration+1)*n_stars]
    
    ## Load in mock dataset
    mock_data=np.load(mock_data_file) #dataset
    mu_times = mock_data.f.obs_time[these_stars] #time of birth
    sigma_times = mock_data.f.obs_time_err[these_stars] #error on age
    all_els = mock_data.f.elements

    full_abundances = mock_data.f.abundances[these_stars] # chemical element abundances for data
    full_errors = mock_data.f.abundance_errs[these_stars] # error on abundances

    # Filter out correct elements:
    els = ['C','Fe','He','Mg','N','Ne','O','Si'] # TNG elements
    n_els = len(els)
    el_indices=np.zeros(len(els),dtype=int)
    for e,el in enumerate(els):
        for j in range(len(all_els)):
            if els[e]==str(all_els[j]):
                el_indices[e]=j
                break
            if j==len(all_els)-1:
                print("Failed to find element %s"%el)
    obs_abundances = full_abundances[:,el_indices]
    obs_errors = full_errors[:,el_indices]

    # Now standardize dataset
    norm_data=(obs_abundances-output_mean)/output_std
    norm_sd = obs_errors/output_std

    data_obs = norm_data.ravel()
    data_sd = np.asarray(norm_sd).ravel()

    std_times_mean = (mu_times-input_mean[-1])/input_std[-1]
    std_times_width = sigma_times/input_std[-1]
    
    # Define stacked local priors
    Local_prior_mean = np.vstack([np.hstack([std_Theta_prior_mean,std_times_mean[i]]) for i in range(n_stars)])
    Local_prior_sigma = np.vstack([np.hstack([std_Theta_prior_width,std_times_width[i]]) for i in range(n_stars)])
    
    # Bound variables to ensure they don't exit the training parameter space
    lowBound = tt._shared(np.asarray([-5,std_log_SFR_crit,-5,std_min_time]))
    upBound = tt._shared(np.asarray([5,5,5,std_max_time]))
    
    # Create stacked mean and variances
    loc_mean=np.hstack([np.asarray(std_Theta_prior_mean).reshape(1,-1)*np.ones([n_stars,1]),std_times_mean.reshape(-1,1)])
    loc_std=np.hstack([np.asarray(std_Theta_prior_width).reshape(1,-1)*np.ones([n_stars,1]),std_times_width.reshape(-1,1)])
    
    # Share theano variables
    w0=tt._shared(w_array_0)
    b0=tt._shared(b_array_0)
    w1=tt._shared(w_array_1)
    b1=tt._shared(b_array_1)
    ones_tensor = tt.ones([n_stars,1])
    b0_all = ma.matrix_dot(ones_tensor,b0)
    b1_all = ma.matrix_dot(ones_tensor,b1)
    
    # Define PyMC3 Model
    simple_model=pm.Model()
    
    with simple_model:
        # Define priors
        Lambda = pm.Normal('Std-Lambda',mu=std_Lambda_prior_mean,
                            sd=std_Lambda_prior_width,
                            shape=(1,len(std_Lambda_prior_mean)))

        Locals = pm.Normal('Std-Local',mu=loc_mean,sd=loc_std,shape=loc_mean.shape,
                          transform=pm.distributions.transforms.Interval(lowBound,upBound),
                           )
        TimeSq = tt.reshape(Locals[:,-1]**2.,(n_stars,1))

        TruLa = pm.Deterministic('Lambda',Lambda*input_std[:2]+input_mean[:2])
        TruTh = pm.Deterministic('Thetas',Locals[:,:3]*input_std[2:5]+input_mean[2:5])
        TruTi = pm.Deterministic('Times',Locals[:,-1]*input_std[-1]+input_mean[-1])

        ## NEURAL NET
        Lambda_all = ma.matrix_dot(ones_tensor,Lambda)
        InputVariables = ma.concatenate([Lambda_all,Locals,TimeSq],axis=1)

        layer1 = ma.matrix_dot(InputVariables,w0)+b0_all
        output = ma.matrix_dot(ma.tanh(layer1),w1)+b1_all

        if elem_err:
            # ERRORS
            #element_error = pm.Normal('Element-Error',mu=-2,sd=1,shape=(1,n_els))
            element_error = pm.HalfCauchy('Std-Element-Error',beta=0.01/output_std,shape=(1,n_els))
            TruErr = pm.Deterministic('Element-Error',element_error*output_std)
            stacked_error = ma.matrix_dot(ones_tensor,element_error)
            tot_error = ma.sqrt(stacked_error**2.+norm_sd**2.) # NB this is all standardized by output_std here
        else:
            tot_error = norm_sd # NB: all quantities are standardized here

        predictions = pm.Deterministic("Predicted-Abundances",output*output_std+output_mean)

        # Define likelihood function (unravelling output to make a multivariate gaussian)
        likelihood=pm.Normal('likelihood', mu=output.ravel(), sd=tot_error.ravel(), 
                             observed=norm_data.ravel())
        
    # Now sample
    init_time = ttime.time()
    with simple_model:
        samples=pm.sample(draws=n_samples,chains=chains,cores=cores,tune=tune,
                          nuts_kwargs={'target_accept':0.9},init='advi+adapt_diag',n_init=n_init)
    end_time = ttime.time()-init_time

    def construct_output(samples):
        Lambda=samples.get_values('Lambda')[:,0,:]
        Thetas=samples.get_values('Thetas')[:,:,:]
        Times=samples.get_values('Times')[:,:]
        
        predictions = samples.get_values('Predicted-Abundances')[:,:,:]
        
        if elem_err:
            Errs = samples.get_values('Element-Error')[:,0,:]
            return Lambda,Thetas,Times,Errs,predictions
        else:
            return Lambda,Thetas,Times,predictions

    print("Finished after %.2f seconds"%end_time)
    
    if elem_err:
        Lambda,Thetas,Times,Errs,predictions=construct_output(samples)
        return Lambda,Thetas,Times,end_time,Errs,predictions
    else:
        Lambda,Thetas,Times,predictions=construct_output(samples)
        return Lambda,Thetas,Times,end_time,predictions
    


## RUN THE INFERENCE ##
chain_params=[]
for nn in all_n:
    mini_chain=[]
    for iteration in range(max_stars//nn):
        if iteration>=max_iteration:
            break
        print("Starting inference using %d stars iteration %d of %d"%(nn,iteration+1,min(max_iteration,max_stars//nn)))
        try:
            mini_chain.append(n_star_inference(nn,iteration,elem_err=elem_err,n_init=n_init,
                                               n_samples=n_samples,max_stars=max_stars))
        except ValueError or FloatingPointError:
            mini_chain.append(n_star_inference(nn,iteration,elem_err=elem_err,n_init=n_init,
                                                   n_samples=n_samples,max_stars=max_stars))
    chain_params.append(mini_chain)

## Save output
print("Saving output")
all_n = all_n[:len(chain_params)]
all_Lambda = [[cc[0] for cc in c] for c in chain_params]
all_Thetas = [[cc[1][:,:,:] for cc in c] for c in chain_params]
all_Times = [[cc[2] for cc in c] for c in chain_params]
all_timescale = [[cc[3] for cc in c] for c in chain_params]
if elem_err:
    all_Err = [[cc[4] for cc in c] for c in chain_params]
else:
    all_Err=0.
all_predictions = [[cc[-1] for cc in c] for c in chain_params]

mean_timescale = [np.mean(all_timescale[i],axis=0) for i in range(len(all_timescale))]

np.savez(outfile,n_stars=all_n,Lambdas=all_Lambda,Thetas=all_Thetas,Times=all_Times,
            runtimes=all_timescale,Errors=all_Err,mean_runtimes=mean_timescale)
print("Inference complete: output saved to %s"%outfile)
