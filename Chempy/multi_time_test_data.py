from Chempy.parameter import ModelParameters
a=ModelParameters()
import multiprocessing as mp
import numpy as np
import tqdm
import time
from Chempy.cem_function import single_timestep_chempy


# First create test parameter set
N_samples = int(1e4) # number of elements in test set
widths = np.asarray([0.6, 0.6, 0.6, 0.2, 0.2]) # prior widths used to generate training data
from scipy.stats import norm,uniform
all_params=np.zeros([N_samples,len(a.p0)+1])
for i in range(len(all_params)):
    trial_par-np.zeros(len(a.p0))
    while trial_par[3]<0.29402: # to avoid SFR errors
        trial_par=norm.rvs(loc=a.p0,scale=widths)
    all_params[i,:5]=trial_par
    all_params[i,5]=uniform.rvs(0.5,13.5) # generate age uniform distribution

def runner(index):
    """Function to compute the Chempy predictions for each parameter set"""
    b=ModelParameters()
    params=all_params[index]
    output=single_timestep_chempy((params,b))
    if type(output)==float:
        if output==np.inf:
            del b
        outs=np.zeros(len(els)),params
    else: 
        abun=output[0]
        del b;
        outs=abun,params
    return outs

if __name__=='__main__':
    init_time=time.time()
    
    # Compute elements by running code once:
    output=single_timestep_chempy((a.p0,a))
    if output==np.inf:
        raise Exception("Bad SFR input even for prior parameters - check for bugs")
    else:
        _,els=output
    
    # Now run multiprocessing
    cpus=mp.cpu_count()
    p=mp.Pool(min(16,cpus))
    output=list(tqdm.tqdm(p.imap_unordered(runner,range(N_samples)),total=N_samples))
    abuns=[o[0] for o in output]
    pars=[o[1] for o in output]
    
    end_time=time.time()
    
    print("multiprocessing complete after %d seconds"%(end_time-init_time));
    
    # Now save output
    np.savez("Variable_Time_Test_Predictions.npz",abundances=abuns,elements=els,params=pars);
