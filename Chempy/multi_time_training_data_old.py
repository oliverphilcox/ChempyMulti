from Chempy.parameter import ModelParameters
a=ModelParameters()
import multiprocessing as mp
import numpy as np
import tqdm
import time
from Chempy.cem_function import multi_timestep_chempy


# First create training dataset
# Define training size
N = 15.  # 15 grid points per parameter
widths = np.asarray([0.6, 0.6, 0.6, 0.2, 0.2]) # true training data distribution widths

from scipy.stats import norm as gaussian
prob = np.linspace(1./(N+1.), 1.-1/(N+1), int(N))
grids = [gaussian.ppf(prob) for _ in range(len(a.p0))] # Normalize to unit Gaussian
norm_grid = np.array(np.meshgrid(*grids)).T.reshape(-1,len(a.p0))

# Create grid in parameter space
full_widths = list(widths)
means = list(a.p0)
param_grid = [np.asarray(item)*full_widths+means for item in norm_grid]

# Save parameter set:
np.savez("All_Timestep_Training_Parameters.npz",param_grid=param_grid,norm_grid=norm_grid)

N_samples=len(param_grid)

def runner(index):
    """Function to compute the Chempy predictions for each parameter set"""
    b=ModelParameters()
    params=param_grid[index]
    norm_params=norm_grid[index]
    output=multi_timestep_chempy((params,b))
    if type(output)==float:
        if output==np.inf:
            del b
        outs=np.zeros([len(els),len(timestep)]),params,norm_params
    else: 
        abun=output[0]
        del b;
        outs=abun,params,norm_params
    return outs

if __name__=='__main__':
    init_time=time.time()
    
    # Compute elements by running code once:
    output=multi_timestep_chempy((a.p0,a))
    if output==np.inf:
        raise Exception("Bad SFR input even for prior parameters - check for bugs")
    else:
        _,els,timestep=output
    
    N_samples=10
    
    # Now run multiprocessing
    cpus=mp.cpu_count()
    p=mp.Pool(min(16,cpus))
    output=list(tqdm.tqdm(p.imap_unordered(runner,range(N_samples)),total=N_samples))
    abuns=[o[0] for o in output]
    pars=[o[1] for o in output]
    n_pars=[o[2] for o in output]
    
    end_time=time.time()
    
    print("multiprocessing complete after %d seconds"%(end_time-init_time));
    
    # Now save output
    np.savez("All_Timestep_Training_Predictions",abundances=abuns,elements=els,params=pars,norm_params=n_pars,timestep=timestep);
