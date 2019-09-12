import numpy as np
import os
from .sfr import SFR
from .solar_abundance import solar_abundances
import time
from .data_to_test import likelihood_function, wildcard_likelihood_function, elements_plot, arcturus, sol_norm, plot_processes, save_abundances,  cosmic_abundance_standard, ratio_function, star_function, gas_reservoir_metallicity
import multiprocessing as mp
from .wrapper import initialise_stuff, Chempy, Chempy_all_times

try:
    from scipy.misc import logsumexp
except ImportError:
    from scipy.special import logsumexp ## for scipy version control

import numpy.ma as ma
from .data_to_test import likelihood_evaluation, read_out_wildcard, likelihood_evaluation_int

def gaussian_log(x,x0,xsig):
    '''
    function to calculate the gaussian probability (its normed to Pmax and given in log)

    INPUT:

    x = where is the data point or parameter value

    x0 = mu

    xsig = sigma
    '''
    return -np.divide((x-x0)*(x-x0),2*xsig*xsig)

def lognorm_log(x,mu,factor):
    '''
    this function provides Prior probability distribution where the factor away from the mean behaves like the sigma deviation in normal_log

    for example if mu = 1 and factor = 2

    for	1 it returns 0

    for 0,5 and 2 it returns -0.5

    for 0.25 and 4 it returns -2.0

    and so forth

    Can be used to specify the prior on the yield factors
    '''
    y = np.log(np.divide(x,mu))
    y = np.divide(y,np.log(factor))
    y = gaussian_log(y,0.,1.)
    return y

def gaussian(x,x0,xsig):
    '''
    function to calculate the gaussian probability (its normed to Pmax and given in log)

    INPUT:

    x = where is the data point or parameter value

    x0 = mu

    xsig = sigma
    '''
    factor = 1. / (np.sqrt(xsig * xsig * 2. * np.pi))
    exponent = -np.divide((x - x0) * (x - x0),2 * xsig * xsig)
    return factor * np.exp(exponent)

def lognorm(x,mu,factor):
    '''
    this function provides Prior probability distribution where the factor away from the mean behaves like the sigma deviation in normal_log
    BEWARE: this function is not a properly normalized probability distribution. It only provides relative values.

    INPUT:
    x = where to evaluate the function, can be an array
    mu = peak of the distribution
    factor = the factor at which the probability decreases to 1 sigma

    Can be used to specify the prior on the yield factors
    '''
    y = np.log(np.divide(x,mu))
    y = np.divide(y,np.log(factor))
    y = gaussian(y,0.,1.)
    return y

def shorten_sfr(a,age=None):
    '''
    This function crops the SFR to the length of the age of the star and ensures that enough stars are formed at the stellar birth epoch
    INPUT:
    a = Modelparameters
    OUTPUT:

    the function will update the modelparameters, such that the simulation will end when the star is born and it will also check whether there is enough sfr left at that epoch
    '''
    if age==None:
        try:
            star = np.load('%s.npy' %(a.stellar_identifier))
        except Exception as ex:
            from . import localpath
            star = np.load(localpath + 'input/stars/' + a.stellar_identifier + '.npy')
        age_of_star = star['age'][0]
    else:
        age_of_star=age
    assert (age_of_star <= 13.0), "Age of the star must be below 13Gyr"


    new_timesteps = int((a.time_steps-1)*a.end/(a.end-age_of_star)+1)
    new_end = (a.end-age_of_star)*(new_timesteps-1)/(a.time_steps-1)

    ## Compute SFR with the new end-point and timesteps
    basic_sfr = SFR(a.start,new_end,new_timesteps)
    ## Also compute old SFR without discretization of final time
    # This is to ensure we form the correct amount of mass in the simulation
    old_sfr = SFR(a.start,a.end,a.time_steps)



    if a.basic_sfr_name == 'gamma_function':
        getattr(basic_sfr, a.basic_sfr_name)(S0 = a.S_0 * a.mass_factor,a_parameter = a.a_parameter, loc = a.sfr_beginning, scale = a.sfr_scale)
        getattr(old_sfr, a.basic_sfr_name)(S0 = a.S_0 * a.mass_factor,a_parameter = a.a_parameter, loc = a.sfr_beginning, scale = a.sfr_scale)
    elif a.basic_sfr_name == 'model_A':
        basic_sfr.model_A(a.mass_factor*a.S_0,a.t_0,a.t_1)
        old_sfr.model_A(a.mass_factor*a.S_0,a.t_0,a.t_1)
    elif a.basic_sfr_name == 'prescribed':
        basic_sfr.prescribed(a.mass_factor, a.name_of_file)
        old_sfr.prescribed(a.mass_factor, a.name_of_file)
    elif a.basic_sfr_name == 'doubly_peaked':
        basic_sfr.doubly_peaked(S0 = a.mass_factor*a.S_0, peak_ratio = a.peak_ratio, decay = a.sfr_decay, t0 = a.sfr_t0, peak1t0 = a.peak1t0, peak1sigma = a.peak1sigma)
        old_sfr.prescribed(a.mass_factor, a.name_of_file)

    # compute a small correction from changing the simulation end time.
    # this is performed analytically here
    #
    # from scipy.stats import gamma
    # basic_gamma = np.sum(gamma.pdf(basic_sfr.t,a.a_parameter,a.sfr_beginning,a.sfr_scale))*basic_sfr.dt
    # old_gamma = np.sum(gamma.pdf(old_sfr.t,a.a_parameter,a.sfr_beginning,a.sfr_scale))*old_sfr.dt
    # correction = basic_gamma/old_gamma
    #a.total_mass*=correction


    ## NB: normalization doesn't matter here since it will be renormalized later
    basic_sfr.sfr = a.total_mass * np.divide(basic_sfr.sfr,sum(basic_sfr.sfr))
    mass_normalisation = a.total_mass
    mean_sfr = sum(basic_sfr.sfr) / new_end

    # at which time in the simulation is the star born
    init_end = a.end
    star_time = a.end-age_of_star#basic_sfr.t[-1] - age_of_star
    cut = [np.where(np.abs(basic_sfr.t - star_time) == np.min(np.abs(basic_sfr.t - star_time)))]
    if len(cut[0][0]) != 1:
        cut = cut[0][0][0]

    # updating the end time and the model steps and rescale the total mass
    time_model = float(basic_sfr.t[tuple(cut)])
    a.end = time_model
    a.time_steps = int(cut[0][0]) + 1
    a.total_mass = sum(basic_sfr.sfr[0:a.time_steps])

    # check whether the sfr is enough at end to produce reasonable number of stars (which is necessary in order to have a probability to observe a star at all)
    sfr_at_end = float(basic_sfr.sfr[tuple(cut)] / basic_sfr.dt)
    fraction_of_mean_sfr = sfr_at_end / mean_sfr

    #a.shortened_sfr_rescaling = a.total_mass / mass_normalisation

    # Compute SFR total mass rescaling from analytic integrals of the SFR
    from scipy.special import gammainc
    cut_sfr_mass = gammainc(a.a_parameter,star_time/a.sfr_scale)
    full_sfr_mass = gammainc(a.a_parameter,init_end/a.sfr_scale)
    a.shortened_sfr_rescaling = cut_sfr_mass/full_sfr_mass
    
    if fraction_of_mean_sfr<0.05:
        return np.inf
    else:
        return a

    #assert fraction_of_mean_sfr > 0.05, ('The total SFR of the last age bin is below 5% of the mean SFR', 'stellar identifier = ', a.stellar_identifier, 'star time = ', star_time, 'model time = ', time_model )

def cem(changing_parameter,a):
    '''
    This is the function calculating the chemical evolution for a specific parameter set (changing_parameter) and for a specific observational constraint specified in a (e.g. 'solar_norm' calculates the likelihood of solar abundances coming out of the model). It returns the posterior and a list of blobs. It can be used by an MCMC.
    This function actually encapsulates the real cem function in order to capture exceptions and in that case return -inf. This makes the MCMC runs much more stable
    INPUT:

    changing_parameter = parameter values of the free parameters as an array

    a = model parameters specified in parameter.py. There are also the names of free parameters specified here
    OUTPUT:

    log posterior, array of blobs

    the blobs contain the prior values, the likelihoods and the actual values of each predicted data point (e.g. elemental abundance value)
    '''
    try:
        posterior, blobs = cem_real(changing_parameter,a)
        return posterior, blobs
    except Exception as ex:
        import traceback; traceback.print_exc()
    return -np.inf, [0]

def cem_real(changing_parameter,a):
    '''
    real chempy function. description can be found in cem
    '''
    for i,item in enumerate(a.to_optimize):
        setattr(a, item, changing_parameter[i])
        val = getattr(a, item)

    start_time = time.time()
    ### PRIOR calculation, values are stored in parameter.py
    prior_names = []
    prior = []
    for name in a.to_optimize:
        (mean, std, functional_form) = a.priors.get(name)
        val = getattr(a, name)
        prior_names.append(name)
        if functional_form == 0:
            prior.append(gaussian_log(val, mean, std))
        elif functional_form == 1:
            prior.append(lognorm_log(val, mean, std))
    a.prior = prior

    for name in a.to_optimize:
        (lower, upper) = a.constraints.get(name)
        val = getattr(a, name)
        if lower is not None and val<lower:
            print('%s lower border is violated with %.5f' %(name,val))
            return -np.inf, [0]
        if upper is not None and val>upper:
            print('%s upper border is violated' %(name))
            return -np.inf, [0]

    if not a.testing_output:
        print(changing_parameter,mp.current_process()._identity[0])#,a.observational_constraints_index
    else:
        print(changing_parameter)

    ### So that the parameter can be plotted in linear space
    if 'log10_N_0' in a.to_optimize:
        a.N_0 = np.power(10,a.log10_N_0)
    if 'log10_sn1a_time_delay' in a.to_optimize:
        a.sn1a_time_delay = np.power(10,a.log10_sn1a_time_delay)
    if 'log10_starformation_efficiency' in a.to_optimize:
        a.starformation_efficiency = np.power(10,a.log10_starformation_efficiency)
    if 'log10_gas_reservoir_mass_factor' in a.to_optimize:
        a.gas_reservoir_mass_factor = np.power(10,a.log10_gas_reservoir_mass_factor)
    if 'log10_sfr_scale' in a.to_optimize:
        a.sfr_scale = np.power(10,a.log10_sfr_scale)

    if a.imf_type_name == 'salpeter':
        a.imf_parameter = (a.high_mass_slope)
    elif a.imf_type_name == 'Chabrier_2':
        a.imf_parameter = (a.chabrier_para1, a.chabrier_para2, a.chabrier_para3,a.high_mass_slope)
    elif a.imf_type_name == 'Chabrier_1':
        a.imf_parameter = (a.chabrier_para1, a.chabrier_para2, a.high_mass_slope)
    elif a.imf_type_name == 'normed_3slope':
        a.imf_parameter = (a.imf_slope_1,a.imf_slope_2,a.high_mass_slope,a.imf_break_1,a.imf_break_2)
    if a.time_delay_functional_form == 'maoz':
        a.sn1a_parameter = [a.N_0,a.sn1a_time_delay,a.sn1a_exponent,a.dummy]
    elif a.time_delay_functional_form == 'normal':
        a.sn1a_parameter = [a.number_of_pn_exlopding,a.sn1a_time_delay,a.sn1a_timescale,a.sn1a_gauss_beginning]
    elif a.time_delay_functional_form == 'gamma_function':
        a.sn1a_parameter = [a.sn1a_norm,a.sn1a_a_parameter,a.sn1a_beginning,a.sn1a_scale]

    basic_solar = solar_abundances()
    getattr(basic_solar, a.solar_abundance_name)()
    elements_to_trace = a.elements_to_trace

    directory = 'model_temp/'
    ### Model is calculated
    if a.calculate_model:
        cube, abundances = Chempy(a)
        cube1 = cube.cube
        gas_reservoir = cube.gas_reservoir
        if a.testing_output:
            if os.path.exists(directory):
                print(directory, ' already exists. Content might be overwritten')
            else:
                os.makedirs(directory)
            np.save(directory + '%s_elements_to_trace' %(a.name_string), elements_to_trace)
            np.save(directory + '%s_gas_reservoir' %(a.name_string),gas_reservoir)
            np.save(directory + '%s_cube' %(a.name_string),cube1)
            np.save(directory + '%s_abundances' %(a.name_string),abundances)
    else:
        cube1 = np.load(directory + '%s_cube.npy' %(a.name_string))
        abundances = np.load(directory + '%s_abundances.npy' %(a.name_string))
        gas_reservoir = np.load(directory + '%s_gas_reservoir.npy' %(a.name_string))
        elements_to_trace = np.load(directory + '%s_elements_to_trace.npy' %(a.name_string))


    ### LIKELIHOOD is being calculated
    a.probability = []
    a.abundance_list = []
    a.names = []
    ### these functions need to return lists of the probabilities, likelihoods and elements names. The latter ones are important for the blobs so that the MCMC result can be enriched with elemental likelihoods and the like.
    if 'gas_reservoir' in a.observational_constraints_index:
        probabilities, result, names = gas_reservoir_metallicity(a.summary_pdf,a.name_string,np.copy(abundances),np.copy(cube1),elements_to_trace,np.copy(gas_reservoir),a.number_of_models_overplotted,a.produce_mock_data,a.use_mock_data,a.error_inflation, np.copy(basic_solar.z))
        a.probability.append(probabilities)
        a.abundance_list.append(result)
        a.names.append(names)
    if 'sn_ratio' in a.observational_constraints_index:
        probabilities, result, names = ratio_function(a.summary_pdf,a.name_string,np.copy(abundances),np.copy(cube1),elements_to_trace,np.copy(gas_reservoir),a.number_of_models_overplotted,a.produce_mock_data,a.use_mock_data,a.error_inflation)
        a.probability.append(probabilities)
        a.abundance_list.append(result)
        a.names.append(names)
    if 'cas' in a.observational_constraints_index:
        probabilities, abundance_list, element_names = cosmic_abundance_standard(a.summary_pdf,a.name_string,np.copy(abundances),np.copy(cube1),elements_to_trace,np.copy(basic_solar.table),a.number_of_models_overplotted,a.produce_mock_data,a.use_mock_data,a.error_inflation)
        a.probability.append(probabilities)
        a.abundance_list.append(abundance_list)
        a.names.append(element_names)
    if 'sol_norm' in a.observational_constraints_index:
        probabilities, abundance_list, element_names = sol_norm(a.summary_pdf,a.name_string,np.copy(abundances),np.copy(cube1),elements_to_trace,a.element_names,np.copy(basic_solar.table),a.number_of_models_overplotted,a.produce_mock_data,a.use_mock_data,a.error_inflation)
        a.probability.append(probabilities)
        a.abundance_list.append(abundance_list)
        a.names.append(element_names)
    if 'arcturus' in a.observational_constraints_index:
        probabilities, abundance_list, element_names = arcturus(a.summary_pdf,a.name_string,np.copy(abundances),np.copy(cube1),elements_to_trace,a.element_names,np.copy(basic_solar.table),a.number_of_models_overplotted,a.arcturus_age,a.produce_mock_data,a.use_mock_data,a.error_inflation)
        a.probability.append(probabilities)
        a.abundance_list.append(abundance_list)
        a.names.append(element_names)
    if 'wildcard' in a.observational_constraints_index:
        probabilities, abundance_list, element_names = wildcard_likelihood_function(a.summary_pdf,a.stellar_identifier, np.copy(abundances))
        a.probability.append(probabilities)
        a.abundance_list.append(abundance_list)
        a.names.append(element_names)
    ### These functions are for plotting and saving purposes they just return 0 probability not interfering with the likelihood. But they will make the MCMC blobs crash. Therefore take them out when running the MCMC
    if 'stars_at_end' in a.observational_constraints_index:
        a.probability.append(star_function(a.summary_pdf,a.name_string,np.copy(abundances),np.copy(cube1),elements_to_trace,np.copy(gas_reservoir),a.number_of_models_overplotted))
    if 'save_abundances' in a.observational_constraints_index:
        a.probability.append(save_abundances(a.summary_pdf,a.name_string,np.copy(abundances)))
    if 'plot_processes' in a.observational_constraints_index:
        a.probability.append(plot_processes(a.summary_pdf,a.name_string,cube.sn2_cube,cube.sn1a_cube,cube.agb_cube,a.element_names,np.copy(cube1),a.number_of_models_overplotted))
    if 'elements' in a.observational_constraints_index:
        a.probability.append(elements_plot(a.name_string,basic_ssp.agb.elements, basic_ssp.sn2.elements, basic_ssp.sn1a.elements,elements_to_trace, basic_solar.table,60))

    ### to flatten the sublists so that the likelihood can be calculated and the blobs are in a flattened format
    a.names =  [item for sublist in a.names for item in sublist]
    a.names += ['m-%s' %(item) for item in a.names]
    a.abundance_list = [item for sublist in a.abundance_list for item in sublist]
    a.probability = [item for sublist in a.probability for item in sublist]
    a.names += prior_names
    if a.testing_output:
        #print a.names
        np.save("model_temp/blobs_name_list", a.names)
    if np.isnan(sum(a.probability)):
        return -np.inf, [0]
    if a.testing_output:
        print('l: ', sum(a.probability), 'pr: ', sum(a.prior), 'po: ', sum(a.prior) + sum(a.probability))#, mp.current_process()._identity[0]
    else:
        print('l: ', sum(a.probability), 'pr: ', sum(a.prior), 'po: ', sum(a.prior) + sum(a.probability),'|', mp.current_process()._identity[0])
    return sum(a.probability) + sum(a.prior), np.hstack((a.probability,a.abundance_list,a.prior))

def cem2(a):
    '''
    This is the function calculating the chemical evolution for a specific parameter set (changing_parameter) and for a specific observational constraint specified in a (e.g. 'solar_norm' calculates the likelihood of solar abundances coming out of the model). It returns the posterior and a list of blobs. It can be used by an MCMC.
    This function actually encapsulates the real cem function in order to capture exceptions and in that case return -inf. This makes the MCMC runs much more stable
    INPUT:

    a = model parameters specified in parameter.py and alteres by posterior_function
    OUTPUT:

    predictions, name_of_prediction

    the predicted element abundances for the time of the birth of the star (specified in a) are given back, as well as the corona metallicity at that time and the SN-ratio at that time.
    '''
    try:
        posterior, blobs = cem_real2(a)
        return posterior, blobs
    except Exception as ex:
        import traceback; traceback.print_exc()
    return -np.inf, [0]

def cem_real2_all_times(a):
    """ real chempy function for returning predictions at all timesteps."""
    a = shorten_sfr(a,0.) # use 13 max possible time here
    if a==np.inf:
        return np.inf

    basic_solar = solar_abundances()
    getattr(basic_solar,a.solar_abundance_name)() # need for normalizations

    elements_to_trace = list(a.elements_to_trace)

    cube, abundances = Chempy_all_times(a)
    if type(cube)==float:
        # Something became negative - discard this run.
        return np.inf
    cube1 = cube.cube
    gas_reservoir = cube.gas_reservoir

    # List of timesteps:
    time_steps = cube.time[1:]

    abundance_list = np.zeros([len(elements_to_trace),len(time_steps)])

    # predicted values are written out and returned together with corona metallicity and SN-ratio
    #abundance_list = []
    for i,item in enumerate(elements_to_trace):
        abundance_list[i][:]=abundances[item][1:]

    """

    abundance_list[-2]=gas_reservoir['Z'][1:]
    elements_to_trace.append('Zcorona')
    abundance_list[-1]=np.divide(cube1['sn2'][1:],cube1['sn1a'][1:])
    elements_to_trace.append('SNratio')
    """
    return(abundance_list, elements_to_trace, time_steps)


def cem_real2_single_time(a,this_time):
    """ real chempy function for returning predictions at a single timestep."""
    age=a.end-this_time
    a = shorten_sfr(a,age) # use 13 max possible time here
    if a==np.inf:
        return np.inf

    basic_solar = solar_abundances()
    getattr(basic_solar,a.solar_abundance_name)() # need for normalizations

    elements_to_trace = list(a.elements_to_trace)

    cube, abundances = Chempy_all_times(a)
    if type(cube)==float:
        # Something became negative - discard this run.
        return np.inf
    cube1 = cube.cube
    gas_reservoir = cube.gas_reservoir

    abundance_list = np.zeros(len(elements_to_trace))

    # predicted values are written out and returned together with corona metallicity and SN-ratio
    #abundance_list = []
    for i,item in enumerate(elements_to_trace):
        abundance_list[i]=abundances[item][-1]

    """

    abundance_list[-2]=gas_reservoir['Z'][1:]
    elements_to_trace.append('Zcorona')
    abundance_list[-1]=np.divide(cube1['sn2'][1:],cube1['sn1a'][1:])
    elements_to_trace.append('SNratio')
    """
    return(abundance_list, elements_to_trace)

def cem_real2(a):
    '''
    real chempy function. description can be found in cem2. \
    If a.UseNeural==True, then this uses the output from a PRE-TRAINED neural network instead.
    '''
    ## The time until which Chempy is calculated is cropped to the stellar birth time. Also the SFR should not be below 1/20th of the mean SFR
    a = shorten_sfr(a)
    basic_solar = solar_abundances()
    getattr(basic_solar, a.solar_abundance_name)()
    elements_to_trace = list(a.elements_to_trace)
    directory = 'model_temp/'
    ### Model is calculated
    if a.UseNeural==True:
        # Alternative path using a Neural network to predict the outcome instead of Chempy
        from Chempy.neural import neural_output
        param = [a.high_mass_slope,a.log10_N_0,a.log10_starformation_efficiency,a.log10_sfr_scale,a.outflow_feedback_fraction]
        neural_abundances = neural_output(param)
        elements_to_trace.append('Zcorona')
        elements_to_trace.append('SNratio')
        abundance_list=[]
        j=0 # This indexes neural_abundances for ordering
        for i,name in enumerate(elements_to_trace):
            if name in a.neural_names:
                abundance_list.append(neural_abundances[j]) # Required elements for later
                j = j+1
            else:
                abundance_list.append(0) # All unwanted elements set to arbitrary value

    else:
        if a.calculate_model:
            cube, abundances = Chempy(a)
            cube1 = cube.cube
            gas_reservoir = cube.gas_reservoir
            if a.testing_output:
                if os.path.exists(directory):
                    if a.verbose:
                        print(directory, ' already exists. Content might be overwritten')
                else:
                    os.makedirs(directory)
                np.save(directory + '%s_elements_to_trace' %(a.name_string), elements_to_trace)
                np.save(directory + '%s_gas_reservoir' %(a.name_string),gas_reservoir)
                np.save(directory + '%s_cube' %(a.name_string),cube1)
                np.save(directory + '%s_abundances' %(a.name_string),abundances)
        else:
            cube1 = np.load(directory + '%s_cube.npy' %(a.name_string))
            abundances = np.load(directory + '%s_abundances.npy' %(a.name_string))
            gas_reservoir = np.load(directory + '%s_gas_reservoir.npy' %(a.name_string))
            elements_to_trace = np.load(directory + '%s_elements_to_trace.npy' %(a.name_string))

        # predicted values are written out and returned together with corona metallicity and SN-ratio
        abundance_list = []
        for item in elements_to_trace:
            abundance_list.append(abundances[item][-1])

        abundance_list.append(gas_reservoir['Z'][-1])
        elements_to_trace.append('Zcorona')

        abundance_list.append(cube1['sn2'][-1]/cube1['sn1a'][-1])
        elements_to_trace.append('SNratio')

    return(abundance_list,elements_to_trace)


def posterior_function(changing_parameter,a):
    '''
    The posterior function is the interface between the optimizing function and Chempy. Usually the likelihood will be calculated with respect to a so called 'stellar wildcard'.
    Wildcards can be created according to the tutorial 6. A few wildcards are already stored in the input folder. Chempy will try the current folder first. If no wildcard npy file with the name a.stellar_identifier is found it will look into the Chempy/input/stars folder.
    INPUT:

    changing_parameter = parameter values of the free parameters as an array

    a = model parameters specified in parameter.py. There are also the names of free parameters specified here
    OUTPUT:

    log posterior, array of blobs

    the blobs contain the likelihoods and the actual values of each predicted data point (e.g. elemental abundance value)
    '''
    try:
        posterior, blobs = posterior_function_real(changing_parameter,a)
        return posterior, blobs
    except Exception as ex:
        import traceback; traceback.print_exc()
    return -np.inf, [0]


def posterior_function_real(changing_parameter,a):
    '''
    This is the actual posterior function. But the functionality is explained in posterior_function.
    '''

    start_time = time.time()
    # the values in a are updated according to changing_parameters and the prior list is appended
    a = extract_parameters_and_priors(changing_parameter, a)


    # the log prior is calculated
    prior = sum(np.log(a.prior))


    precalculation = time.time()
    #print('precalculation: ', start_time - precalculation)

    # The endtime is changed for the actual calculation but restored to default afterwards
    backup = a.end ,a.time_steps, a.total_mass

    a.shortened_sfr_rescaling = 1. # Restore in order to not rescale all the time?

    if a.stellar_identifier is 'prior':
        likelihood = 0.
        abundance_list = 0
    else:
        # call Chempy and return the abundances at the end of the simulation = time of star's birth and the corresponding element names as a list
        abundance_list,elements_to_trace = cem_real2(a)
        a.end ,a.time_steps, a.total_mass = backup

        # The last two entries of the abundance list are the Corona metallicity and the SN-ratio
        abundance_list = abundance_list[:-2]
        elements_to_trace = elements_to_trace[:-2]

        model = time.time()
        #print('model: ', precalculation - model)

        # a likelihood is calculated where the model error is optimized analytically if you do not want model error uncomment one line in the likelihood function
        likelihood, element_list, model_error, star_error_list, abundance_list, star_abundance_list = likelihood_function(a.stellar_identifier, abundance_list, elements_to_trace)
        #likelihood = 0.
        #abundance_list = [0]

    error_optimization = time.time()
    #print('error optimization: ', model - error_optimization)
    if a.verbose:
        if not a.testing_output:
            print('prior = ', prior, 'likelihood = ', likelihood, mp.current_process()._identity[0])
        else:
            print('prior = ', prior, 'likelihood = ', likelihood)

    return(prior+likelihood,abundance_list)


def posterior_function_for_minimization(changing_parameter,a):
    '''
    calls the posterior function but just returns the negative log posterior instead of posterior and blobs
    '''
    posterior, blobs = posterior_function(changing_parameter,a)
    return -posterior

def posterior_function_returning_predictions(args):
    '''
    calls the posterior function but just returns the negative log posterior instead of posterior and blobs
    '''
    changing_parameter,a = args
    posterior, abundance_list, element_list = posterior_function_predictions(changing_parameter,a)
    return abundance_list,element_list

def multi_timestep_chempy(args):
    """
    Calls Chempy and returns posterior values for all timesteps, up to some maximum age.
    """
    changing_parameter,a=args

    # Update the parameters + priors:
    a = extract_parameters_and_priors(changing_parameter,a)

    # Save the output time etc. which is changed by the actual calculation but restored to default afterwards
    backup = a.end, a.time_steps, a.total_mass

    # Call Chempy and return all abundances and the list of element names
    output=cem_real2_all_times(a)
    if output==np.inf:
        return np.inf
    else:
        abundance_list, elements_to_trace, timesteps = output

        # Restore default values
        a.end, a.time_steps, a.total_mass = backup

        return abundance_list, elements_to_trace, timesteps

def single_timestep_chempy(args):
    """
    Calls Chempy and returns posterior values for a single pre-determined timestep, given in args.
    """
    all_parameter,a=args
    changing_parameter=all_parameter[:-1] # physics parameters
    birth_time = all_parameter[-1] # time of star birth

    # Update the parameters + priors:
    a = extract_parameters_and_priors(changing_parameter,a)

    # Save the output time etc. which is changed by the actual calculation but restored to default afterwards
    backup = a.end, a.time_steps, a.total_mass

    # Call Chempy and return all abundances and the list of element names
    output=cem_real2_single_time(a,birth_time)
    if output==np.inf:
        return np.inf
    else:
        abundance_list, elements_to_trace = output

        # Restore default values
        a.end, a.time_steps, a.total_mass = backup

        return abundance_list, elements_to_trace

def posterior_function_predictions(changing_parameter,a):
    '''
    This is like posterior_function_real. But returning the predicted elements as well.
    '''

    start_time = time.time()
    # the values in a are updated according to changing_parameters and the prior list is appended
    a = extract_parameters_and_priors(changing_parameter, a)


    # the log prior is calculated
    prior = sum(np.log(a.prior))


    precalculation = time.time()
    #print('precalculation: ', start_time - precalculation)

    # The endtime is changed for the actual calculation but restored to default afterwards
    backup = a.end ,a.time_steps, a.total_mass


    # call Chempy and return the abundances at the end of the simulation = time of star's birth and the corresponding element names as a list
    abundance_list,elements_to_trace = cem_real2(a)

    a.end ,a.time_steps, a.total_mass = backup

    # The last two entries of the abundance list are the Corona metallicity and the SN-ratio
    abundance_list = abundance_list[:-2]
    elements_to_trace = elements_to_trace[:-2]

    model = time.time()
    #print('model: ', precalculation - model)

    # a likelihood is calculated where the model error is optimized analytically if you do not want model error uncomment one line in the likelihood function
    likelihood, element_list, model_error, star_error_list, abundance_list, star_abundance_list = likelihood_function(a.stellar_identifier, abundance_list, elements_to_trace)
    #likelihood = 0.
    #abundance_list = [0]

    error_optimization = time.time()
    #print('error optimization: ', model - error_optimization)
    if a.verbose:
        if not a.testing_output:
            print('prior = ', prior, 'likelihood = ', likelihood, mp.current_process()._identity[0])
        else:
            print('prior = ', prior, 'likelihood = ', likelihood)

    return(prior+likelihood,abundance_list, element_list)

def get_prior(changing_parameter, a):
    """
    This function calculates the prior probability
    INPUT:
    changing_parameter = the values of the parameter vector
    a = the model parameters including the names of the parameters (which is needed to identify them with the prescribed priors in parameters.py)
    OUTPUT:
    the log prior is returned
    """

    for i,item in enumerate(a.to_optimize):
        setattr(a, item, changing_parameter[i])
        val = getattr(a, item)

    ### PRIOR calculation, values are stored in parameter.py
    prior_names = []
    prior = []
    for name in a.to_optimize:
        (mean, std, functional_form) = a.priors.get(name)
        val = getattr(a, name)
        prior_names.append(name)
        if functional_form == 0:
            prior.append(gaussian(val, mean, std))
        elif functional_form == 1:
            prior.append(lognorm(val, mean, std))
    return(sum(np.log(prior)))

def global_optimization(changing_parameter, result):
    '''
    This function is a buffer function if global_optimization_real fails and it only returns the negative posterior
    '''
    try:
        posterior, error_list, elements = global_optimization_real(changing_parameter, result)
        return posterior
    except Exception as ex:
        import traceback; traceback.print_exc()
    return np.inf

def global_optimization_error_returned(changing_parameter, result):
    '''
    this is a buffer function preventing failures from global_optimization_real and returning all its output including the best model error
    '''
    try:
        posterior, error_list, elements = global_optimization_real(changing_parameter, result)
        return -posterior, error_list, elements
    except Exception as ex:
        import traceback; traceback.print_exc()
    return np.inf, [0], [0]

def global_optimization_real(changing_parameter, result):
    '''
    This function calculates the predictions from several Chempy zones in parallel. It also calculates the likelihood for common model errors
    BEWARE: Model parameters are called as saved in parameters.py!!!
    INPUT:
    changing_parameter = the global SSP parameters (parameters that all stars share)
    result = the complete parameter set is handed over as an array of shape(len(stars),len(all parameters)). From those the local ISM parameters are taken

    OUTPUT:
    -posterior = negative log posterior for all stellar zones
    error_list = the optimal standard deviation of the model error
    elements = the corresponding element symbols
    '''
    import multiprocessing as mp
    import numpy.ma as ma
    from scipy.stats import beta
    from .cem_function import get_prior, posterior_function_returning_predictions
    from .data_to_test import likelihood_evaluation
    from .parameter import ModelParameters

    ## Calculating the prior
    a = ModelParameters()
    a.to_optimize = a.SSP_parameters_to_optimize
    prior = get_prior(changing_parameter,a)

    ## Handing over to posterior_function_returning_predictions
    parameter_list = []
    p0_list = []
    for i,item in enumerate(a.stellar_identifier_list):
        parameter_list.append(ModelParameters())
        parameter_list[-1].stellar_identifier = item
        p0_list.append(np.hstack((changing_parameter,result[i,len(a.SSP_parameters):])))
    args = zip(p0_list,parameter_list)
    p = mp.Pool(len(parameter_list))
    t = p.map(posterior_function_returning_predictions, args)
    p.close()
    p.join()
    z = np.array(t)
    # Predictions including element symbols are returned

    # Reading out the wildcards
    elements = np.unique(np.hstack(z[:,1]))
    from Chempy.data_to_test import read_out_wildcard
    args = zip(a.stellar_identifier_list, z[:,0], z[:,1])
    list_of_l_input = []
    for item in args:
        list_of_l_input.append(read_out_wildcard(*item))
        list_of_l_input[-1] = list(list_of_l_input[-1])
    # Now the input for the likelihood evaluating function is almost ready


    # Masking the elements that are not given for specific stars and preparing the likelihood input
    star_errors = ma.array(np.zeros((len(elements),len(a.stellar_identifier_list))), mask = True)
    star_abundances = ma.array(np.zeros((len(elements),len(a.stellar_identifier_list))), mask = True)
    model_abundances = ma.array(np.zeros((len(elements),len(a.stellar_identifier_list))), mask = True)

    for star_index,item in enumerate(list_of_l_input):
        for element_index,element in enumerate(item[0]):
            assert element in elements, 'observed element is not predicted by Chempy'
            new_element_index = np.where(elements == element)[0][0]
            star_errors[new_element_index,star_index] = item[1][element_index]
            model_abundances[new_element_index,star_index] = item[2][element_index]
            star_abundances[new_element_index,star_index] = item[3][element_index]

    # Brute force testing of a few model errors
    model_errors = np.linspace(a.flat_model_error_prior[0],a.flat_model_error_prior[1],a.flat_model_error_prior[2])
    if a.beta_error_distribution[0]:
        if 'log10_beta' in a.to_optimize:
            a.beta = 10**changing_parameter[0]
        elif 'beta_param' in a.to_optimize:
            a.beta = changing_parameter[0]
        else:
            a.beta = a.beta_param
        error_weight = beta.pdf(model_errors, a = a.beta_error_distribution[1], b = a.beta)
        error_weight/= sum(error_weight)
    else:
        error_weight = np.ones_like(model_errors) * 1./float(flat_model_error_prior[2])
    error_list = []
    likelihood_list = []
    for i,element in enumerate(elements):
        error_temp = []
        for item in model_errors:
            error_temp.append(likelihood_evaluation(item, star_errors[i] , model_abundances[i], star_abundances[i]))
        cut = np.where(np.hstack(error_temp)==np.max(error_temp))
        if len(cut) == 2:
            cut = cut[0][0]
        error_list.append(float(model_errors[cut]))
    ### So that the parameter can be plotted in linear space
    if 'log10_beta' in a.to_optimize:
        a.beta_param = np.power(10,a.log10_beta)
        ## Adding the marginalization over the model error (within the prior borders). Taking the average of the likelihoods (they are log likelihoods so exp needs to be called)
        if a.error_marginalization:
            likelihood_list.append(logsumexp(error_temp, b = error_weight))
        else:
            if a.zero_model_error:
                likelihood_list.append(error_temp[0])
            else:
                likelihood_list.append(np.max(error_temp))

    error_list = np.hstack(error_list)
    likelihood_list = np.hstack(likelihood_list)
    likelihood = np.sum(likelihood_list)

    # returning the best likelihood together with the prior as posterior
    return(-(prior + likelihood), error_list, elements)

def extract_parameters_and_priors(changing_parameter, a):
    '''
    This function extracts the parameters from changing parameters and writes them into the ModelParamaters (a), so that Chempy can evaluate the changed parameter settings
    '''
    for i,item in enumerate(a.to_optimize):
        setattr(a, item, changing_parameter[i])
        val = getattr(a, item)

    start_time = time.time()
    ### PRIOR calculation, values are stored in parameter.py
    prior_names = []
    prior = []
    for name in a.to_optimize:
        (mean, std, functional_form) = a.priors.get(name)
        val = getattr(a, name)
        prior_names.append(name)
        if functional_form == 0:
            prior.append(gaussian(val, mean, std))
        elif functional_form == 1:
            prior.append(lognorm(val, mean, std))
    a.prior = prior

    # check the borders of the free parameters
    for name in a.to_optimize:
        (lower, upper) = a.constraints.get(name)
        val = getattr(a, name)
    #	if lower is not None and val<lower:
    #		assert False, '%s lower border is violated with %.5f' %(name,val)
    #	if upper is not None and val>upper:
    #		assert False, '%s upper border is violated' %(name)
    if a.verbose:
        if not a.testing_output:
            print(changing_parameter,mp.current_process()._identity[0])#,a.observational_constraints_index
        else:
            print(changing_parameter)

    ### So that the parameter can be plotted in linear space
    if 'log10_beta' in a.to_optimize:
        a.beta_param = np.power(10,a.log10_beta)
        a.beta_error_distribution[2] = np.power(10,a.log10_beta)
    if 'beta_param' in a.to_optimize:
        a.beta_error_distribution[2] = a.beta_param
    if 'log10_N_0' in a.to_optimize:
        a.N_0 = np.power(10,a.log10_N_0)
    if 'log10_sn1a_time_delay' in a.to_optimize:
        a.sn1a_time_delay = np.power(10,a.log10_sn1a_time_delay)
    if 'log10_starformation_efficiency' in a.to_optimize:
        a.starformation_efficiency = np.power(10,a.log10_starformation_efficiency)
    if 'log10_gas_reservoir_mass_factor' in a.to_optimize:
        a.gas_reservoir_mass_factor = np.power(10,a.log10_gas_reservoir_mass_factor)
    if 'log10_sfr_scale' in a.to_optimize:
        a.sfr_scale = np.power(10,a.log10_sfr_scale)
    if 'log10_sfr_factor_for_cosmic_accretion' in a.to_optimize:
        a.sfr_factor_for_cosmic_accretion = np.power(10.,a.log10_sfr_factor_for_cosmic_accretion)

    if a.imf_type_name == 'salpeter':
        a.imf_parameter = (a.high_mass_slope)
    elif a.imf_type_name == 'Chabrier_2':
        a.imf_parameter = (a.chabrier_para1, a.chabrier_para2, a.chabrier_para3,a.high_mass_slope)
    elif a.imf_type_name == 'Chabrier_1':
        a.imf_parameter = (a.chabrier_para1, a.chabrier_para2, a.high_mass_slope)
    elif a.imf_type_name == 'Chabrier_TNG':
        a.imf_parameter = (a.chabrier_para1, a.chabrier_para2, a.chabrier_para3, a.chabrier_para4, a.high_mass_slope)
    elif a.imf_type_name == 'normed_3slope':
        a.imf_parameter = (a.imf_slope_1,a.imf_slope_2,a.high_mass_slope,a.imf_break_1,a.imf_break_2)
    if a.time_delay_functional_form == 'maoz':
        a.sn1a_parameter = [a.N_0,a.sn1a_time_delay,a.sn1a_exponent,a.dummy]
    elif a.time_delay_functional_form == 'normal':
        a.sn1a_parameter = [a.number_of_pn_exlopding,a.sn1a_time_delay,a.sn1a_timescale,a.sn1a_gauss_beginning]
    elif a.time_delay_functional_form == 'gamma_function':
        a.sn1a_parameter = [a.sn1a_norm,a.sn1a_a_parameter,a.sn1a_beginning,a.sn1a_scale]
    return(a)

def posterior_function_local_for_minimization(changing_parameter, stellar_identifier, global_parameters, errors, elements):
    '''
    calls the local posterior function but just returns the negative log posterior instead of posterior and blobs
    '''
    posterior, blobs = posterior_function_local(changing_parameter, stellar_identifier, global_parameters, errors, elements)
    return -posterior

def posterior_function_local(changing_parameter, stellar_identifier, global_parameters, errors, elements):
    '''
    The posterior function is the interface between the optimizing function and Chempy. Usually the likelihood will be calculated with respect to a so called 'stellar wildcard'.
    Wildcards can be created according to the tutorial 6 from the github page. A few wildcards are already stored in the input folder. Chempy will try the current folder first. If no wildcard npy file with the name a.stellar_identifier is found it will look into the Chempy/input/stars folder.
    INPUT:

    changing_parameter = parameter values of the free parameters as an array

    a = model parameters specified in parameter.py. There are also the names of free parameters specified here
    global_parameters = the SSP Parameters which are fixed for this optimization but need to be handed over to Chempy anyway
    errors = the model error for each element
    elements = the corresponding names of the elements
    OUTPUT:

    log posterior, array of blobs

    the blobs contain the actual values of each predicted data point (e.g. elemental abundance value)
    '''
    try:
        posterior, blobs = posterior_function_local_real(changing_parameter, stellar_identifier, global_parameters, errors, elements)
        return posterior, blobs
    except Exception as ex:
        import traceback; traceback.print_exc()
    return -np.inf, [0]

def posterior_function_local_real(changing_parameter, stellar_identifier, global_parameters, errors, elements):
    '''
    This is the actual posterior function. But the functionality is explained in posterior_function.
    '''
    from .parameter import ModelParameters
    a = ModelParameters()
    a.stellar_identifier = stellar_identifier

    start_time = time.time()
    # the values in a are updated according to changing_parameters and the prior list is appended
    changing_parameter = np.hstack((global_parameters,changing_parameter))
    a = extract_parameters_and_priors(changing_parameter, a)


    # the log prior is calculated
    prior = sum(np.log(a.prior))


    precalculation = time.time()
    #print('precalculation: ', start_time - precalculation)

    # The endtime is changed for the actual calculation but restored to default afterwards
    #backup = a.end ,a.time_steps, a.total_mass

    # call Chempy and return the abundances at the end of the simulation = time of star's birth and the corresponding element names as a list
    abundance_list,elements_to_trace = cem_real2(a)
    #a.end ,a.time_steps, a.total_mass = backup

    # The last two entries of the abundance list are the Corona metallicity and the SN-ratio
    abundance_list = abundance_list[:-2]
    elements_to_trace = elements_to_trace[:-2]

    model = time.time()
    #print('model: ', precalculation - model)

    # a likelihood is calculated where the model error is optimized analytically if you do not want model error uncomment one line in the likelihood function
    if a.error_marginalization:
        from scipy.stats import beta
        likelihood_list = []
        model_errors = np.linspace(a.flat_model_error_prior[0],a.flat_model_error_prior[1],a.flat_model_error_prior[2])
        if a.beta_error_distribution[0]:
            if 'log10_beta' in a.to_optimize:
                a.beta_param = 10**changing_parameter[0]
            elif 'beta_param' in a.to_optimize:
                a.beta_param = changing_parameter[0]
            else:
                a.beta_param = a.beta_param # no change
            error_weight = beta.pdf(model_errors, a = a.beta_error_distribution[1], b = a.beta_param)
            error_weight/= sum(error_weight)
        else:
            error_weight = np.ones_like(model_errors) * 1./float(flat_model_error_prior[2])
        for i, item in enumerate(model_errors):
            error_temp = np.ones_like(errors) * item
            likelihood_temp, element_list, model_error, star_error_list, abundance_list_dump, star_abundance_list = likelihood_function(a.stellar_identifier, abundance_list, elements_to_trace, fixed_model_error = error_temp, elements = elements)
            likelihood_list.append(likelihood_temp)
        likelihood = logsumexp(likelihood_list, b = error_weight)
        abundance_list = abundance_list_dump
    else:
        if a.zero_model_error:
            errors = np.zeros_like(errors)
        likelihood, element_list, model_error, star_error_list, abundance_list, star_abundance_list = likelihood_function(a.stellar_identifier, abundance_list, elements_to_trace, fixed_model_error = errors, elements = elements)
    #likelihood = 0.
    #abundance_list = [0]

    error_optimization = time.time()
    #print('error optimization: ', model - error_optimization)
    if a.verbose:
        if not a.testing_output:
            print('prior = ', prior, 'likelihood = ', likelihood, mp.current_process()._identity[0])
        else:
            print('prior = ', prior, 'likelihood = ', likelihood)

    return(prior+likelihood,abundance_list)


def posterior_function_many_stars(changing_parameter,error_list,elements):
    '''
    The posterior function is the interface between the optimizing function and Chempy. Usually the likelihood will be calculated with respect to a so called 'stellar wildcard'.
    Wildcards can be created according to the tutorial 6. A few wildcards are already stored in the input folder. Chempy will try the current folder first. If no wildcard npy file with the name a.stellar_identifier is found it will look into the Chempy/input/stars folder.
    The posterior function for many stars evaluates many Chempy instances for different stars and adds up their common likelihood. The list of stars is given in parameter.py under stellar_identifier_list.
    The names in the list must be represented by wildcards in the same folder.
    INPUT:

    changing_parameter = parameter values of the free parameters as an array

    error_list = the model error list for each element
    elements = the corresponding element symbols
    OUTPUT:

    log posterior, array of blobs

    the blobs contain the likelihoods and the actual values of each predicted data point (e.g. elemental abundance value)
    '''
    try:
        posterior, blobs = posterior_function_many_stars_real(changing_parameter,error_list,elements)
        return posterior, blobs
    except Exception as ex:
        import traceback; traceback.print_exc()
    return -np.inf, [0]


def posterior_function_many_stars_real(changing_parameter,error_list,error_element_list):
    '''
    This is the actual posterior function for many stars. But the functionality is explained in posterior_function_many_stars.
    '''
    import numpy.ma as ma
    from .cem_function import get_prior, posterior_function_returning_predictions
    from .data_to_test import likelihood_evaluation, read_out_wildcard
    from .parameter import ModelParameters

    ## Initialising the model parameters
    a = ModelParameters()

    ## extracting from 'changing_parameters' the global parameters and the local parameters
    global_parameters = changing_parameter[:len(a.SSP_parameters)]
    local_parameters = changing_parameter[len(a.SSP_parameters):]
    local_parameters = local_parameters.reshape((len(a.stellar_identifier_list),len(a.ISM_parameters)))

    ## getting the prior for the global parameters in order to subtract it in the end for each time it was evaluated too much
    a.to_optimize = a.SSP_parameters_to_optimize
    global_parameter_prior = get_prior(global_parameters,a)

    ## Chempy is evaluated one after the other for each stellar identifier with the prescribed parameter combination and the element predictions for each star are stored
    predictions_list = []
    elements_list = []
    log_prior_list = []

    for i, item in enumerate(a.stellar_identifier_list):
        b = ModelParameters()
        b.stellar_identifier = item
        changing_parameter = np.hstack((global_parameters,local_parameters[i]))
        args = (changing_parameter,b)
        abundance_list,element_list = posterior_function_returning_predictions(args)
        predictions_list.append(abundance_list)
        elements_list.append(element_list)
        log_prior_list.append(get_prior(changing_parameter,b))

    ## The wildcards are read out so that the predictions can be compared with the observations
    args = zip(a.stellar_identifier_list, predictions_list, elements_list)
    list_of_l_input = []
    for item in args:
        list_of_l_input.append(read_out_wildcard(*item))
        list_of_l_input[-1] = list(list_of_l_input[-1])

    ## Here the predictions and observations are brought into the same array form in order to perform the likelihood calculation fast
    elements = np.unique(np.hstack(elements_list))
    # Masking the elements that are not given for specific stars and preparing the likelihood input
    star_errors = ma.array(np.zeros((len(elements),len(a.stellar_identifier_list))), mask = True)
    star_abundances = ma.array(np.zeros((len(elements),len(a.stellar_identifier_list))), mask = True)
    model_abundances = ma.array(np.zeros((len(elements),len(a.stellar_identifier_list))), mask = True)

    for star_index,item in enumerate(list_of_l_input):
        for element_index,element in enumerate(item[0]):
            assert element in elements, 'observed element is not predicted by Chempy'
            new_element_index = np.where(elements == element)[0][0]
            star_errors[new_element_index,star_index] = item[1][element_index]
            model_abundances[new_element_index,star_index] = item[2][element_index]
            star_abundances[new_element_index,star_index] = item[3][element_index]

    ## given model error from error_list is read out and brought into the same element order (compatibility between python 2 and 3 makes the decode method necessary)
    if not a.error_marginalization:
        error_elements_decoded = []
        for item in error_element_list:
            error_elements_decoded.append(item)#.decode('utf8')) # DECODING NOT NEEDED IN PYTHON 3
        error_element_list = np.hstack(error_elements_decoded)


        error_list = np.hstack(error_list)
        model_error = []
        for element in elements:
            assert element in error_element_list, 'for this element the model error was not given, %s' %(element)
            model_error.append(error_list[np.where(error_element_list == element)])
        model_error = np.hstack(model_error)




    ## likelihood is calculated (the model error vector is expanded)
    if a.error_marginalization:
        from scipy.stats import beta
        likelihood_list = []
        model_errors = np.linspace(a.flat_model_error_prior[0],a.flat_model_error_prior[1],a.flat_model_error_prior[2])
        if a.beta_error_distribution[0]:
            if 'log10_beta' in a.to_optimize:
                beta_param = 10**changing_parameter[0] # must be first value in parameter list
            elif 'beta_param' in a.to_optimize:
                beta_param = changing_parameter[0]
            else:
                beta_param = a.beta_param
            error_weight = beta.pdf(model_errors, a = a.beta_error_distribution[1], b = a.beta_error_distribution[2])
            error_weight/= sum(error_weight)
        else:
            error_weight = np.ones_like(model_errors) * 1./float(flat_model_error_prior[2])
        for i, item in enumerate(model_errors):
            error_temp = np.ones(len(elements)) * item
            likelihood_list.append(likelihood_evaluation(error_temp[:,None], star_errors , model_abundances, star_abundances))
        likelihood = logsumexp(likelihood_list, b = error_weight)
    else:
        if a.zero_model_error:
            model_error = np.zeros_like(model_error)
        likelihood = likelihood_evaluation(model_error[:,None], star_errors , model_abundances, star_abundances)

    ## Prior from all stars is added
    prior = sum(log_prior_list)
    ## Prior for global parameters is subtracted
    prior -= (len(a.stellar_identifier_list)-1) * global_parameter_prior
    posterior = prior+likelihood
    assert np.isnan(posterior) == False, ('returned posterior = ', posterior , 'prior = ' , prior, 'likelihood = ', likelihood, 'changing parameter = ', changing_parameter )
    ########
    if a.verbose:
        print('prior = ', prior, 'likelihood = ', likelihood)

    return(posterior,model_abundances)

def posterior_function_for_integration(changing_parameter,b):
    '''
    This is the actual posterior function. But the functionality is explained in posterior_function.
    This is a cut down version for integration - ONLY using beta function, and solar data

    Inputs:
        changing_parmeter is 6D parameter vector
        b is file from the score_function preload_vars file to avoid multiple calculation

    MUST CHECK THAT THE MODIFIED LIKELIHOOD FUNCTION GIVES THE CORRECT RESULTS

    ERRORS ARE NOT TREATED CORRECTLY HERE - RELOOK AT
    '''
    from .parameter import ModelParameters
    #from .cem_function import posterior_function_returning_predictions
    from .data_to_test import likelihood_evaluation_int

    a = ModelParameters()
    #stellar_identifier = a.stellar_identifier

    # the values in a are updated according to changing_parameters and the prior list is appended
    a = extract_parameters_and_priors(changing_parameter, a)

    # the log prior is calculated
    prior = sum(np.log(a.prior))

    #wildcard = np.load('Chempy/input/stars/Proto-sun.npy')
    wildcard = b.wildcard

    # call Chempy and return the abundances at the end of the simulation = time of star's birth and the corresponding element names as a list
    #abundance_list,elements_to_trace = cem_real2(a)
    abundance_list = cem_real2_int(a,b)

    # The last two entries of the abundance list are the Corona metallicity and the SN-ratio
    list_of_abundances = abundance_list[:-2]
    #elements_to_trace = elements_to_trace[:-2]

    #elements = []
    abundance_list = []
    #star_abundance_list = []
    #star_error_list = []
    # Change to arrays to speed up?
    for i,item in enumerate(a.elements_to_trace):
        if item in list(wildcard.dtype.names):
            #elements.append(item)
            abundance_list.append(float(list_of_abundances[i]))
            #star_abundance_list.append(float(wildcard[item][0]))
            #star_error_list.append(float(wildcard[item][1]))
    #abundance_list = np.hstack(abundance_list)
    #star_abundance_list = np.hstack(star_abundance_list)
    #star_error_list = np.hstack(star_error_list)
    #elements = np.hstack(elements)
    star_abundance_list = b.star_abundance_list
    star_error_list = b.star_error_list
    elements = b.elements

    # a likelihood is calculated where the model error is optimized analytically if you do not want model error uncomment one line in the likelihood function
    #from scipy.stats import beta
    #model_errors = np.linspace(a.flat_model_error_prior[0],a.flat_model_error_prior[1],a.flat_model_error_prior[2])
    #error_weight = beta.pdf(model_errors, a = a.beta_error_distribution[1], b = a.beta_error_distribution[2])
    #error_weight/= sum(error_weight)
    model_errors = b.model_errors
    errors_list = b.errors_list

    likelihood_list = np.zeros(len(model_errors))
    # Can we vectorize this??
    for i, item in enumerate(model_errors):
        likelihood_list[i] = likelihood_evaluation_int(errors_list[i], abundance_list, star_abundance_list)
    likelihood = logsumexp(likelihood_list, b = b.error_weight)

    return(prior+likelihood)

def cem_real2_int(a,b):
    '''
    real chempy function. description can be found in cem2. \
    If a.UseNeural==True, then this uses the output from a PRE-TRAINED neural network instead.
    This is a cut-down version for integrations
    '''
    from .neural import neural_output_int
    ## The time until which Chempy is calculated is cropped to the stellar birth time. Also the SFR should not be below 1/20th of the mean SFR
    #a = shorten_sfr(a)
    #basic_solar = solar_abundances()
    #getattr(basic_solar, a.solar_abundance_name)()
    #elements_to_trace = list(a.elements_to_trace)
    #directory = 'model_temp/'
    ### Model is calculated

    #if a.UseNeural==True:

    # Alternative path using a Neural network to predict the outcome instead of Chempy
    param = []
    if 'high_mass_slope' in a.to_optimize:
        param.append(a.high_mass_slope)
    if 'log10_N_0' in a.to_optimize:
        param.append(a.log10_N_0)
    if 'log10_sn1a_time_delay' in a.to_optimize:
        param.append(a.log10_sn1a_time_delay)
    if 'log10_starformation_efficiency' in a.to_optimize:
        param.append(a.log10_starformation_efficiency)
    if 'log10_sfr_scale' in a.to_optimize:
        param.append(a.log10_sfr_scale)
    if 'outflow_feedback_fraction' in a.to_optimize:
        param.append(a.outflow_feedback_fraction)
    #if 'log10_starformation_efficiency' not in a.to_optimize:
    #	a.log10_starformation_efficiency = np.log10(a.starformation_efficiency)
    #if 'log10_sfr_scale' not in a.to_optimize:
    #	a.log10_sfr_scale = np.log10(a.sfr_scale)
    #param = [a.high_mass_slope,a.log10_N_0,a.log10_starformation_efficiency,a.log10_sfr_scale,a.outflow_feedback_fraction] # don't include beta here
    neural_abundances = neural_output_int(param,a,b)
    #elements_to_trace.append('Zcorona')
    #elements_to_trace.append('SNratio')
    abundance_list=np.zeros(len(b.elements_to_trace))
    j=0 # This indexes neural_abundances for ordering
    for i,name in enumerate(b.elements_to_trace):
        if name in a.neural_names:
            abundance_list[i] = neural_abundances[j] # Required elements for later
            j = j+1
        else:
            abundance_list[i] = 0 # All unwanted elements set to arbitrary value

    #else:
    #if a.calculate_model:
    #	cube, abundances = Chempy(a)
    #	cube1 = cube.cube
    #	gas_reservoir = cube.gas_reservoir
    #	if a.testing_output:
    #		if os.path.exists(directory):
    #			if a.verbose:
    #				print(directory, ' already exists. Content might be overwritten')
    #		else:
    #			os.makedirs(directory)
    #		np.save(directory + '%s_elements_to_trace' %(a.name_string), elements_to_trace)
    #		np.save(directory + '%s_gas_reservoir' %(a.name_string),gas_reservoir)
    #		np.save(directory + '%s_cube' %(a.name_string),cube1)
    #		np.save(directory + '%s_abundances' %(a.name_string),abundances)
    #else:
    #	cube1 = np.load(directory + '%s_cube.npy' %(a.name_string))
    #	abundances = np.load(directory + '%s_abundances.npy' %(a.name_string))
    #	gas_reservoir = np.load(directory + '%s_gas_reservoir.npy' %(a.name_string))
    #	elements_to_trace = np.load(directory + '%s_elements_to_trace.npy' %(a.name_string))

    # predicted values are written out and returned together with corona metallicity and SN-ratio
    #abundance_list = []
    #for item in elements_to_trace:
    #	abundance_list.append(abundances[item][-1])

    #abundance_list.append(gas_reservoir['Z'][-1])
    #elements_to_trace.append('Zcorona')

    #abundance_list.append(cube1['sn2'][-1]/cube1['sn1a'][-1])
    #elements_to_trace.append('SNratio')

    #return(abundance_list,elements_to_trace)
    return(abundance_list)


def posterior_function_mcmc_quick(changing_parameter,error_element_list,preload):
    '''
    This is the actual posterior function for many stars. But the functionality is explained in posterior_function_many_stars.
    '''
    from .parameter import ModelParameters
    from scipy.stats import beta

    ## Initialising the model parameters
    a = ModelParameters()

    ## extracting from 'changing_parameters' the global parameters and the local parameters
    #global_parameters = changing_parameter[:len(a.SSP_parameters)]
    #local_parameters = changing_parameter[len(a.SSP_parameters):]
    #local_parameters = local_parameters.reshape((len(a.stellar_identifier_list),len(a.ISM_parameters)))

    ## getting the prior for the global parameters in order to subtract it in the end for each time it was evaluated too much
    #a.to_optimize = a.SSP_parameters_to_optimize
    #global_parameter_prior = get_prior(global_parameters,a)

    ## Chempy is evaluated one after the other for each stellar identifier with the prescribed parameter combination and the element predictions for each star are stored

    # REMOVE
    predictions_list = []
    elements_list = []
    #log_prior_list = []

    #for i, item in enumerate(a.stellar_identifier_list):
    #	b = ModelParameters()
    #	b.stellar_identifier = item
    #	changing_parameter = np.hstack((global_parameters,local_parameters[i]))
    #	args = (changing_parameter,b)
    #	abundance_list,element_list = posterior_function_returning_predictions(args)
    #	predictions_list.append(abundance_list)
    #	elements_list.append(element_list)
    #	#log_prior_list.append(get_prior(changing_parameter,b))
    prior = get_prior(changing_parameter,a)

    # SPEED UP
    abundance_list = list(posterior_function_predictions_quick(changing_parameter,a,preload))
    element_list = preload.elements

    # REMOVE
    predictions_list.append(abundance_list)
    elements_list.append(element_list)

    ## The wildcards are read out so that the predictions can be compared with the observations
    args = zip(a.stellar_identifier_list, predictions_list, elements_list)
    list_of_l_input = []

    output = (list(preload.elements),preload.star_error_list,np.array(predictions_list[0]),preload.star_abundance_list)

    for item in args:
        #list_of_l_input.append(read_out_wildcard(*item))
        list_of_l_input.append(output)
        list_of_l_input[-1] = list(list_of_l_input[-1])

    # Here the predictions and observations are brought into the same array form in order to perform the likelihood calculation fast
    #elements = preload.elements
    elements = np.unique(np.hstack(elements_list))
    #print("Elements in posterior_function_mcmc_quick are",elements)
    # Masking the elements that are not given for specific stars and preparing the likelihood input
    #star_errors = ma.array(np.zeros((len(elements),len(a.stellar_identifier_list))), mask = True)
    star_abundances = ma.array(np.zeros((len(elements),len(a.stellar_identifier_list))), mask = True)
    model_abundances = ma.array(np.zeros((len(elements),len(a.stellar_identifier_list))), mask = True)

    for star_index,item in enumerate(list_of_l_input):
        for element_index,element in enumerate(item[0]):
            assert element in elements, 'observed element is not predicted by Chempy'
            new_element_index = np.where(elements == element)[0][0]
    #    	star_errors[new_element_index,star_index] = item[1][element_index]
            model_abundances[new_element_index,star_index] = item[2][element_index]
            star_abundances[new_element_index,star_index] = item[3][element_index]
    star_errors = preload.star_error_list
    #star_abundances = preload.star_abundance_list
    #model_abundances = predictions_list
    elements = preload.elements
    #print("Second time in posterior_function_mcmc_quick:",elements)

    ## given model error from error_list is read out and brought into the same element order (compatibility between python 2 and 3 makes the decode method necessary)

    ## likelihood is calculated (the model error vector is expanded)
    #from scipy.stats import beta
    #model_errors = np.linspace(a.flat_model_error_prior[0],a.flat_model_error_prior[1],a.flat_model_error_prior[2])
    model_errors = preload.model_errors
    likelihood_list = np.zeros(len(model_errors))
    #error_weight = beta.pdf(model_errors, a = a.beta_error_distribution[1], b = a.beta_error_distribution[2])
    #error_weight/= sum(error_weight)

    if 'beta_param' in a.to_optimize:
        a.beta_error_distribution[2] = changing_parameter[0] # beta must be FIRST value in a.p0
    if 'log10_beta' in a.to_optimize:
        a.beta_error_distribution[2] = np.power(10,changing_parameter[0])
        a.beta_param = np.power(10,changing_parameter[0])
    error_weight = beta.pdf(model_errors, a = a.beta_error_distribution[1], b = a.beta_error_distribution[2])
    error_weight/= sum(error_weight)


    # Bug fix to avoid NaN values - fix this to beta = 1 value
    for i in range(len(error_weight)):
        if np.isnan(error_weight[i]):
            error_weight = np.ones(len(error_weight))/len(error_weight)


    for i in range(len(model_errors)):
        #likelihood_list[i] = likelihood_evaluation_int(e,m,s)
        #from .data_to_test import likelihood_evaluation
        #error_temp = np.ones(len(elements))*item
        #likelihood_list[i] = likelihood_evaluation(error_temp[:,None],star_errors,model_abundances,star_abundances)
        #err = np.sqrt(np.multiply(error_temp[:,None],error_temp[:,None]) + np.multiply(star_errors,star_errors))

        ## VECTORIZE THIS?
        likelihood_list[i] = likelihood_evaluation_int(preload.err[i], model_abundances,star_abundances)
    #print(likelihood_list[-1])
    #print(likelihood_evaluation_int(preload.err[-1],model_abundances,star_abundances))
    likelihood = logsumexp(likelihood_list, b = error_weight)
    #print(likelihood)
    ## Prior from all stars is added
    #prior = sum(log_prior_list)
    ## Prior for global parameters is subtracted
    #prior -= (len(a.stellar_identifier_list)-1) * global_parameter_prior
    #posterior = prior+likelihood
    #assert np.isnan(posterior) == False, ('returned posterior = ', posterior , 'prior = ' , prior, 'likelihood = ', likelihood, 'changing parameter = ', changing_parameter )
    ########
    #if a.verbose:
    #	print('prior = ', prior, 'likelihood = ', likelihood)
    return (prior+likelihood,model_abundances)

def posterior_function_predictions_quick(changing_parameter,a,preload):
    '''
    This is like posterior_function_real. This is cut down for one zone, for MCMC	'''

    # the values in a are updated according to changing_parameters and the prior list is appended
    #a = extract_parameters_and_priors(changing_parameter, a)
    for i,item in enumerate(a.to_optimize):
        setattr(a, item, changing_parameter[i])
        #val = getattr(a, item)


    # call Chempy and return the abundances at the end of the simulation = time of star's birth and the corresponding element names as a list
    abundance_list = cem_real2_int(a,preload)

    list_of_abundances = abundance_list[:-2]

    abundance_list = []
    for i,item in enumerate(a.elements_to_trace):
        if item in list(preload.wildcard.dtype.names):
            abundance_list.append(float(list_of_abundances[i]))
    #abundance_list = np.hstack(abundance_list)

    return abundance_list

def posterior_function_for_minimization_quick(changing_parameter,a,preload):
    '''
    calls the posterior function but just returns the negative log posterior instead of posterior and blobs
    '''
    posterior, blobs = posterior_function_quick(changing_parameter,a,preload)

    return -posterior

def posterior_function_quick(changing_parameter,a,preload):
    '''
    This is the actual posterior function. But the functionality is explained in posterior_function.

    THIS USES OPTIMAL MODEL ERRORS - NOT BETA FUNCTION!!!!
    '''

    #start_time = time.time()
    # the values in a are updated according to changing_parameters and the prior list is appended
    a = extract_parameters_and_priors(changing_parameter, a)


    # the log prior is calculated
    prior = sum(np.log(a.prior))


    #precalculation = time.time()
    #print('precalculation: ', start_time - precalculation)

    # The endtime is changed for the actual calculation but restored to default afterwards
    backup = a.end ,a.time_steps, a.total_mass

    #if a.stellar_identifier is 'prior':
    #	likelihood = 0.
    #	abundance_list = 0
    #else:
        # call Chempy and return the abundances at the end of the simulation = time of star's birth and the corresponding element names as a list
    abundance_list = cem_real2_int(a,preload)
    a.end ,a.time_steps, a.total_mass = backup

        # The last two entries of the abundance list are the Corona metallicity and the SN-ratio
    list_of_abundances = abundance_list[:-2]
    #elements_to_trace = elements_to_trace[:-2]

    #model = time.time()
        #print('model: ', precalculation - model)

        # a likelihood is calculated where the model error is optimized analytically if you do not want model error uncomment one line in the likelihood function
    wildcard = preload.wildcard
    abundance_list = []
    element_list = []
    star_abundance_list = []
    star_error_list = []
    for i,item in enumerate(preload.elements_to_trace):
        if item in list(wildcard.dtype.names):
            element_list.append(item)
            abundance_list.append(float(list_of_abundances[i]))
            star_abundance_list.append(float(wildcard[item][0]))
            star_error_list.append(float(wildcard[item][1]))
    abundance_list = np.hstack(abundance_list)
    star_abundance_list = np.hstack(star_abundance_list)
    star_error_list = np.hstack(star_error_list)

    #print('In posterior_function_quick elements are',element_list)

    model_error = []
    for i, item in enumerate(element_list):
        if (abundance_list[i] - star_abundance_list[i]) * (abundance_list[i] - star_abundance_list[i]) <= star_error_list[i] * star_error_list[i]:
            model_error.append(0.)
        else:
            model_error.append(np.sqrt((abundance_list[i] - star_abundance_list[i]) * (abundance_list[i] - star_abundance_list[i]) - star_error_list[i] * star_error_list[i]))
    model_error = np.hstack(model_error)

    likelihood = likelihood_evaluation(model_error, star_error_list, abundance_list, star_abundance_list)

    #likelihood, element_list, model_error, star_error_list, abundance_list, star_abundance_list = likelihood_function(a.stellar_identifier, abundance_list, preload.elements_to_trace)
        #likelihood = 0.
        #abundance_list = [0]

    #error_optimization = time.time()
    #print('error optimization: ', model - error_optimization)
    #if a.verbose:
    #	if not a.testing_output:
    #		print('prior = ', prior, 'likelihood = ', likelihood, mp.current_process()._identity[0])
    #	else:
    #		print('prior = ', prior, 'likelihood = ', likelihood)

    return prior+likelihood,abundance_list
