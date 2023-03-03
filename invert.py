#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 12:18:11 2023

@author: llinares
"""

import forward_function as forward
import geometric_scaling_factors
# import site_parameters
from constants import constants
from jax import random, jit
import numpyro
import numpy as np
import numpyro.distributions as dist
from numpyro.distributions import constraints, ImproperUniform
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS
import parameters

""" Input seismic scenario """
seismic_scenario={}
SR = 0.1 # long term slip rate of your fault (mm/yr)
#SR = param_site/cumulative_heigth
preexp = 200 # Pre-expositionn period (yr)
erosion_rate = 0 # Erosion rate (mm/yr)
number_of_events = 3

seismic_scenario['SR'] = SR
seismic_scenario['preexp'] = preexp
seismic_scenario['erosion_rate'] = erosion_rate

""" Input parameters"""
param=parameters.param()
alpha = param.alpha # colluvial wedge slope (degrees)
beta = param.beta # fault-plane dip (degrees)
gamma = param.gamma # upper surface slope (degrees)
scarp_height = param.H_scarp # total post-glacial height of the fault-plane (cm)
rho_coll = param.rho_coll # colluvial wedge mean density
rho_rock = param.rho_rock  # rock sample mean density
trench_depth = param.trench_depth # depth of your trench (cm), set to 0 if no samples where collected under the colluvial wedge
Hfinal = param.Hfinal # (cm)
cumulative_height = param.long_term_relief # cumulative height (mm)

# import data files
data = param.data.copy()
coll = param.coll.copy()
sf = param.sf.copy()
cl36AMS = param.cl36AMS
sig_cl36AMS = param.sig_cl36AMS
Data = cl36AMS

depth_rock, depth_coll, surf_rock, S_S = geometric_scaling_factors.neutron_scaling(param, constants, 5)
scaling_factors={}
scaling_factors['Lambda_f_e'] = surf_rock['lambda_e']
scaling_factors['so_f_e'] = surf_rock['s_e']
scaling_factors['Lambda_f_diseg'] = depth_rock['lambda_diseg']
scaling_factors['so_f_diseg'] = depth_rock['s_diseg']
scaling_factors['Lambda_f_beta_inf'] = depth_coll['lambda_beta']
scaling_factors['so_f_beta_inf'] = depth_coll['s_beta']
scaling_factors['S_S'] = S_S


rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)


def model(obs):

    """ Ages and seismic slip are inversed"""
    
    ages = numpyro.sample('ages', ImproperUniform(constraints.positive_ordered_vector, (), event_shape=(number_of_events,)))
    # # # print('ages', ages)
    ages = ages[::-1]
    # ages = np.array([7100, 2900, 560])
    # slips = jnp.zeros((number_of_events))
    
    # slips = jnp.array([400, 400, 200, 24])
    
    slips=np.array([1200, 500, 300])
    # slips1 = numpyro.sample('slip_1', ImproperUniform(constraints.less_than(Hfinal-10), (), event_shape=(1,))).astype(float)
    # slips = slips.at[0].set(jnp.array(slips1[0], float))

    # for i in range(1, 4):
    #     slips2 = numpyro.sample('slip_'+str(i+1), ImproperUniform(((constraints.less_than(Hfinal-jnp.sum(slips[0:i])) and constraints.greater_than(0.1))), (), event_shape=(1,)))
    #     slips = slips.at[i].set(jnp.array(slips2[0], float))

    # quiescence = numpyro.sample('quiescence', dist.Uniform(0, 100*1e3))

    # if quiescence != 0:
    #     ages = jnp.hstack((quiescence, ages))
    #     slips = jnp.hstack((0, slips))

    # if trench_depth != 0:
    #     ages = jnp.hstack((ages, 0))
    #     slips = jnp.hstack((slips, trench_depth))
    # print('slips :', slips, ', type', type(slips))
    # print(slips[0])
    # print('ages :', ages, 'type', type(slips))
    # print('quiescence: ', quiescence)
    seismic_scenario['ages'] = ages
    seismic_scenario['slips'] = slips
    # seismic_scenario['quiescence'] = quiescence
    t = forward.mds_no_scale(param, constants, seismic_scenario, scaling_factors)
    # t=0
    return numpyro.sample('obs', dist.Normal(t, 0.5), obs=obs)


""" usage MCMC """
rng_key = random.PRNGKey(0)
rng_key, rng_key_ = random.split(rng_key)
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=6000)
mcmc.run(rng_key, obs=Data)
mcmc.print_summary()
