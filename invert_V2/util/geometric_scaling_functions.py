#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:10:04 2023

@author: llinares
"""

import numpy as np
from numpy import sin, cos, tan, arctan, pi, exp
import scipy.optimize

""" Useful functions for the scaling of chlorine 36 production due to cosmic rays
    Modified for python from Shlengenhauf et al. 2010 (https://doi.org/10.1111/j.1365-246X.2010.04622.x)
    and Tesson & Benedetti 2019 (link)
    
    Changes : exponential fit done prior to forward function and through bayesian inference """

def exp_law(x, s, Lambda):
    
    """ This function calculate an exponential decay function defined as follows:
        INPUTS : x, absisca values, numpy array
                 s, initial constant, float
                Lambda, decay constant, float
        OUTPUT : y-values, numpy array """
        
    return s * exp(-x/Lambda) 

def fitexp(x, y):
    
    """ This function is used to fit parameters of the exp_law function with scipy
        f(x) = n0*exp(-x/Lambda)
        
        INPUTS : x, absisca values, numpy array
                y, values you want to fit, numpy array 
        
        OUTPUTS : n_0, fitted initial constant, float
                 lambda_fit, fitted decay constant, float """
                 
    p0 = (0.9, 100) # start with values near those we expect
    params, cv = scipy.optimize.curve_fit(exp_law, x, y, p0)
    n_0, lambda_fit = params
    return n_0 , lambda_fit


# #TODO
# def fit_exp_pyro(x_values, y_values, n0_name, lambda_name):
    
#     """ This function will caluculate the n0 and Lambda coefficients of an exponental decaying function:
        
#        f(x)= n0*exp(-x/Lambda)
       
#        ! Only jax.numpy objects are allowed as inputs, usage of numpy array will result in errors !
#        Please convert your numpy arrays in jax.numpy array (x = jnp.array(x))
       
#         INPUTS : x_values, abssisca values of the function you wish to fit, dtype: jaxlib.xla_extension.Array (shape (1,))
#                  y_values, values you wish to fit, dtype : jaxlib.xla_extension.Array (shape (1,))
#                  n0_name, name of the n0 coefficient, dtype : str
#                  lambda_name, name of the n0 coefficient, dtype : str
                 
#         OUTPUTS : n0, calculated as mean of estimated samples, dtype : jaxlib.xla_extension.Array (value) 
#                   Lambda, calculated as mean of estimated samples, dtype : jaxlib.xla_extension.Array (value) """
    
#     """ Define your model for the MCMC run"""
    
#     print("start")
    
#     def model(obs):
        
#         n0_estimated_mcmc = numpyro.sample(n0_name, dist.Uniform(0, 10)) # uniform a-priori distribution [0, 10]
#         Lambda_estimated_mcmc = numpyro.sample(lambda_name, dist.Uniform(10, 500)) # uniform a-priori distribution [10, 500]
#         y_fit_numpyro = exp_law (x_values, n0_estimated_mcmc, Lambda_estimated_mcmc) # calculation of y-values with estimated parameters
#         return numpyro.sample('obs', dist.Normal(y_fit_numpyro, 0.5), obs=obs) # returning all samples, with some statistical tests
    
#     rng_key = random.PRNGKey(0) # random start
#     rng_key, rng_key_ = random.split(rng_key)
#     kernel = NUTS(model) # use of No-UTuRNS sampling strategy (Hoffman & Gilber, 2011)
#     mcmc = MCMC(kernel, num_warmup=150, num_samples=500) # set your burning and samples (30% of warmup recommended)
#     mcmc.run(rng_key, obs = y_values) # runing the MCMC 
#     mcmc.print_summary() # watch for the std (standard dev.) of your samples and r_hat = 1 (Gilman-Rubin criterion)
#     posterior_samples = mcmc.get_samples() # store your samples in a dictionnary
    
#     return mean(posterior_samples[n0_name]), mean(posterior_samples[lambda_name]) # mean of estimated parameters


def f (beta, theta, phi):
    
    """ This function is used in scrock, to scale the incidence of cosmic rays, depending on fault dip
        INPUTS : beta, fault dip, radian, dtype : float
                 theta, phi, frame defined with a meshgrid
        OUTPUT : d, incidence of cosmic rays, radian; dtype : float"""
        
    num=1
    den=sin(theta)*cos(beta)-sin(beta)*cos(theta)*sin(phi)
    den[np.where(den==0)]=1/100000
    d=np.abs(num/den)
    return d

def scrock(h, Lambda, beta, rho_rock):
    
    """ Calculates the production in bedrock depending on incidence of cosmic rays. Takes in acount
    the effect of the air.
    INPUTS : h, height, cm, dtype : float
             Lambda, true attenuation of fast neutrons (208), dtype : float
             beta, fault dip, degrees, dtype : float"""
    
    if h==0:
        h=1/10000 # to avoid nan
    else :
        h=h
    
    m=2.3
    beta=beta*pi/180 # converting beta in rad
    
    theta= np.arange(0, 91)*pi/180
    phi= np.arange(0, 181)*pi/180
    
    THETA, PHI=np.meshgrid(theta, phi)
    
    dphi=pi/180
    dtheta=dphi
    
    dv=f(beta, THETA, PHI+pi)
    dv=exp(-h*rho_rock*dv/Lambda)
    dv=dv*(sin(THETA)**m)*cos(THETA)*dphi*dtheta
    dv=np.sum(dv)
    
    
    B=arctan(tan(beta)*sin(PHI))
    da=f(beta, THETA, PHI)
    da=exp(-h*rho_rock*da/Lambda)
    da=da[np.where(THETA>B)]*(sin(THETA[np.where(THETA>B)])**m)*cos(THETA[np.where(THETA>B)])*dphi*dtheta
    da=np.sum(da)
    
    
    
    S_air=da*(m+1)/(2*pi)
    
    S_rock=dv*(m+1)/(2*pi)
    Sr=S_air+S_rock
    return Sr


def f2 (gamma, beta, theta, phi):
    
    """ This function is used in scsurf, to scale the incidence of cosmic rays, depending on fault dip and
    angle of eroded scarp
        INPUTS : beta, fault dip, radian, dtype : float
                 theta, phi, frame defined with a meshgrid
        OUTPUT : d, incidence of cosmic rays, radian; dtype : float""" 
        
    if beta-gamma==0:
        gamma=gamma-0.0001
    
    else: 
        gamma=gamma
    num=sin(beta-gamma)
    den=sin(theta)*cos(gamma)-sin(gamma)*cos(theta)*sin(phi)
    den[np.where(den==0)]=1/100000
    d=np.abs(num/den)
    return d

def scsurf(Z, H, Lambda, beta, gamma, rho_rock):
    
    """ Scaling factor for exhumated samples, from Schlagenhauf 2010 

    INPUT : Z (cm) depth of sample, measured on the scarp, type: float
            H (cm) : height of the scarp, type :float
            Lambda (g.cm-2) : true attenuation length (208 g.cm-2 for neutrons), type : float
            beta (deg) : scarp dip, type : float
            gamma (deg) : dip of the upper eroded part, type : float
            rho_rock (g.cm-3) : density of the rock, type : float

    OUTPUT : S_S, Scaling factor for exhumated rock, type : float """
    
    
    if H-Z==0:
        hz=1/100000 # to avoid nan
    else :
        hz=H-Z
    
    
    m=2.3
    beta=beta*pi/180 # converting beta in rad
    gamma=gamma*pi/180
    
    theta= np.arange(0, 91)*pi/180
    phi= np.arange(0, 181)*pi/180
    
    THETA, PHI=np.meshgrid(theta, phi)
    
    dphi=pi/180
    dtheta=dphi
    
    dv=(sin(THETA)**m)*cos(THETA)*dphi*dtheta
    dv=np.sum(dv)
    
    B=arctan(tan(beta)*sin(PHI))
    da=(sin(THETA[np.where(THETA>B)])**m)*cos(THETA[np.where(THETA>B)])*dphi*dtheta
    da=np.sum(da)
    
    C=arctan(tan(gamma)*sin(PHI))
    dr=f2(gamma, beta, THETA, PHI)
    dr=exp(-hz*rho_rock*(dr/Lambda))
    dr=dr[np.where((THETA<B) & (THETA>C))]*(sin(THETA[np.where((THETA<B) & (THETA>C))])**m)*cos(THETA[np.where((THETA<B) & (THETA>C))])
    dr=dr*dphi*dtheta
    dr=np.sum(dr)
   
    S_air=(da+dv)*(m+1)/(2*pi)
    S_rock=dr*(m+1)/(2*pi)
    Ss=S_air+S_rock
    return Ss


def f3 (gamma, beta, theta, phi):
    
    """ This function is used in scrock, to scale the incidence of cosmic rays, depending on fault dip and 
    eroded scarp
        INPUTS : gamma, erroded scarp dip, radian, dtype : float
                 beta, fault dip, radian, dtype : float
                 theta, phi, frame defined with a meshgrid
        OUTPUT : d, incidence of cosmic rays, radian; dtype : float"""
   
   
    num=sin(beta-gamma)
    den=sin(theta)*cos(gamma)-sin(gamma)*cos(theta)*sin(phi)
    den[np.where(den==0)]=1/100000  # Avoid nans
    d=np.abs(num/den)
    return d

def scdepth(Z, H, Lambda, alpha, beta, gamma, rho_rock, rho_coll):
    """ Scaling factor for buried samples, from Schlagenhauf 2010 

    INPUT : Z (cm) depth of sample, measured on the scarp, type: float
            H (cm) : height of the scarp, type :float
            Lambda (g.cm-2) : true attenuation length (208 g.cm-2 for neutrons), type : float
            beta (deg) : scarp dip, type : float
            gamma (deg) : dip of the upper eroded part, type : float
            rho_rock (g.cm-3) : density of the rock, type : float

    OUTPUT : S_S, Scaling factor for exhumated rock, type : float """
   
    
    m=2.3
    alpha=alpha*pi/180
    beta=beta*pi/180 # converting beta in rad
    gamma=gamma*pi/180
    
    theta= np.arange(0, 91)*pi/180
    phi= np.arange(0, 181)*pi/180
    
    THETA, PHI=np.meshgrid(theta, phi)
    
    dphi=pi/180
    dtheta=dphi
    
    dv=f3(alpha, beta, THETA, PHI+pi)
    dv=exp(Z*rho_coll*dv/Lambda)
    dv=dv*(sin(THETA)**m)*cos(THETA)*dphi*dtheta
    dv=np.sum(dv)
    
    B=arctan(tan(beta)*sin(PHI))
    da=f3(alpha, beta, THETA, PHI)
    da=exp(Z*rho_coll*da/Lambda)
    da=da[np.where(THETA>B)]*(sin(THETA[np.where(THETA>B)])**m)*cos(THETA[np.where(THETA>B)])*dphi*dtheta
    da=np.sum(da)
    
    
    C=arctan(tan(gamma)*sin(PHI))
    dr=f3(gamma, beta, THETA, PHI)
    dr=exp(-(H-Z)*rho_rock*dr/Lambda)
    dr=dr[np.where((THETA<B) & (THETA>C))]*(sin(THETA[np.where((THETA<B) & (THETA>C))])**m)*cos(THETA[np.where((THETA<B) & (THETA>C))])
    dr=dr*dphi*dtheta
    dr=np.sum(dr)
    
    S_air=(da+dv)*(m+1)/(2*pi)
    S_rock=dr*(m+1)/(2*pi)
    Sd=S_air+S_rock
    return Sd