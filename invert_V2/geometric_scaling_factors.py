#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:28:56 2023

@author: llinares
"""
import numpy as np
from numpy import pi, sin
from util import geometric_scaling_functions as gscale

""" Useful functions for the scaling of chlorine 36 production due to cosmic rays
    Modified for python from Shlengenhauf et al. 2010 (https://doi.org/10.1111/j.1365-246X.2010.04622.x)
    and Tesson & Benedetti 2019 (link)
    
    Changes : exponential fit done prior to forward function  """
    


def neutron_scaling (params, constants, N_eq):
    
    """ This function will approximate scaling factors for chlorine 36 production related
        to the geometry and the height of the preserved scarp 
        
        INPUTS : params, parameters of the site (see parameters.py)
                 constants, dictionary containing all useful constants (see constants.py)
                 N_eq, number of earthquakes, dtype : integer 
                 
        OUTPUTS : scaling_depth_rock, dictionary containing scaling factors for buried samples
                  scaling_depth_coll, dictionary containing scaling factors for the colluvium
                  scaling_surf_rock, dictionary containing scaling factors for the exhumed samples """

    
    Hfinal = params.Hfinal # preserved scarp height(cm)
    alpha = params.alpha # colluium dip (degrees)
    beta = params.beta # preserved scarp dip (degrees)
    gamma = params.gamma # eroded scarp dip (degrees)
    rho_rock = params.rho_rock # rock mean density
    rho_coll = params.rho_coll # colluvium mean density
    Lambda = constants['Lambda'] # True attenuation length for fast neutron (g.cm-2)
    
    
    slip = np.zeros((N_eq))+(Hfinal/N_eq) # define slip vector
    # slip = np.array([0, 1200, 600, 200, 400]) # from scenario MA3
    
    """ SURFACE SCALING """
    R = np.sum(slip)  # total cumulative slip
    Rc = np.cumsum(slip)   
    Rc = np.hstack((0, Rc)) # slip added up after each earthquake
    Hinit = params.Hfinal - R  # initial height of the scarp during pre-exposure

    
    Zs = np.arange(0, R)  # initialization of Zs  one calculation point every cm  
    S_S = np.zeros((len(Zs)))  # initialization of S_S (Surface Scaling)
    #
    for i in range (0,len(Zs)):    # loop on Zs
        S_S[i] = gscale.scsurf(Zs[i], Hfinal, Lambda, beta, gamma, rho_rock) 
        # S_S[i] = a 
 
    """ DEPTH SCALING FOR NEUTRONS IN ROCK"""
    
    so_f_diseg = np.zeros(N_eq)  # initialization of so_f_d_iseg
    Lambda_f_diseg = np.zeros(N_eq)  # initialization of Lambda_f_d_iseg
    
    x=np.zeros((N_eq, len(np.arange(0, R+1, 10))))
    y=np.zeros((N_eq, len(np.arange(0, R+1, 10))))
    
    for si in range (0, N_eq):     # earthquake loop
        Hiseg = Hinit + Rc[si]  # Height of exhumed scarp after each earthquake
        Ziseg = np.arange(0, R+1, 10)  # one calculation point every 10 cm is sufficient  
        Ziseg = -Ziseg  # negative because at depth
        S_D_iseg = np.zeros(len(Ziseg))  # initialization of S_D_iseg
        for i in range (0,len(Ziseg)):     # loop on z
            S_D_iseg[i] = gscale.scdepth(Ziseg[i], Hiseg, Lambda, alpha, beta, gamma, rho_rock, rho_coll) 
            # S_D_iseg[i] = a 
             
        
        dd, ee = gscale.fitexp(-Ziseg*rho_coll, S_D_iseg)  
        so_f_diseg[si] = dd  # constant so
        Lambda_f_diseg[si] = ee  # attenuation length for neutron in direction z 
        x[si,:] = -Ziseg*rho_coll
        y[si,:] = S_D_iseg

    # attenuation length perpendicular to colluvium surface after each
    # earthquake (with H increasing after each earthquake):
    Lambda_f_diseg = Lambda_f_diseg*sin((beta - alpha)*pi/180)
    
    scaling_depth_rock={} # dict initialization
    scaling_depth_rock['s_diseg'] = so_f_diseg
    scaling_depth_rock['lambda_diseg'] = Lambda_f_diseg
    scaling_depth_rock['x_iseg'] = x # useful to check the succes of exp_fit
    scaling_depth_rock['y_iseg'] = y # useful to check the succes of exp_fit
    
    """ DEPTH SCALING FOR NEUTRONS IN COLLUVIUM """
    
    # For beta infinite plane (used in B2 and C6):
    Zbeta_inf = np.arange(0, 1001, 10) 
    Zbeta_inf = -Zbeta_inf  # initialization
    S_D_beta_inf = np.zeros(len(Zbeta_inf))
    
    # ??? POURQUOI 2000 dans sc_depth (présent dans Shlagenhauf/Tesson) ???
    for i in range (0,len(Zbeta_inf)):     # loop on z
            S_D_beta_inf[i] = gscale.scdepth(Zbeta_inf[i], 2000, Lambda, alpha, beta, gamma, rho_rock, rho_coll) 
            # S_D_beta_inf[iv] = a 
    
    
    so_f_beta_inf, Lambda_f_beta_inf = gscale.fitexp(-Zbeta_inf*rho_coll, S_D_beta_inf)  # fit by fitexp.m
    Lambda_f_beta_inf = Lambda_f_beta_inf*sin((beta - alpha)*pi/180)  # attenuation perp. to colluvium surface
    
    scaling_depth_coll={} # dict initilization 
    scaling_depth_coll['s_beta'] = so_f_beta_inf
    scaling_depth_coll['lambda_beta'] = Lambda_f_beta_inf 
    scaling_depth_coll['x'] = -Zbeta_inf*rho_coll # useful to check the succes of exp_fit
    scaling_depth_coll['y'] = S_D_beta_inf # useful to check the succes of exp_fit
    
    """ ROCK SCALING FOR NEUTRONS """
    
    e = np.arange(0, 101)  # e is in cm and perpendicular to scarp surface
    Se = np.zeros(len(e))  # initialization of scaling Se
    for i in range (0,len(e)):         # Loop on e
    	Se[i] = gscale.scrock(e[i],Lambda,beta,rho_rock) 

    so_f_e, Lambda_f_e = gscale.fitexp(e*rho_rock, Se)  
    
    scaling_surf_rock = {} # dict initilization 
    scaling_surf_rock['x'] = e*rho_rock # useful to check the succes of exp_fit
    scaling_surf_rock['y'] = Se # useful to check the succes of exp_fit
    scaling_surf_rock['s_e'] = so_f_e
    scaling_surf_rock['lambda_e'] = Lambda_f_e
    
    return scaling_depth_rock, scaling_depth_coll, scaling_surf_rock, S_S
