#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:13:00 2023

@author: llinares
"""
import numpy as np

""" Useful constants (from Shlengenhauf et al. 2010 (https://doi.org/10.1111/j.1365-246X.2010.04622.x)) 
    Modify at your own risks""" 

constants={}

constants['Psi_Cl36_Ca_0'] = 42.2
constants['Lambda'] = 208 # True attenuation length for fast neutron (g.cm-2)
constants['lambda36'] = 2.303*1e-6 # Radioactive decay constant for 36Cl (a-1)
constants['Avogadro'] = 6.02214 * 1e23  # Avogadro Number
constants['Lambda_mu'] = 1500  # g.cm-2

constants['A_a'] = 14.5  # Average atomic weight of air
constants['Sigma_eth_a'] = 0.0548  # Macroscopic absorption and moderation x-section in atm. (cm2 g-1) - Constant (Chloe)
constants['D_th_a'] = 0.9260472  # Thermal neutron diffusion coeff in atm. (g*cm-2) - Constant (Chloe)
constants['Sigma_sc_a'] = 0.3773  # Macroscopic neutron scaterring cross section of atmosphere (cm2.g-1) - Constant (Chloe)
constants['Sigma_th_a'] = 0.060241  # Constant from Chloe - macroscopic thermal neutron cross section of atm (cm2 g-1)

constants['f_n_Ca'] = 0.045  # +/- 0.005 Heisinger et al. (2002)
constants['f_n_K'] = 0.035  # +/- 0.005 Heisinger et al. (2002)
constants['f_i_Ca'] = 0.969  # Fabryka-Martin (1988)
constants['f_i_K'] = 0.933  # Fabryka-Martin (1988)
constants['f_d_Ca'] = 0.864  # Fabryka-Martin (1988)
constants['f_d_K'] = 0.83  # Fabryka-Martin (1988)
constants['Psi_mu_0'] = 190  # 190 slow negative muon stopping rate at land surface (muon/g/an), Heisinger et al. (2002)
# f_n_K = 0.02  # Fabryka-Martin (1988)
# f_n_Ca = 0.062  # Fabryka-Martin (1988)

constants['p_E_th_a'] = 0.56  # Resonance escape probability of the atmosphere - Constant (Chloe)

# Psi_Cl36_K_0 = 162  # Spallation production rate at surface of 39K
constants['Psi_Cl36_K_0'] = 4
# (at of Cl36 /g of K per yr) [162 ? 24 Evans et al. 1997]
constants['phi_mu_f_0'] = 7.9 * 1e5  # Fast muon flux at land surface, sea level, high latitude, Gosse & Phillips, 2001 (? cm-2 yr-1)
constants['Psi_Cl36_Fe_0'] = 1.9  # Spallation production rate at surface of Fe
# (at of Cl36 /g of Fe per yr) [1.9 ? 0.2 Stone 2005]
constants['Psi_Cl36_Ti_0'] = 13  # Spallation production rate at surface of Ti
# (at of Cl36 /g of Ti per yr) [13 ? 3 Fink et al. 2000]
constants['P_f_0'] = 626  # Production rate of epithermal neutrons from fast neutrons in atm at land/atm interface (n cm-2 yr-1), Gosse & Philipps, 2001.
constants['R_th_a'] = 1
constants['R_eth_a'] = 1

# Shielding factors
constants['S_L_th'] = 1  # diffusion out of objects (poorly constrained)
constants['S_L_eth'] = 1  # diffusion out of objects (poorly constrained)


# CHEMICAL ELEMENTS
    #
    # from 1 to 10  : As Ba Be Bi Cd Ce Co Cr Cs Cu
    # from 11 to 20 : Dy Er Eu Ga Gd Ge Hf Ho In La
    # from 21 to 30 : Lu Mo Nb Nd Ni Pb Pr Rb Sb Sm
    # from 31 to 40 : Sn Sr Ta Tb Th Tm U  V  W  Y
    # from 41 to 50 : Yb Zn Zr SiO2(Si) Al2O3(Al) Fe2O3(Fe) MnO(Mn) MgO(Mg) CaO(Ca) Na2O(Na)
    # from 51 to 61 : K2O(K) TiO2(Ti) P2O5(P) B Li H2Otot(H) Stot(S) CO2tot(C) O_rock O_water CltotalAMS
    # 62 : [Ca] in ppm from ICP
    

# Num_k = Atomic number of element k
constants['Num_k'] = np.array(
    [33, 56, 4, 83, 48, 58, 27, 24, 55, 29, 66, 68, 63, 31, 64, 32, 72, 67, 49, 57, 71, 42, 41, 60, 28, 82, 59, 37,
     51, 62, 50, 38, 73, 65, 90, 69, 92, 23, 74, 39, 70, 30, 40, 14, 13, 26, 25, 12, 20, 11, 19, 22, 15, 5, 3, 1,
     16, 6, 8, 8, 17])

# Xi_k = average log-decrement of energy loss per collision for element k
constants['Xi_k'] = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0.038, 0, 0, 0, 0, 0, 0, 0.013, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.013, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.07, 0.072, 0.035, 0.036, 0.08, 0.049, 0.084, 0.05, 0.041, 0, 0.174, 0.264, 1,
     0, 0.158, 0.12, 0.12, 0.055])

# sigma_sc_k = neutron scattering x-section of element k (barns)
constants['sigma_sc_k'] = np.array(
    [0, 0, 0, 0, 0, 0, 0, 3.38, 0, 0, 0, 0, 0, 0, 172, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 2.04, 1.41, 11.35, 2.2, 3.42, 2.93, 3.025, 2.04, 4.09, 5, 4.27, 0.95, 20.5, 0, 4.74,
     3.76, 3.76, 15.8])

# sigma_th_k = thermal neutron absorbtion x-section of element k (barns)
constants['sigma_th_k'] = np.array(
    [0, 0, 0, 0, 0, 0, 0, 3.1, 0, 0, 0, 0, 0, 0, 41560, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9640, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0.17, 0.23, 2.56, 13.3, 0.063, 0.43, 0.53, 2.15, 6.1, 0.2, 767, 70.5, 0.33, 0,
     0.0034, 0.0002, 0.0002, 33.5])

# I_a_k = dilute resonance integral for absorption of epithermal neutrons by element k (barns)
constants['I_a_k'] = np.array(
    [0, 0, 0, 0, 0, 0, 0, 1.6, 0, 0, 0, 0, 0, 0, 390, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1400, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0.127, 0.17, 1.39, 14, 0.038, 0.235, 0.311, 1, 3.1, 0, 1722, 0, 0, 0, 0.0016, 0.0004,
     0.0004, 13.7])

# f_d_k = proportion of muons stopped in element k that are captured by the nucleus
constants['f_d_k'] = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0.671, 0.582, 0.906, 0, 0.538, 0.864, 0.432, 0.83, 0, 0, 0, 0, 0, 0, 0.09, 0.223, 0.223, 0])

# Y_n = average neutron yield per captured muon
constants['Y_n'] = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0.86, 1.26, 1.125, 0, 0.6, 0.75, 1, 1.25, 0, 0, 0, 0, 0, 0, 0.76, 0.8, 0.8, 0])

# S_i = mass stopping power (MeV/(g.cm-2))
constants['S_i'] = np.array(
    [0, 0, 0.000529, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0.000454, 0.000444, 0.000351, 0, 0.000461, 0.000428, 0.000456, 0.000414, 0.000375,
     0.000433, 0.000527, 0.000548, 0, 0.000439, 0.000561, 0.000527, 0.000527, 0])

# Y_U_n = neutron yield (n/an/g/ppm de U)
constants['Y_U_n'] = np.array(
    [0, 0, 265, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0.69, 5.1, 0.19, 0, 5.8, 0, 14.5, 0.45, 0, 0, 62.3, 21.1, 0, 0, 0.45, 0.23, 0.23, 0])

# Y_TH_n = neutron yield (n/an/g/ppm de Th)
constants['Y_Th_n'] = np.array(
    [0, 0, 91.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 0, 0, 0.335, 2.6, 0.205, 0, 2.6, 0, 6.8, 0.305, 0, 0, 19.2, 9.6, 0, 0, 0.18, 0.079, 0.079, 0])

constants['A_k'] = np.array(
        [74.9, 137.33, 9.01218, 209.0, 112.4, 140.1, 58.9332, 51.996, 132.9054, 63.5, 162.5, 167.3, 152.0, 69.7, 157.25,
         72.6, 178.5, 164.9, 114.8, 138.9, 175.0, 95.94, 92.9, 144.2, 58.7, 207.2, 140.9, 85.4678, 121.8, 150.4, 118.7,
         87.62, 180.9, 158.9, 232, 168.9, 238.029, 50.9, 183.8, 88.9, 173.0, 65.4, 91.22, 28.0855, 26.98154, 55.847,
         54.938, 24.305, 40.08, 22.98977, 39.0983, 47.9, 30.97376, 10.81, 6.941, 1.0079, 32.06, 12.011, 15.9994,
         15.9994, 35.453])