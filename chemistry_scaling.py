# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 20:08:04 2022

@author: Maureen
"""

import numpy as np
import jax.numpy as jnp
from jax.numpy import exp, sqrt
from constants import constants
import parameters 

""" Constants """
# Global constants
Lambda_mu = constants['Lambda_mu'] # attenuation length for muons g.cm-2
Avogadro = constants['Avogadro']
lambda36 = constants['lambda36']

# Chemistry elements
A_k = constants['A_k'] # A_k = atomic mass of element k (g.mol-1)
Num_k = constants['Num_k'] # Atomic number of element k
Xi_k = constants['Xi_k'] # average log-decrement of energy loss per collision for element k
sigma_sc_k = constants['sigma_sc_k'] # neutron scattering x-section of element k (barns)
sigma_th_k = constants['sigma_th_k'] # thermal neutron absorbtion x-section of element k (barns)
I_a_k = constants['I_a_k'] # dilute resonance integral for absorption of epithermal neutrons by element k (barns)
f_d_k = constants['f_d_k'] # proportion of muons stopped in element k that are captured by the nucleus
Y_n = constants['Y_n']  # average neutron yield per captured muon
S_i = constants['S_i'] # S_i = mass stopping power (MeV/(g.cm-2))
Y_U_n = constants['Y_U_n'] # Y_U_n = neutron yield (n/an/g/ppm de U)
Y_Th_n = constants['Y_Th_n'] # Y_TH_n = neutron yield (n/an/g/ppm de Th)

# Spallation
Psi_Cl36_K_0 = constants['Psi_Cl36_K_0'] # Spallation production rate at surface of 39K
Psi_Cl36_Fe_0 = constants['Psi_Cl36_Fe_0']  # Spallation production rate at surface of Fe
Psi_Cl36_Ti_0 = constants['Psi_Cl36_Ti_0']  # Spallation production rate at surface of Ti

# Slow negative muons (Ca and K)
Psi_Cl36_Ca_0 = constants['Psi_Cl36_Ca_0']
f_n_Ca = constants['f_n_Ca']   # +/- 0.005 Heisinger et al. (2002) 
f_n_K = constants['f_n_K']  # +/- 0.005 Heisinger et al. (2002)
f_i_Ca = constants['f_i_Ca']  # Fabryka-Martin (1988)
f_i_K = constants['f_i_K'] # Fabryka-Martin (1988)
f_d_Ca = constants['f_d_Ca']  # Fabryka-Martin (1988)
f_d_K = constants['f_d_K']  # Fabryka-Martin (1988)
Psi_mu_0 = constants['Psi_mu_0']  # 190 slow negative muon stopping rate at land surface (muon/g/an), Heisinger et al. (2002)
A_a = constants['A_a']

# Epithermal neutrons
R_eth_a = constants['R_eth_a']
P_f_0 = constants['P_f_0']
Sigma_eth_a = constants['Sigma_eth_a']  # Macroscopic absorption and moderation x-section in atm. (cm2 g-1) - Constant (Chloe)
D_th_a = constants['D_th_a'] # Thermal neutron diffusion coeff in atm. (g*cm-2) - Constant (Chloe)
Sigma_sc_a = constants['Sigma_sc_a']  # Macroscopic neutron scaterring cross section of atmosphere (cm2.g-1) - Constant (Chloe)
phi_mu_f_0 = constants['phi_mu_f_0']# Fast muon flux at land surface, sea level, high latitude, Gosse & Phillips, 2001 (? cm-2 yr-1)
p_E_th_a = constants['p_E_th_a']   # Resonance escape probability of the atmosphere - Constant (Chloe)
Sigma_th_a = constants['Sigma_th_a']  # Constant from Chloe - macroscopic thermal neutron cross section of atm (cm2 g-1)
R_th_a = constants['R_th_a']

# Shielding factors
S_L_th = constants['S_L_th']  # diffusion out of objects (poorly constrained)
S_L_eth = constants['S_L_eth']  # diffusion out of objects (poorly constrained)

""" Parameters """
param=parameters.param()
rho_rock=param.rho_rock
rho_coll=param.rho_coll

def convert_2_oxyde(ppm, constants):
    """ This function converts input chemistry data into oxyde 
        INPUTS : ppm, chemistry, dtype: array, shape (1, 66)
                 constants, dictionary of all constants
        OUTPUT : ppm with all elements converted in oxyde """
    
    ppm[43] = ppm[43] * A_k[43] / (A_k[43] + 2 * A_k[58])  # Si in percent
    ppm[44] = ppm[44] * 2 * A_k[44] / (2 * A_k[44] + 3 * A_k[58])  # Al in percent
    ppm[45] = ppm[45] * 2 * A_k[45] / (2 * A_k[45] + 3 * A_k[58])  # Fe in percent
    ppm[46] = ppm[46] * A_k[46] / (A_k[46] + A_k[58])  # Mn in percent
    ppm[47] = ppm[47] * A_k[47] / (A_k[47] + A_k[58])  # Mg in percent
    ppm[48] = ppm[48] * A_k[48] / (A_k[48] + A_k[58])  # Ca in percent
    ppm[49] = ppm[49] * 2 * A_k[49] / (2 * A_k[49] + A_k[58])  # Na in percent
    ppm[50] = ppm[50] * 2 * A_k[50] / (2 * A_k[50] + A_k[58])  # K in percent
    ppm[51] = ppm[51] * A_k[51] / (A_k[51] + 2 * A_k[58])  # Ti in percent
    ppm[52] = ppm[52] * 2 * A_k[52] / (2 * A_k[53] + 5 * A_k[58])  # P in percent
    ppm[55] = ppm[55] * 2 * A_k[55] / (2 * A_k[55] + A_k[58])  # H water in percent
    O_water = ppm[55] - ppm[55]  # O_water in percent
    ppm[57] = ppm[57] * A_k[57] / (A_k[57] + 2 * A_k[58])  # C in percent

    ppm[58] = (np.sum(ppm[43:53]) + ppm[57]) - (np.sum(ppm[43:53]) + ppm[57])  # O rock in percent
    ppm[59] = O_water
    ppm[43:53] = ppm[43:53] * 1e4  # in ppm
    ppm[55] = ppm[55] * 1e4  # in ppm
    ppm[57] = ppm[57] * 1e4  # in ppm
    ppm[58] = ppm[58] * 1e4  # in ppm
    ppm[59] = ppm[59] * 1e4  # in ppm
    
    return ppm

    

def clcoll(coll, sample, Lambda_f, so_f, EL_f, EL_mu):
    
    """ This function allows you to calculate the chlorine 36 production under the colluvium
        INPUTS : coll, chemistry of the colluvium, numpy array (one dimentionnal)
                 sample, chemistry of the collected sample, numpy array (one dimentionnal)
                 Lambda_f, calculated with fitexp, float
                 so_f, calculatated with fitexp, float
                 EL_f, float
                 EL_mu, float
    
        OUTPUTS : P_cosmo, float """
    
    chimie = sample[0:62]
    z = sample[62]  # (g.cm-2)
    thick = sample[63] # (g.cm-2)
    th2 = thick/2 
    

    """ COVERSION IN OXYDE : ROCK"""

    # ppm = chimie.copy() 
    # ppm = ppm.at[43].set(chimie[43]*A_k[43]/(A_k[43] + 2*A_k[58]))  # Si in percent
    # ppm = ppm.at[44].set(chimie[44]*2*A_k[44]/(2*A_k[44] + 3*A_k[58]))  # Al in percent
    # ppm = ppm.at[45].set(chimie[45]*2*A_k[45]/(2*A_k[45] + 3*A_k[58]))  # Fe in percent
    # ppm = ppm.at[46].set(chimie[46]*A_k[46]/(A_k[46] + A_k[58]))  # Mn in percent
    # ppm = ppm.at[47].set(chimie[47]*A_k[47]/(A_k[47] + A_k[58]))  # Mg in percent
    # ppm = ppm.at[48].set(chimie[48]*A_k[48]/(A_k[48] + A_k[58]))  # Ca in percent
    # ppm = ppm.at[49].set(chimie[49]*2*A_k[49]/(2*A_k[49] + A_k[58]))  # Na in percent
    # ppm = ppm.at[50].set(chimie[50]*2*A_k[50]/(2*A_k[50] + A_k[58]))  # K in percent
    # ppm = ppm.at[51].set(chimie[51]*A_k[51]/(A_k[51] + 2*A_k[58]))  # Ti in percent
    # ppm = ppm.at[52].set(chimie[52]*2*A_k[52]/(2*A_k[52] + 5*A_k[58]))  # P in percent
    # ppm = ppm.at[55].set(chimie[55]*2*A_k[55]/(2*A_k[55] + A_k[58]))  # H water in percent
    # O_water = chimie[55] - ppm[55]  # O_water in percent
    # ppm = ppm.at[57].set(chimie[57]*A_k[57]/(A_k[57] + 2*A_k[58]))  # C in percent

    # ppm = ppm.at[58].set(jnp.sum(chimie[43:53])+chimie[57]) - (np.sum(ppm[43:53]) + ppm[57])  # O rock in percent
    # ppm = ppm.at[59].set(O_water) 
    # ppm = ppm.at[43:53].set(ppm[43:53]*1e4 ) # in ppm
    # ppm = ppm.at[55].set(ppm[55]*1e4)  # in ppm
    # ppm = ppm.at[57].set(ppm[57]*1e4)  # in ppm
    # ppm = ppm.at[58].set(ppm[58]*1e4)  # in ppm
    # ppm = ppm.at[59].set(ppm[59]*1e4)  # in ppm
    ppm = convert_2_oxyde(chimie, constants)
    
    """ CONVERSION IN OXYDE : COLLUVIUM """

    # ppmc = coll.copy() 
    # ppmc[43] = coll[43]*A_k[43]/(A_k[43] + 2*A_k[58])  # Si in percent
    # ppmc[44] = coll[44]*2*A_k[44]/(2*A_k[44] + 3*A_k[58])  # Al in percent
    # ppmc[45] = coll[45]*2*A_k[45]/(2*A_k[45] + 3*A_k[58])  # Fe in percent
    # ppmc[46] = coll[46]*A_k[46]/(A_k[46] + A_k[58])  # Mn in percent
    # ppmc[47] = coll[47]*A_k[47]/(A_k[47] + A_k[58])  # Mg in percent
    # ppmc[48] = coll[48]*A_k[48]/(A_k[48] + A_k[58])  # Ca in percent
    # ppmc[49] = coll[49]*2*A_k[49]/(2*A_k[49] + A_k[58])  # Na in percent
    # ppmc[50] = coll[50]*2*A_k[50]/(2*A_k[50] + A_k[58])  # K in percent
    # ppmc[51] = coll[51]*A_k[51]/(A_k[51] + 2*A_k[58])  # Ti in percent
    # ppmc[52] = coll[52]*2*A_k[52]/(2*A_k[52] + 5*A_k[58])  # P in percent
    # ppmc[55] = coll[55]*2*A_k[55]/(2*A_k[55] + A_k[58])  # H water in percent
    # O_water2 = coll[55] - ppmc[55]  # O_water in percent
    # ppmc[57] = coll[57]*A_k[57]/(A_k[57] + 2*A_k[58])  # C in percent

    # ppmc[58] = (np.sum(coll[43:53]) + coll[57]) - (np.sum(ppmc[43:53]) + ppmc[57])  # O rock in percent
    # ppmc[59] = O_water2
    # ppmc[43:53] = ppmc[43:53]*1e4  # in ppm
    # ppmc[55] = ppmc[55]*1e4  # in ppm
    # ppmc[57] = ppmc[57]*1e4  # in ppm
    # ppmc[58] = ppmc[58]*1e4  # in ppm
    # ppmc[59] = ppmc[59]*1e4  # in ppm
    # ppmc[61]=ppmc[48] # because [Ca]_coll [ppm] not determined by ICP
    #-----------------------------------------------------------------
    ppmc = convert_2_oxyde(coll, constants)

    N_k = (ppm[0:61]/A_k)*Avogadro*1e-6  # Concentrations in atom/g ROCK
    N_kc = (ppmc[0:61]/A_k)*Avogadro*1e-6  # Concentrations in atom/g COLLUVIUM
    N_k[55] = N_k[55]/rho_rock  # divided by bulk-rock density according to CHLOE for H
    N_kc[55] = N_kc[55]/rho_coll  # divided by bulk-rock density according to CHLOE for H

    
    thick = sample[63]

    """ PRODUCTION RATE : SPALLATION """

    C_Ca = ppm[61]*1e-6  # Mass concentration of Ca (g of Ca per g of rock) # from ICP
    P_sp_Ca = Psi_Cl36_Ca_0*C_Ca  # result unscaled 36Cl production by spallation of 40Ca (atoms 36Cl g-1 yr-1)

    C_K = ppm[50]*1e-6  # Mass concentration of K (g of K per g of rock)
    P_sp_K = Psi_Cl36_K_0*C_K  # result unscaled 36Cl production by spallation of 39K (atoms 36Cl g-1 yr-1)

    C_Ti = ppm[51]*1e-6  # Mass concentration of Ti (g of Ti per g of rock)
    P_sp_Ti = Psi_Cl36_Ti_0*C_Ti  # result unscaled 36Cl production by spallation of Ti (atoms 36Cl g-1 yr-1)

    C_Fe = ppm[45]*1e-6  # Mass concentration of Fe (g of Fe per g of rock)
    P_sp_Fe = Psi_Cl36_Fe_0*C_Fe  # result unscaled 36Cl production by spallation of Fe (atoms 36Cl g-1 yr-1)
   
    P_sp = (P_sp_Ca + P_sp_K + P_sp_Ti + P_sp_Fe)*exp(-z/Lambda_f)  # Unscaled Spallation production rate (atoms 36Cl g-1 yr-1)


    """  Direct capture of slow negative muons by target elements Ca and K """
    
    f_c_Ca = (Num_k[48]*ppm[61]*1e-6/A_k[48])/(np.sum(Num_k*ppmc[0:61]/A_k)*1e-6)  # for Ca (ICP)
    f_c_K = (Num_k[50]*ppm[50]*1e-6/A_k[50])/(np.sum(Num_k*ppmc[0:61]/A_k)*1e-6)  # for K

    Y_Sigma_Ca = f_c_Ca*f_i_Ca*f_d_Ca*f_n_Ca  # 36Cl production per stopped muon 
    # Y_Sigma_Ca DEPENDS ON CHEMICAL COMPOSITION
    Y_Sigma_K = f_c_K*f_i_K*f_d_K*f_n_K  # 36Cl production per stopped muon 
    # Y_Sigma_K DEPENDS ON CHEMICAL COMPOSITION

    Y_Sigma = Y_Sigma_Ca + Y_Sigma_K 


    P_mu = Y_Sigma*Psi_mu_0*exp(-z/Lambda_mu)  # Unscaled slow negative muon production rate (atoms 36Cl g-1 yr-1)

    """ EPITHERMAL NEUTRONS"""

    B = np.sum(Xi_k*sigma_sc_k*N_kc)*1e-24  # Scattering rate parameter
    # B DEPENDS ON CHEMICAL COMPOSITION of the medium crossed - COLLUVIUM

    I_eff = np.sum(I_a_k*N_kc)*1e-24  # (Eq 3.9, Gosse & Phillips, 2001)
    # Effective macroscopic resonance integral for absorbtion of epith neutrons (cm2.g-1)
    # I_eff DEPENDS ON CHEMICAL COMPOSITION of the medium crossed - COLLUVIUM

    f_eth = N_k[60]*I_a_k[60]*(1e-24)/I_eff  # (Eq 3.17, Gosse & Phillips, 2001)
    # Fraction of epith neutrons absorbed by Cl35
    # f_eth DEPENDS ON CHEMICAL COMPOSITION

    p_E_th = exp(-I_eff/B)  # (Eq 3.8, Gosse & Phillips, 2001)
    # Resonance escape probability of a neutron from the epith energy range in subsurface
    # p_E_th DEPENDS ON CHEMICAL COMPOSITION

    A = np.sum(A_k*N_kc)/np.sum(N_kc) 
    # Average atomic weight (g/mol) -  COLLUVIUM

    
    R_eth = sqrt(A/A_a)  # (Eq 3.24, Gosse & Phillips, 2001)
    # Ratio of epithermal neutron production in subsurface to that in atm
    # R_eth DEPENDS ON CHEMICAL COMPOSITION of the medium crossed - COLLUVIUM
    

    Sigma_sc = np.sum(sigma_sc_k*N_kc)*1e-24  # (Eq 3.22, Gosse & Phillips, 2001)
    # Macroscopic neutron scattering cross-section (cm2.g-1)
    # Sigma_sc DEPENDS ON CHEMICAL COMPOSITION of the medium crossed - COLLUVIUM

    Xi = B/Sigma_sc   # Eq 3.19 Goss and Phillips
    # Average log decrement energy loss per neutron collision
    # Xi DEPENDS ON CHEMICAL COMPOSITION of the medium crossed - COLLUVIUM

    Sigma_eth = Xi*(I_eff + Sigma_sc)  # (Eq 3.18, Gosse & Phillips, 2001)
    # Effective epithermal loss cross-section (cm2.g-1)
    # Sigma_eth DEPENDS ON CHEMICAL COMPOSITION

    Lambda_fth = 1/Sigma_eth  # (Eq 3.18,Gosse & Phillips, 2001)
    # Attenuation length for absorbtion and moderation of epith neutrons flux (g.cm-2)
    # Lambda_fth DEPENDS ON CHEMICAL COMPOSITION

    D_eth = 1/(3*Sigma_sc*(1 - 2/(3*A)))  # (Eq 3.21, Gosse & Phillips, 2001)
    # Epithermal neutron diffusion coefficient (g.cm-2)
    # D_eth DEPENDS ON CHEMICAL COMPOSITION

    D_eth_a = 1/(3*Sigma_sc_a*(1 - 2/(3*A_a)))  # (Eq 3.21, Gosse & Phillips, 2001)
    # Epithermal neutron diffusion coefficient in atmosphere (g.cm-2)

    phi_star_eth = P_f_0*R_eth/(Sigma_eth - (D_eth/(Lambda_f**2)))  # Epithermal neutron flux at land/atmosphere
    # interface that would be observed in ss if interface was not present (n cm-2 yr-1)

    Y_s = np.sum(f_d_k*Y_n*ppmc[0:61]*Num_k/A_k)/np.sum(ppmc[0:61]*Num_k/A_k) 
    # Average neutron yield per stopped negative muon
    # Y_s DEPENDS ON CHEMICAL COMPOSITION of the medium crossed - COLLUVIUM

    

    phi_star_eth_a = P_f_0*R_eth_a/(Sigma_eth_a - (D_eth_a/(Lambda_f**2)))  # Epithermal neutron flux at land/atmosphere interface that would be observed in atm 
    # if interface was not present (n cm-2 yr-1)
    
    P_n_mu_0 = (Y_s*Psi_mu_0 + 5.8*1e-6*phi_mu_f_0)  # Fast muon flux at land surface SLHL, Eq.3.49 Gosse & Phillips, 2001 (n cm-2 yr-1)

    Deltaphi_2star_eth_a = phi_star_eth - D_eth_a*phi_star_eth_a/D_eth  # Adjusted difference between hypothetical equilibrium epithermal neutron fluxes in atm and ss (n cm-2 yr-1)

    L_eth = 1/sqrt(3*Sigma_sc*Sigma_eth) # Epithermal neutron diffusion length (g cm-2)
    # L_eth DEPENDS ON CHEMICAL COMPOSITION

    L_eth_a = 1/sqrt(3*Sigma_sc_a*Sigma_eth_a) # Epithermal neutron diffusion length in atm (g cm-2)

    FDeltaphi_star_eth = ((D_eth_a/L_eth_a)*(phi_star_eth_a - phi_star_eth) - Deltaphi_2star_eth_a*(D_eth/Lambda_f))/((D_eth_a/L_eth_a) + (D_eth/L_eth))  # EQ. 3.28 Gosse & Phillips, 2001

    # Difference between phi_star_eth,ss and actual epithermal neutron flux at land surface

    
    R_mu = EL_mu*P_n_mu_0/(EL_f*P_f_0*R_eth)  #Ratio of muon production rate to epithermal neutron production rate

    phi_eth_total = phi_star_eth*exp(-z/Lambda_f) + (1 + R_mu*R_eth)*FDeltaphi_star_eth*exp(-z/L_eth) + R_mu*phi_star_eth*exp(-z/Lambda_mu)  # Epithermal neutron flux (concentration) (n cm-2 yr-1)

  
    P_eth = (f_eth/Lambda_fth)*phi_eth_total*(1 - p_E_th) 

    A_eth = phi_star_eth  
    A_eth = A_eth*(f_eth/Lambda_fth)*(1 - p_E_th) 
    B_eth = (1 + R_mu*R_eth)*FDeltaphi_star_eth  
    B_eth = B_eth*(f_eth/Lambda_fth)*(1 - p_E_th) 
    C_eth = R_mu*phi_star_eth  
    C_eth = C_eth*(f_eth/Lambda_fth)*(1 - p_E_th) 

    """ THERMAL NEUTRONS """

    Sigma_th = np.sum(N_kc*sigma_th_k)*1e-24  # Eq 3.6 de Gosse and Phillips, 2001
    # macroscopic thermal neutron absorbtion cross-section 
    # Sigma_th DEPENDS ON CHEMICAL COMPOSITION of the medium crossed - COLLUVIUM

    f_th = sigma_th_k[60]*N_k[60]*1e-24/Sigma_th  # Eq 3.32 de Gosse and Phillips, 2001
    # fraction of thermal neutrons absorbed by Cl35
    # f_th DEPENDS ON ROCK CHEMICAL COMPOSITION 

    Lambda_th = 1/Sigma_th  # Eq 3.35 Gosse anf Phillips, 2001
    # Attenuation length for absorbtion of thermal neutrons flux (g.cm-2)
    # Lambda_th DEPENDS ON CHEMICAL COMPOSITION - COLLUVIUM

    
    R_th = p_E_th/p_E_th_a  # Ratio of thermal neutron production in ss to that in atm  Eq 3.34 Gosse and Phillips, 2001
    D_th = D_eth  # D_th = 2.99
    Deltaphi_star_eth_a = phi_star_eth - phi_star_eth_a  #difference in equilibrium epithermal neutron fluxes between atm and ss
    FDeltaphi_star_eth_a = (D_eth*Deltaphi_star_eth_a/L_eth - D_eth*Deltaphi_2star_eth_a/Lambda_f)/ (D_eth_a / L_eth_a + D_eth / L_eth )
        

    
    phi_star_th = (p_E_th_a*R_th*phi_star_eth)/(Lambda_fth*(Sigma_th - D_th/(Lambda_f**2))) 
    # thermal neutron flux at land/atm interface that would be observed in atm if interface not present (n.cm_2.a-1)

    JDeltaphi_star_eth = (p_E_th_a*R_th*FDeltaphi_star_eth)/(Lambda_fth*(Sigma_th - D_th/(L_eth**2)))  # Eq. 3.39 Gosse & Phillips, 2001
    # Portion of difference between phi_star_eth,ss and actual flux due to epithermal flux profile
    JDeltaphi_star_eth_a = (p_E_th_a*R_th_a*FDeltaphi_star_eth_a)/((1/Sigma_eth_a)*(Sigma_th_a - D_th_a/(L_eth_a**2))) 
    # Portion of difference between phi_star_eth,a and actual flux due to epithermal flux profile

    L_th = sqrt(D_th/Sigma_th) 
    L_th_a = sqrt(D_th_a/Sigma_th_a)  # thermal neutron diffusion length in atm (g cm-2)
    phi_star_th_a = (p_E_th_a*R_th_a*phi_star_eth_a)/(1/Sigma_eth_a*(Sigma_th_a - D_th_a/(Lambda_f**2)))  
    # thermal neutron flux at land/atmosphere interface that would be observed in atm if interface was not present (n cm-2 yr-1)

    Deltaphi_star_th = phi_star_th_a - phi_star_th  #difference between hypothetical equilibrium thermal neutron fluxes in atmosphere and ss

    JDeltaphi_star_th = (D_th_a*(phi_star_th_a/Lambda_f - JDeltaphi_star_eth_a/L_eth_a) - D_th*(phi_star_th/Lambda_f + JDeltaphi_star_eth/L_eth) + (D_th_a/L_th_a)*(Deltaphi_star_th + JDeltaphi_star_eth_a - JDeltaphi_star_eth))/((D_th/L_th) + (D_th_a/L_th_a))  #portion of difference between phi_star_th,ss and actual flux due to thermal flux profile


    
    # thermal neutron flux at land/atm interface that would be observed in atm if interface not present (n.cm_2.a-1)
    R_prime_mu = (p_E_th_a/p_E_th)*R_mu  # ratio of muon production rate to thermal neutron production rate

    phi_th_total = phi_star_th*exp(-z/Lambda_f) + (1 + R_prime_mu)*JDeltaphi_star_eth*exp(-z/L_eth) + (1 + R_prime_mu*R_th)*JDeltaphi_star_th*exp(-z/L_th) + R_prime_mu*phi_star_th*exp(-z/Lambda_mu)  # Thermal neutron flux (n.cm_2.a-1)


    P_th = (f_th/Lambda_th)*phi_th_total  # Result unscaled sample specific 36Cl production rate by capture of thermal neutrons (atoms 36Cl g-1 yr-1)

    A_th = phi_star_th  
    A_th = A_th*(f_th/Lambda_th) 
    B_th = (1 + R_prime_mu)*JDeltaphi_star_eth  
    B_th = B_th*(f_th/Lambda_th) 
    C_th = (1 + R_prime_mu*R_th)*JDeltaphi_star_th 
    C_th = C_th*(f_th/Lambda_th) 
    D_th = R_prime_mu*phi_star_th  
    D_th = D_th*(f_th/Lambda_th) 

    """ RADIOGENIC PROCTION """

    # Cosmogenic production:
    
    # ------------------------------------ Sample thickness factors -----------------------------------------
    #           Sample thickness factors as a function of sample position along direction e.
    # For spallation
    Q_sp = 1 + (th2**2/(6*(Lambda_f**2)))
    #

    # For epithermal neutrons
    A_eth_corr = 1 + ((th2/Lambda_f)**2)/6 
    B_eth_corr = 1 + ((th2/L_eth)**2)/6 
    C_eth_corr = 1 + ((th2/Lambda_mu)**2)/6 

    Q_eth = A_eth*exp(-z/Lambda_f)*A_eth_corr + B_eth*exp(-z/L_eth)*B_eth_corr + C_eth*exp(-z/Lambda_mu)*C_eth_corr 
 
    Q_eth = Q_eth/P_eth 
    
    # For thermal neutrons
    A_th_corr = 1 + ((th2/Lambda_f)**2)/6 
    B_th_corr = 1 + ((th2/L_eth)**2)/6 
    C_th_corr = 1 + ((th2/L_th)**2)/6 
    D_th_corr = 1 + ((th2/Lambda_mu)**2)/6 

    Q_th = A_th*exp(-z/Lambda_f)*A_th_corr + B_th*exp(-z/L_eth)*B_th_corr + C_th*exp(-z/L_th)*C_th_corr + D_th*exp(-z/Lambda_mu)*D_th_corr 

    Q_th = Q_th/P_th 

    # For muons
    Q_mu = 1 + (th2**2/(6*(Lambda_mu**2))) 

    P_cosmo = so_f*EL_f*(Q_sp*P_sp + S_L_th*Q_th*P_th + S_L_eth*Q_eth*P_eth) + so_f*EL_mu*Q_mu*P_mu 
    return P_cosmo

def clrock(sample, e, Lambda_e, so_e, EL_f, EL_mu):

    """ This function allows you to calculate the chlorine 36 production in the scarp
        INPUTS : sample, chemistry of the collected sample, numpy array (one dimentionnal)
                 e, float
                 Lambda_e, calculated with fitexp, float
                 so_e, calculatated with fitexp, float
                 EL_f, float
                 EL_mu, float
                 Psi_Cl36_Ca_0, float
                 rho_rock, rock density

        OUTPUTS : P_cosmo, float
                  P_rad, float
    """
    chimie = sample[0:62]
    # z = sample(n-3)
    thick = sample[63]  # g.cm-2
    th2 = thick/2

    # Conversion of oxyde percents into percents of the oxyded element
    # (Elements are given directly in ppm)
    # ppm = chimie.copy()
    # ppm[43] = chimie[43]*A_k[43]/(A_k[43] + 2*A_k[58])  # Si in percent
    # ppm[44] = chimie[44]*2*A_k[44]/(2*A_k[44] + 3*A_k[58])  # Al in percent
    # ppm[45] = chimie[45]*2*A_k[45]/(2*A_k[45] + 3*A_k[58])  # Fe in percent
    # ppm[46] = chimie[46]*A_k[46]/(A_k[46] + A_k[58])  # Mn in percent
    # ppm[47] = chimie[47]*A_k[47]/(A_k[47] + A_k[58])  # Mg in percent
    # ppm[48] = chimie[48]*A_k[48]/(A_k[48] + A_k[58])  # Ca in percent
    # ppm[49] = chimie[49]*2*A_k[49]/(2*A_k[49] + A_k[58])  # Na in percent
    # ppm[50] = chimie[50]*2*A_k[50]/(2*A_k[50] + A_k[58])  # K in percent
    # ppm[51] = chimie[51]*A_k[51]/(A_k[51] + 2*A_k[58])  # Ti in percent
    # ppm[52] = chimie[52]*2*A_k[52]/(2*A_k[53] + 5*A_k[58])  # P in percent
    # ppm[55] = chimie[55]*2*A_k[55]/(2*A_k[55] + A_k[58])  # H water in percent
    # O_water = chimie[55] - ppm[55]  # O_water in percent
    # ppm[57] = chimie[57]*A_k[57]/(A_k[57] + 2*A_k[58])  # C in percent

    # ppm[58] = (np.sum(chimie[43:53] )+chimie[57]) - (np.sum(ppm[43:53])+ ppm[57])  # O rock in percent
    # ppm[59]= O_water
    # ppm[43:53] = ppm[43:53]*1e4  # in ppm
    # ppm[55] = ppm[55]*1e4  # in ppm
    # ppm[57] = ppm[57]*1e4  # in ppm
    # ppm[58] = ppm[58]*1e4  # in ppm
    # ppm[59] = ppm[59]*1e4  # in ppm
    ppm = convert_2_oxyde(chimie, constants)

    N_k = (ppm[0:61]/A_k)*Avogadro*1e-6  # Concentrations in atom/g
    N_k[55] = N_k[55]/rho_rock  # divided by bulk-rock density according to CHLOE for H


    """ PRODUCTION RATE : SPALLATION"""


    C_Ca = ppm[61]*1e-6  # Mass concentration of Ca (g of Ca per g of rock) # from ICP
    P_sp_Ca = Psi_Cl36_Ca_0*C_Ca  # result unscaled 36Cl production by spallation of 40Ca (atoms 36Cl g-1 yr-1)

    
    
    C_K = ppm[50]*1e-6  # Mass concentration of K (g of K per g of rock)
    P_sp_K = Psi_Cl36_K_0*C_K  # result unscaled 36Cl production by spallation of 39K (atoms 36Cl g-1 yr-1)


    C_Ti = ppm[51]*1e-6  # Mass concentration of Ti (g of Ti per g of rock)
    P_sp_Ti = Psi_Cl36_Ti_0*C_Ti  # result unscaled 36Cl production by spallation of Ti (atoms 36Cl g-1 yr-1)

    C_Fe = ppm[45]*1e-6  # Mass concentration of Fe (g of Fe per g of rock)
    P_sp_Fe = Psi_Cl36_Fe_0*C_Fe  # result unscaled 36Cl production by spallation of Fe (atoms 36Cl g-1 yr-1)


    P_sp = (P_sp_Ca + P_sp_K + P_sp_Ti + P_sp_Fe)*exp(-e/Lambda_e)  # Unscaled Spallation production rate (atoms 36Cl g-1 yr-1)


    """ DIRECT CAPTURE OF SLOW NEGATIVES MUONS """

    f_c_Ca = (Num_k[48]*ppm[61]*1e-6/A_k[48])/(np.sum(Num_k*ppm[0:61]/A_k)*1e-6)  # for Ca (ICP)
    f_c_K = (Num_k[50]*ppm[50]*1e-6/A_k[50])/(np.sum(Num_k*ppm[0:61]/A_k)*1e-6)  # for K

    Y_Sigma_Ca = f_c_Ca*f_i_Ca*f_d_Ca*f_n_Ca  # 36Cl production per stopped muon
    # Y_Sigma_Ca DEPENDS ON CHEMICAL COMPOSITION
    Y_Sigma_K = f_c_K*f_i_K*f_d_K*f_n_K  # 36Cl production per stopped muon
    # Y_Sigma_K DEPENDS ON CHEMICAL COMPOSITION

    Y_Sigma = Y_Sigma_Ca + Y_Sigma_K


    P_mu = Y_Sigma*Psi_mu_0*exp(-e/Lambda_mu)  # Unscaled slow negative muon production rate (atoms 36Cl g-1 yr-1)


    """ EPITHERMAL NEUTRONS """
    B = np.sum(Xi_k*sigma_sc_k*N_k)*1e-24  # Scattering rate parameter
    # B DEPENDS ON CHEMICAL COMPOSITION

    I_eff = np.sum(I_a_k*N_k)*1e-24  # (Eq 3.9, Gosse & Phillips, 2001)
    # Effective macroscopic resonance integral for absorbtion of epith neutrons (cm2.g-1)
    # I_eff DEPENDS ON CHEMICAL COMPOSITION

    f_eth = N_k[60]*I_a_k[60]*(1e-24)/I_eff  # (Eq 3.17, Gosse & Phillips, 2001)
    # Fraction of epith neutrons absorbed by Cl35
    # f_eth DEPENDS ON CHEMICAL COMPOSITION

    p_E_th = exp(-I_eff/B)  # (Eq 3.8, Gosse & Phillips, 2001)
    # Resonance escape probability of a neutron from the epith energy range in subsurface
    # p_E_th DEPENDS ON CHEMICAL COMPOSITION

    A = np.sum(A_k*N_k)/np.sum(N_k)
    # Average atomic weight (g/mol)

    R_eth = sqrt(A/A_a)  # (Eq 3.24, Gosse & Phillips, 2001)
    # Ratio of epithermal neutron production in subsurface to that in atm
    # R_eth DEPENDS ON CHEMICAL COMPOSITION

    Sigma_sc = np.sum(sigma_sc_k*N_k)*1e-24  # (Eq 3.22, Gosse & Phillips, 2001)
    # Macroscopic neutron scattering cross-section (cm2.g-1)
    # Sigma_sc DEPENDS ON CHEMICAL COMPOSITION

    Xi = B/Sigma_sc   # Eq 3.19 Goss and Phillips
    # Average log decrement energy loss per neutron collision
    # Xi DEPENDS ON CHEMICAL COMPOSITION

    Sigma_eth = Xi*(I_eff + Sigma_sc)  # (Eq 3.18, Gosse & Phillips, 2001)
    # Effective epithermal loss cross-section (cm2.g-1)
    # Sigma_eth DEPENDS ON CHEMICAL COMPOSITION

    Lambda_eth = 1/Sigma_eth  # (Eq 3.18,Gosse & Phillips, 2001)
    # Attenuation length for absorbtion and moderation of epith neutrons flux (g.cm-2)
    # Lambda_eth DEPENDS ON CHEMICAL COMPOSITION

    D_eth = 1/(3*Sigma_sc*(1 - 2/(3*A)))  # (Eq 3.21, Gosse & Phillips, 2001)
    # Epithermal neutron diffusion coefficient (g.cm-2)
    # D_eth DEPENDS ON CHEMICAL COMPOSITION

    D_eth_a = 1/(3*Sigma_sc_a*(1 - 2/(3*A_a)))  # (Eq 3.21, Gosse & Phillips, 2001)
    # Epithermal neutron diffusion coefficient in atmosphere (g.cm-2)

    phi_star_eth = P_f_0*R_eth/(Sigma_eth - (D_eth/(Lambda_e**2)))  # Epithermal neutron flux at land/atmosphere
    # interface that would be observed in ss if interface was not present (n cm-2 yr-1)

    Y_s = np.sum(f_d_k*Y_n*ppm[0:61]*Num_k/A_k)/np.sum(ppm[0:61]*Num_k/A_k)
    # Average neutron yield per stopped negative muon
    # Y_s DEPENDS ON CHEMICAL COMPOSITION

    phi_star_eth_a = P_f_0*R_eth_a/(Sigma_eth_a - (D_eth_a/(Lambda_e**2)))  # Epithermal neutron flux at land/atmosphere interface that would be observed in atm

    # if interface was not present (n cm-2 yr-1)
    P_n_mu_0 = (Y_s*Psi_mu_0 + 5.8e-6*phi_mu_f_0) # Fast muon flux at land surface SLHL, Eq.3.49 Gosse & Phillips, 2001 (n cm-2 yr-1)

    Deltaphi_2star_eth_a = phi_star_eth - D_eth_a*phi_star_eth_a/D_eth  # Adjusted difference between hypothetical equilibrium epithermal neutron fluxes in atm and ss (n cm-2 yr-1)

    L_eth = 1/sqrt(3*Sigma_sc*Sigma_eth) # Epithermal neutron diffusion length (g cm-2)
    # L_eth DEPENDS ON CHEMICAL COMPOSITION

    L_eth_a = 1/sqrt(3*Sigma_sc_a*Sigma_eth_a) # Epithermal neutron diffusion length in atm (g cm-2)

    FDeltaphi_star_eth = ((D_eth_a/L_eth_a)*(phi_star_eth_a - phi_star_eth) - Deltaphi_2star_eth_a*(D_eth/Lambda_e))/((D_eth_a/L_eth_a) + (D_eth/L_eth))  # EQ. 3.28 Gosse & Phillips, 2001

    # Difference between phi_star_eth,ss and actual epithermal neutron flux at land surface


    R_mu = EL_mu*P_n_mu_0/(EL_f*P_f_0*R_eth)  #Ratio of muon production rate to epithermal neutron production rate

    phi_eth_total = phi_star_eth*exp(-e/Lambda_e) + (1 + R_mu*R_eth)*FDeltaphi_star_eth*exp(-e/L_eth) + R_mu*phi_star_eth*exp(-e/Lambda_mu)  # Epithermal neutron flux (concentration) (n cm-2 yr-1)

    P_eth = (f_eth/Lambda_eth)*phi_eth_total*(1 - p_E_th)

    A_eth = phi_star_eth
    A_eth = A_eth*(f_eth/Lambda_eth)*(1 - p_E_th)
    B_eth = (1 + R_mu*R_eth)*FDeltaphi_star_eth
    B_eth = B_eth*(f_eth/Lambda_eth)*(1 - p_E_th)
    C_eth = R_mu*phi_star_eth
    C_eth = C_eth*(f_eth/Lambda_eth)*(1 - p_E_th)

    # ------------------------------------ Thermal neutrons ------------------------------------

    Sigma_th = np.sum(N_k*sigma_th_k)*1e-24  # Eq 3.6 de Gosse and Phillips, 2001
    # macroscopic thermal neutron absorbtion cross-section
    # Sigma_th DEPENDS ON CHEMICAL COMPOSITION

    f_th = sigma_th_k[60]*N_k[60]*1e-24/Sigma_th  # Eq 3.32 de Gosse and Phillips, 2001
    # fraction of thermal neutrons absorbed by Cl35
    # f_th DEPENDS ON CHEMICAL COMPOSITION

    Lambda_th = 1/Sigma_th  # Eq 3.35 Gosse anf Phillips, 2001
    # Attenuation length for absorbtion of thermal neutrons flux (g.cm-2)
    # Lambda_th DEPENDS ON CHEMICAL COMPOSITION

    R_th = p_E_th/p_E_th_a  # Ratio of thermal neutron production in ss to that in atm  Eq 3.34 Gosse and Phillips, 2001
    D_th = D_eth  # D_th = 2.99
    Deltaphi_star_eth_a = phi_star_eth - phi_star_eth_a  # difference in equilibrium epithermal neutron fluxes between atm and ss
    FDeltaphi_star_eth_a = (D_eth*Deltaphi_star_eth_a/L_eth - D_eth*Deltaphi_2star_eth_a/Lambda_e)/ (D_eth_a / L_eth_a + D_eth / L_eth )

    phi_star_th = (p_E_th_a*R_th*phi_star_eth)/(Lambda_eth*(Sigma_th - D_th/(Lambda_e**2)))

    JDeltaphi_star_eth = (p_E_th_a*R_th*FDeltaphi_star_eth)/(Lambda_eth*(Sigma_th - D_th/(L_eth**2)))  # Eq. 3.39 Gosse & Phillips, 2001
    # Portion of difference between phi_star_eth,ss and actual flux due to epithermal flux profile
    JDeltaphi_star_eth_a = (p_E_th_a*R_th_a*FDeltaphi_star_eth_a)/((1/Sigma_eth_a)*(Sigma_th_a - D_th_a/(L_eth_a**2)))
    # Portion of difference between phi_star_eth,a and actual flux due to epithermal flux profile

    L_th = sqrt(D_th/Sigma_th)
    L_th_a = sqrt(D_th_a/Sigma_th_a)  # thermal neutron diffusion length in atm (g cm-2)
    phi_star_th_a = (p_E_th_a*R_th_a*phi_star_eth_a)/(1/Sigma_eth_a*(Sigma_th_a - D_th_a/(Lambda_e**2)))
    # thermal neutron flux at land/atmosphere interface that would be observed in atm if interface was not present (n cm-2 yr-1)

    Deltaphi_star_th = phi_star_th_a - phi_star_th  # difference between hypothetical equilibrium thermal neutron fluxes in atmosphere and ss

    JDeltaphi_star_th = (D_th_a*(phi_star_th_a/Lambda_e - JDeltaphi_star_eth_a/L_eth_a) - D_th*(phi_star_th/Lambda_e + JDeltaphi_star_eth/L_eth) + (D_th_a/L_th_a)*(Deltaphi_star_th + JDeltaphi_star_eth_a - JDeltaphi_star_eth))/ ((D_th/L_th) + (D_th_a/L_th_a))  # portion of difference between phi_star_th,ss and actual flux due to thermal flux profile

    # thermal neutron flux at land/atm interface that would be observed in atm if interface not present (n.cm_2.a-1)
    R_prime_mu = (p_E_th_a/p_E_th)*R_mu  # ratio of muon production rate to thermal neutron production rate

    phi_th_total = phi_star_th*exp(-e/Lambda_e) + (1 + R_prime_mu)*JDeltaphi_star_eth*exp(-e/L_eth) + (1 + R_prime_mu*R_th)*JDeltaphi_star_th*exp(-e/L_th) +  R_prime_mu*phi_star_th*exp(-e/Lambda_mu)  # Thermal neutron flux (n.cm_2.a-1)

    P_th = (f_th/Lambda_th)*phi_th_total  # Result unscaled sample specific 36Cl production rate by capture of thermal neutrons (atoms 36Cl g-1 yr-1)

    A_th = phi_star_th
    A_th = A_th*(f_th/Lambda_th)
    B_th = (1 + R_prime_mu)*JDeltaphi_star_eth
    B_th = B_th*(f_th/Lambda_th)
    C_th = (1 + R_prime_mu*R_th)*JDeltaphi_star_th
    C_th = C_th*(f_th/Lambda_th)
    D_th = R_prime_mu*phi_star_th
    D_th = D_th*(f_th/Lambda_th)

    """ RADIOGENIC PRODUCTION """

    X = (np.sum(ppm[0:61]*S_i*Y_U_n))/(np.sum(S_i*ppm[0:61]))
    # X DEPENDS ON CHEMICAL COMPOSITION

    Y = (np.sum(ppm[0:61]*S_i*Y_Th_n))/(np.sum(S_i*ppm[0:61]))
    # Y DEPENDS ON CHEMICAL COMPOSITION

    U = ppm[36]  # Concentration en Uranium (ppm)
    Th = ppm[34]  # Concentration en Thorium (ppm)

    P_n_alphan = X*U + Y*Th  # alpha,n reactions
    P_n_sf = 0.429*U  # spontaneous fission
    P_th_r = (P_n_alphan + P_n_sf)*p_E_th  # total radiogenic thermal neutron production
    P_eth_r = (P_n_alphan + P_n_sf)*(1 - p_E_th)  # total radiogenic epithermal neutron production
    P_rad = P_th_r*f_th + P_eth_r*f_eth



    """ COSMOGENIC PRODUCTION """
    # ------------------------------------ Sample thickness factors -----------------------------------------
    #           Sample thickness factors as a function of sample position along direction e.
    # For spallation
    Q_sp = 1 + (th2**2/(6*(Lambda_e**2)))
    #

    # For epithermal neutrons
    A_eth_corr = 1 + ((th2/Lambda_e)**2)/6
    B_eth_corr = 1 + ((th2/L_eth)**2)/6
    C_eth_corr = 1 + ((th2/Lambda_mu)**2)/6

    Q_eth = A_eth*exp(-e/Lambda_e)*A_eth_corr + B_eth*exp(-e/L_eth)*B_eth_corr + C_eth*exp(-e/Lambda_mu)*C_eth_corr

    Q_eth = Q_eth/P_eth

    # For thermal neutrons
    A_th_corr = 1 + ((th2/Lambda_e)**2)/6
    B_th_corr = 1 + ((th2/L_eth)**2)/6
    C_th_corr = 1 + ((th2/L_th)**2)/6
    D_th_corr = 1 + ((th2/Lambda_mu)**2)/6

    Q_th = A_th*exp(-e/Lambda_e)*A_th_corr + B_th*exp(-e/L_eth)*B_th_corr + C_th*exp(-e/L_th)*C_th_corr + D_th*exp(-e/Lambda_mu)*D_th_corr

    Q_th = Q_th/P_th

    # For muons
    Q_mu = 1 + (th2**2/(6*(Lambda_mu**2)))

    P_cosmo = so_e*EL_f*(Q_sp*P_sp + S_L_th*Q_th*P_th + S_L_eth*Q_eth*P_eth) + so_e*EL_mu*Q_mu*P_mu
    # P_sp_sc = so_e*EL_f*Q_sp*P_sp
    # P_mu_sc = so_e*EL_mu*Q_mu*P_mu
    # P_th_sc = so_e*EL_f*S_L_th*Q_th*P_th
    # P_eth_sc = so_e*EL_f*S_L_eth*Q_eth*P_eth

    return P_cosmo, P_rad

