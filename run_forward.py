# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 11:47:42 2022

@author: Maureen
"""



import forward_function
import geometric_scaling_factors
import cl36_concentration as cl36
import parameters
from constants import constants
from seismic_scenario import seismic_scenario

import matplotlib.pyplot as plt
import numpy as np
import time

param = parameters.param() # import site parameters

""" First calculate scaling factors """
tic=time.time()

scaling_factors={}
depth_rock, depth_coll, surf_rock, S_S = geometric_scaling_factors.neutron_scaling(param, constants, len(seismic_scenario['ages']))

scaling_factors['Lambda_f_e'] = surf_rock['lambda_e']
scaling_factors['so_f_e'] = surf_rock['s_e']
scaling_factors['Lambda_f_diseg'] = depth_rock['lambda_diseg']
scaling_factors['so_f_diseg'] = depth_rock['s_diseg']
scaling_factors['Lambda_f_beta_inf'] = depth_coll['lambda_beta']
scaling_factors['so_f_beta_inf'] = depth_coll['s_beta']
scaling_factors['S_S'] = S_S

toc=time.time()
print('scaling CPU : ', toc-tic, 's')

""" Then calculate 36Cl concentration due to long term history """
tic2=time.time()
cl36_long_term, h_longterm = cl36.long_term(param, constants, seismic_scenario, scaling_factors)
toc2=time.time()
print('long_term CPU : ', toc2-tic2, 's')

""" Seismic phase """
tic3=time.time()
synthetic_production, height, out = forward_function.mds(param, constants, seismic_scenario, scaling_factors, cl36_long_term)
toc3=time.time()
print('seismic CPU : ', toc3-tic3, 's')



""" Plotting results """
plt.subplot(2,1,1)
plt.title('Long-term')
plt.plot(cl36_long_term, h_longterm*1e-2, color = 'steelblue', linestyle='', marker='.')
plt.xlabel ('36Cl [at/g]')
plt.ylabel ('Height (m)')



plt.subplot(2, 1, 2)
measured_production = param.cl36AMS
plt.title('Final')
plt.plot(synthetic_production, height*1e-2, color = 'cornflowerblue', linestyle='', marker='.', label='synthetic')
plt.plot(measured_production, height*1e-2, color = 'black', linestyle='', marker='.', label='measured')
plt.xlabel ('36Cl [at/g]')
plt.ylabel ('Height (m)')
plt.legend()
plt.tight_layout()


""" Plotting results """
slip = seismic_scenario['slips']
age = seismic_scenario['ages']
total_height = param.Hfinal
measured_production = param.cl36AMS
uncertainty = param.sig_cl36AMS

# if there is a trench, substract the height of the trench to the total sampling height
# if age[-1] == 0:
#     height = height - slip[-1]
#     true_slip = slip[0:-1]
#     plt.hlines((total_height - slip[-1] - np.cumsum(true_slip)) * 1e-2, min(synthetic_production),
#                 max(synthetic_production), linestyle='dashed', color='black')
#     plt.ylim(min(height) * 1e-2, (total_height - slip[-1]) * 1e-2)
# else:
#     plt.hlines((total_height - np.cumsum(slip[0:-1])) * 1e-2, min(synthetic_production), max(synthetic_production),
#                 linestyle='dashed', color='black')


# plt.plot(measured_production, height * 1e-2, '.', label='data')
# plt.xlabel(' [Cl 36]')
# plt.ylabel('Height (m)')
# plt.legend()

# # RMSw (weighted least square) :
# rmsw = ((measured_production - synthetic_production) / uncertainty) ** 2
# rmsw = np.sqrt(np.sum(rmsw))

# print('RMS', rmsw)
# save_array = np.zeros((len(height), 2))
# save_array[:, 0] = height
# save_array[:, 1] = synthetic_production
# toc=time.time()
# plt.plot(synthetic_production, height * 1e-2, '.k', label='synthetic')
# print('\nCPU time main : ', toc-tic)
# plt.show()
