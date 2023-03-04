# -*- coding: utf-8 -*-
"""

@author: Maureen Llinares, adapted for python from:
    
    Schlagenhauf A., Gaudemer Y., Benedetti L., Manighetti I., Palumbo L.,
    Schimmelpfennig I., Finkel R., Pou K.
    G.J.Int., 2010
    
    Tesson & Benedetti, G.J.Int., 2019
    
"""

import cl36_concentration as cl36
import geometric_scaling_factors
import time 

# !!! Fonction directe avce le scaling géometrique dans l'inversion 
def mds(param, constants, seismic_scenario):
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
    synthetic_production, height, out = cl36.cl36_seismic_sequence(param, constants, seismic_scenario, scaling_factors, cl36_long_term)
    toc3=time.time()
    print('seismic CPU : ', toc3-tic3, 's')

    return synthetic_production

# !!! Fonction directe sans le scaling (gain de temps)
def mds_no_scale(param, constants, seismic_scenario, scaling_factors):

    """ Calculate 36Cl concentration due to long term history """
    cl36_long_term, h_longterm = cl36.long_term(param, constants, seismic_scenario, scaling_factors)
    
    """ Seismic phase """
    synthetic_production, height, out = cl36.cl36_seismic_sequence(param, constants, seismic_scenario, scaling_factors, cl36_long_term)
    # synthetic_production=1
    return synthetic_production
    