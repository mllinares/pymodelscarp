#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 11:15:32 2023

@author: llinares
"""
import numpy as np


def aicc(measurements, calculations, nb_param):
    """ This function allows to calculate the Akaike criterion
        INPUTS : measuremments, your data, numpy array
                 calculations, your synthetic data, numpy array
                 nb_param, integer
        
        OUTPUT : aicc, Akaike criterion, numpy array
        
        From modelscarp.m, A. Schlagenhauf et al. 2010, adapted for python by M. Llinares
        """

    n = len(measurements) 
    aicc = np.sum((measurements - calculations)**2)
    aicc = n*np.log(aicc/n) + (2*n*nb_param)/(n - nb_param - 1)
    return aicc
