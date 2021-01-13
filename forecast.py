# -*- coding: utf-8 -*-
"""
Master Thesis Dominic Scotoni

PV Forecast File

Versions:
    1: Gaussian Samples out-of-sample
    2: Gaussian Samples in-sample 
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
### PACKAGES ###
import numpy as np
import pandas as pd
import pickle as pkl


### SCRIPTS ###
import param as pm

###############################################################################
## PV FORECAST ## 
###############################################################################
def pv_fcst():                
    ### VERSION 1 ###
    if pm.V_FCST == 1:
        nSamples = 10000 # define number of samples
        
        # forecast , change angles for extending solar hours
        if pm.FCSTCASE[0] == 'summer':
            pvMuRaw = 0.57*np.sin(np.linspace(-np.deg2rad(40),\
                                              np.pi+np.deg2rad(55),\
                                              int(24/pm.TIMESTEP)))
        elif pm.FCSTCASE[0] == 'winter':
            pvMuRaw = 0.35*np.sin(np.linspace(-np.deg2rad(65),\
                                              np.pi+np.deg2rad(85),\
                                              int(24/pm.TIMESTEP)))
        
        
        # remove very small values
        pvMuRaw[pvMuRaw < 1e-5] = 0
        
        
        # standard deviation
        pvSigmaRaw = 0.25*np.sin(np.linspace(-np.deg2rad(35),\
                                            np.pi+np.deg2rad(55),\
                                            int(24/pm.TIMESTEP)))*pvMuRaw
            
        pvSigmaRaw[pvSigmaRaw < 1e-5] = 0        
        
        
        # daily profile according to mean and standard deviation above
        pvDaily = np.array([np.random.normal(pvMuRaw[i],pvSigmaRaw[i],nSamples)\
                            for i in range(int(24/pm.TIMESTEP))])
        
        # remove samples with forecast > 1
        pvDailyFiltered = np.zeros((int(24/pm.TIMESTEP),1))
        for i in range(nSamples):
            if np.any(pvDaily[:,i] > 1):
                pvDailyFiltered
            else:
                pvDailyFiltered = np.append(pvDailyFiltered,\
                                            pvDaily[:,i]\
                                .reshape(int(24/pm.TIMESTEP),1),axis=1)
                
        
        ### TAKE VALUES FROM 12:00 +- T/2 ###
        t_middle = 1/pm.TIMESTEP*12
        t_start = int(t_middle - pm.T/2)
        t_end = int(t_middle + pm.T/2)
        
        # average PV production 
        pvMu = pvMuRaw[t_start:t_end]
        pvSigma = pvSigmaRaw[t_start:t_end]
        
        # for pkl export
        dataFcst = pvDailyFiltered[t_start:t_end,1:]
        
        
        ###################################################################
        ## OUT-OF-SAMPLE ANALYSIS ## 
        ###################################################################
        pvMC = np.array([np.random.normal(pvMuRaw[i],pvSigmaRaw[i],nSamples)\
                            for i in range(int(24/pm.TIMESTEP))])
        
        # remove samples with forecast > 1
        pvMCFiltered = np.zeros((int(24/pm.TIMESTEP),1))
        for i in range(nSamples):
            if np.any(pvMC[:,i] > 1):
                pvMCFiltered
            else:
                pvMCFiltered = np.append(pvMCFiltered,\
                                            pvDaily[:,i]\
                                .reshape(int(24/pm.TIMESTEP),1),axis=1)
                
        
        ### TAKE VALUES FROM 12:00 +- T/2 ###
        t_middle = 1/pm.TIMESTEP*12
        t_start = int(t_middle - pm.T/2)
        t_end = int(t_middle + pm.T/2)
        
        
        dataMC = pvMCFiltered[t_start:t_end,1:]
        
        ###################################################################
        ## EXPORT PKL ## 
        ###################################################################
        export = [pvMu, pvSigma, dataMC, dataFcst]
        fcstFile = 'src/fcst/forecastPV_v%s_%s_t%s.pkl'\
                    %(pm.V_FCST,pm.FCSTCASE[0],pm.T)
        output = open(fcstFile, 'wb') # create output file
        pkl.dump(export, output) # write data to output file
        output.close() # close output file
        
       
        
    ### VERSION 2 ###
    if pm.V_FCST == 1:
        nSamples = 10000 # define number of samples
        
        # forecast , change angles for extending solar hours
        if pm.FCSTCASE[0] == 'summer':
            pvMuRaw = 0.57*np.sin(np.linspace(-np.deg2rad(40),\
                                              np.pi+np.deg2rad(55),\
                                              int(24/pm.TIMESTEP)))
        elif pm.FCSTCASE[0] == 'winter':
            pvMuRaw = 0.35*np.sin(np.linspace(-np.deg2rad(65),\
                                              np.pi+np.deg2rad(85),\
                                              int(24/pm.TIMESTEP)))
        
        
        # remove very small values
        pvMuRaw[pvMuRaw < 1e-5] = 0
        
        
        # standard deviation
        pvSigmaRaw = 0.25*np.sin(np.linspace(-np.deg2rad(35),\
                                            np.pi+np.deg2rad(55),\
                                            int(24/pm.TIMESTEP)))*pvMuRaw
            
        pvSigmaRaw[pvSigmaRaw < 1e-5] = 0        
        
        
        # daily profile according to mean and standard deviation above
        pvDaily = np.array([np.random.normal(pvMuRaw[i],pvSigmaRaw[i],nSamples)\
                            for i in range(int(24/pm.TIMESTEP))])
        
        # remove samples with forecast > 1
        pvDailyFiltered = np.zeros((int(24/pm.TIMESTEP),1))
        for i in range(nSamples):
            if np.any(pvDaily[:,i] > 1):
                pvDailyFiltered
            else:
                pvDailyFiltered = np.append(pvDailyFiltered,\
                                            pvDaily[:,i]\
                                .reshape(int(24/pm.TIMESTEP),1),axis=1)
                
        
        ### TAKE VALUES FROM 12:00 +- T/2 ###
        t_middle = 1/pm.TIMESTEP*12
        t_start = int(t_middle - pm.T/2)
        t_end = int(t_middle + pm.T/2)
        
        # average PV production 
        pvMu = pvMuRaw[t_start:t_end]
        pvSigma = pvSigmaRaw[t_start:t_end]
        
        # for pkl export
        dataFcst = pvDailyFiltered[t_start:t_end,1:]
        
        
        ###################################################################
        ## IN-SAMPLE ANALYSIS ## 
        ###################################################################
        dataMC = dataFcst
        
        ###################################################################
        ## EXPORT PKL ## 
        ###################################################################
        export = [pvMu, pvSigma, dataMC, dataFcst]
        fcstFile = 'src/fcst/forecastPV_v%s_%s_t%s.pkl'\
                    %(pm.V_FCST,pm.FCSTCASE[0],pm.T)
        output = open(fcstFile, 'wb') # create output file
        pkl.dump(export, output) # write data to output file
        output.close() # close output file
    
    return pvMu, pvSigma, dataMC, dataFcst