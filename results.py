# -*- coding: utf-8 -*-
"""
Master Thesis Dominic Scotoni

Extract OPF results and sort arrays
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
#### PACKAGES ####
import numpy as np

#### SCRIPTS ####
import param as pm

###############################################################################
## STORE RESULTS ## 
###############################################################################
def read_out(var,dim,sol):
    tmp = np.zeros((dim,pm.T))
    for i in range(len(sol)):
        idxStart = sol[i][0].find('[')
        idxSep = sol[i][0].find(',')
        idxEnd = sol[i][0].find(']')
        if var == sol[i][0][:idxStart]:
            n = int(sol[i][0][idxStart+1:idxSep])
            t = int(sol[i][0][idxSep+1:idxEnd])
            tmp[n,t] = sol[i][1]
            
    return tmp

def read_out_3ph(var,dim,sol):
    tmp = np.zeros((dim*pm.N_PH,pm.T))
    for i in range(len(sol)):
        idxStart = sol[i][0].find('[')
        idxSep = sol[i][0].find(',')
        idxEnd = sol[i][0].find(']')
        if var == sol[i][0][:idxStart]:
            n = int(sol[i][0][idxStart+1:idxSep])
            t = int(sol[i][0][idxSep+1:idxEnd])
            tmp[n,t] = sol[i][1]
            
    return tmp

    
def read_out_phase(var,dim,sol):
    tmp = np.zeros((dim,pm.N_PH,pm.T))
    for i in range(len(sol)):
        idxStart = sol[i][0].find('[')
        idxSep = sol[i][0].find(',')
        idxEnd = sol[i][0].find(']')
        if var == sol[i][0][:idxStart]:
            x = int(sol[i][0][idxStart+1:idxSep])
            if pm.N_PH == 3:  
                n = np.floor_divide(x,pm.N_PH)
                p = np.mod(x,pm.N_PH)
            else:
                n = x
                p=0
            t = int(sol[i][0][idxSep+1:idxEnd])
            tmp[n,p,t] = sol[i][1]
                    
    return tmp

def phase_sort(dim,data):
    tmp = np.zeros((dim,pm.N_PH,pm.T))
    for t in range(pm.T):
        for i in range(dim):
            for j in range(pm.N_PH):
                tmp[i,j,t] = data[i*pm.N_PH+j,t]
    return tmp



