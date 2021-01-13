# -*- coding: utf-8 -*-
"""
Master Thesis Dominic Scotoni

BFS File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
### PACKAGES ###
import time as tm
import numpy as np
import sys

### SCRIPTS ###
import param as pm
import data as dt
from data import bus,oltc,branch


###############################################################################
## BFS ALGORITHM ## 
###############################################################################
def bfs_solve(sNet, time, tau):
    t_tmp = tm.time() # time for BFS
    dimN = dt.n*pm.N_PH # dimension of all buses * phases
    dimN1 = (dt.n-1)*pm.N_PH # dimension of (n-1) buses * phases
    dimL = dt.l*pm.N_PH # dimension of all branches * phases
    vSlack = (bus.vSlackRe[pm.N_PH:] + 1j*bus.vSlackIm[pm.N_PH:]) # complex slack voltage
    dVOLTC = np.tile(np.array([[oltc.dV*bus.phasor_slack[i]*tau[i,t] for t in range(time)]\
                               for i in range(pm.N_PH)]),(dt.n,1)) # voltage difference from OLTC
    
    ### INITIALIZE ###
    # arrays for results
    bfs_vRe = bus.vSlackRe.reshape(dimN,1)*np.ones((1,time)) + np.real(dVOLTC)
    bfs_vIm = bus.vSlackIm.reshape(dimN,1)*np.ones((1,time)) + np.imag(dVOLTC)
    bfs_iRe = np.zeros((dimL,time))
    bfs_iIm = np.zeros((dimL,time))
    
    
    for t in range(time):
        ### RUN BFS FOR EACH TIMESTEP INDEPENDENTLY ###
        # iteration counters
        k = 0
        bfsError = 1 # initial BFS error
        
        # flat start
        vBus = vSlack.reshape(dimN1,1)
        
        # voltage difference from OLTC
        vOLTC = dVOLTC[pm.N_PH:,t].reshape(dimN1)
        
        # nodal current injection
        iInj = np.zeros((dimN1,1))
        
        ### RUN BFS ###
        while bfsError > pm.ETA_BFS:
            # update nodal current injection
            iInj = np.append(iInj,\
                             (np.conj(sNet[:,t])/np.conj(vBus[:,-1])).reshape(dimN1,1),\
                                 axis=1)
            
            # update nodal voltages
            vBus = np.append(vBus,\
                             (vSlack + vOLTC + np.dot(dt.bibv,iInj[:,-1]))\
                                 .reshape(dimN1,1),axis=1)
                
            # calculate BFS error
            if k >= pm.K_MAX:
                sys.exit("NO BFS CONVERGENCE")
            else:
                bfsError = np.max(np.abs(np.abs(vBus[:,-1]) - np.abs(vBus[:,-2])))
                
            # update iteration counter
            k += 1
            
        ### STORE RESULTS ###
        bfs_vRe[pm.N_PH:,t] = np.real(vBus[:,-1])
        bfs_vIm[pm.N_PH:,t] = np.imag(vBus[:,-1])
        bfs_iRe[:,t] = np.real(np.dot(dt.bibc,iInj[:,-1]))
        bfs_iIm[:,t] = np.imag(np.dot(dt.bibc,iInj[:,-1]))
        
    ###########################################################################
    ## POWER LOSSES ## 
    ###########################################################################
    ### VOLTAGE DROP BRANCHES ###
    dVRe = np.zeros((dimL,time))
    dVIm = np.zeros((dimL,time))
    for t in range(time):
        for j in range(dt.l):
            for k in range(pm.N_PH):
                dVRe[j*pm.N_PH+k,t] = bfs_vRe[pm.N_PH*branch.fbus[j]+k,t] -\
                                        bfs_vRe[pm.N_PH*branch.tbus[j]+k,t] # real part
                                        
                dVIm[j*pm.N_PH+k,t] = bfs_vIm[pm.N_PH*branch.fbus[j]+k,t] -\
                                        bfs_vIm[pm.N_PH*branch.tbus[j]+k,t] # imag part
    
    # losses
    bfs_pLoss = np.abs(np.array([[bfs_iRe[i,t]*dVRe[i,t] +\
                                  bfs_iIm[i,t]*dVIm[i,t]\
                                      for t in range(time)] for i in range(dimL)]))
    
    
    ### RETURN RESULTS ###
    t_BFS = tm.time()-t_tmp # time for BFS
    
    return bfs_vRe, bfs_vIm, bfs_iRe, bfs_iIm,t_BFS, bfs_pLoss





    

    







