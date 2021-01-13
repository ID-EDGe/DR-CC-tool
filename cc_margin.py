# -*- coding: utf-8 -*-
"""
Master Thesis Dominic Scotoni

Calculation of uncertainty margins for CC
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
### PACKAGES ###o
import numpy as np


### SCRIPTS ###
import param as pm
import data as dt
from data import cc,pv,inverter
import results as rlt


###############################################################################
## CALCULATE AND UPDATE UNCERTAINTY MARGIN ## 
###############################################################################
def cc_upd(solBFSOPF, iLoad, iPV):    
    dimL = dt.l*pm.N_PH # dimension of all branches (l branch * p phases)
    
    # function to discard very small values for numerical stability
    def clean(data):
        data[np.abs(data) < 1e-10] = 0
        return data
    
    ###########################################################################
    ## RESULTS FROM BFS-OPF ## 
    ###########################################################################    
    ### VOLTAGE ###
    sol_vRe = solBFSOPF[1][0][pm.N_PH:,:] # bfs real part
    sol_vIm = solBFSOPF[1][1][pm.N_PH:,:]  # bfs imag part
    sol_vMag = np.abs(sol_vRe + 1j*sol_vIm) # magnitude
    sol_vSq = np.square(sol_vMag) # square of magnitude
    
    ### BRANCH CURRENT ###
    sol_iRe = solBFSOPF[1][2] # bfs real part
    sol_iIm = solBFSOPF[1][3] # bfs imag part
    sol_iMag = np.abs(sol_iRe + 1j*sol_iIm) # magnitude
    
    
    ### THREE-PHASE POWER FROM PV ###
    # active power
    sol_pR = clean(rlt.read_out_3ph('pR', dt.n, solBFSOPF[0][0])[pm.N_PH:,:]) 
    
    # reactive power
    sol_qR = clean(rlt.read_out_3ph('qR', dt.n, solBFSOPF[0][0])[pm.N_PH:,:])
    
    
    ### AUXILIARY ARRAYS ###  
    # power ratio PV inverter        
    pr_ren = np.array([[sol_qR[i,t]/sol_pR[i,t]\
                        if sol_pR[i,t]!=0 else sol_qR[i,t]\
                        for t in range(pm.T)] for i in range(dimL)])
        
    
    ###########################################################################
    ## SENSITIVITY MATRIX ## 
    ###########################################################################  
    ### DERIVATIVES OF MAGNITUDES TO UNCERTAINTY VECTOR ###
    # function to multiply voltage at each node with uncertainty array & tile 
    # to multiply with topology matrices
    def mult_v(data): 
        # array u containing 1s for phases with PV, otherwise 0
        # threshold 1e-8 for numerical reasons
        uncertArray = np.tile(np.where(pv.icPhase[:,iLoad*dt.pvCase+iPV] > 1e-8,1,0)\
                              .reshape(1,dimL),(dimL,1))
         
        # transpose voltage vector and repeat for l-times along 0 axis
        tmp = np.tile(data.reshape(1,dimL),(dimL,1))

        return tmp*uncertArray
        
        
    ### POWER RATIO GAMMA ###
    if pm.FLGCC_GMA == 0:
        # pre-define gamma 
        pr = np.zeros((dimL,pm.T))
        pr[np.where(inverter.capPV[pm.N_PH:,iLoad*dt.pvCase+iPV] != 0)] = pm.CC_GMA 
    else:
        pr = pr_ren
    
    # pre-allocate derivatives
    dIRe = np.zeros((dimL,dimL,pm.T)) # branch current real part
    dIIm = np.zeros((dimL,dimL,pm.T)) # branch current imag part
    dVRe = np.zeros((dimL,dimL,pm.T)) # bus voltage real part
    dVIm = np.zeros((dimL,dimL,pm.T)) # bus voltage imag part
    
    # derivative of magnitude to each bus
    for t in range(pm.T):
        ### VOLTAGE MULTIPLIED WITH POWER RATIO
        # branch current real part
        v_iRe = (1/sol_vSq[:,t]*(sol_vRe[:,t] + pr[:,t]*sol_vIm[:,t]))
        
        # branch current imag part
        v_iIm = (1/sol_vSq[:,t]*(sol_vIm[:,t] - pr[:,t]*sol_vRe[:,t]))
        
        # bus voltage real part 
        v_vReR = (1/sol_vSq[:,t]*(sol_vRe[:,t] + pr[:,t]*sol_vIm[:,t]))
        v_vReX = (1/sol_vSq[:,t]*(pr[:,t]*sol_vRe[:,t] - sol_vIm[:,t]))
        
        # bus voltage imag part
        v_vImR = (1/sol_vSq[:,t]*(sol_vIm[:,t] - pr[:,t]*sol_vRe[:,t]))
        v_vImX = (1/sol_vSq[:,t]*(sol_vRe[:,t] + pr[:,t]*sol_vIm[:,t]))
        
        
        ### DERIVATIVES ###
        dIRe[:,:,t] = dt.bibc*mult_v(v_iRe)
        dIIm[:,:,t] = dt.bibc*mult_v(v_iIm)
        dVRe[:,:,t] = dt.rTil*mult_v(v_vReR) +\
                        dt.xTil*mult_v(v_vReX)
        dVIm[:,:,t] = dt.rTil*mult_v(v_vImR) +\
                        dt.xTil*mult_v(v_vImX)
      
    
    ### SENSITIVITY MATRIX ###
    # branch current
    # real part divided by magnitude
    iRe = np.array([[sol_iRe[i,t]/sol_iMag[i,t] if sol_iMag[i,t]!=0 else 0\
                     for t in range(pm.T)] for i in range(dimL)])
        
    # imag part divided by magnitude
    iIm = np.array([[sol_iIm[i,t]/sol_iMag[i,t] if sol_iMag[i,t]!=0 else 0\
                     for t in range(pm.T)] for i in range(dimL)])
        
    gmaI_full = np.array([iRe[:,t].reshape(dimL,1)*dIRe[:,:,t] +\
                          iIm[:,t].reshape(dimL,1)*dIIm[:,:,t]\
                         for t in range(pm.T)]).transpose(1,2,0)
    
    # bus voltage
    # real part divided by magnitude
    vRe = np.array([[sol_vRe[i,t]/sol_vMag[i,t] for t in range(pm.T)]\
                    for i in range(dimL)])
        
    # imag part divided by magnitude
    vIm = np.array([[sol_vIm[i,t]/sol_vMag[i,t] for t in range(pm.T)]\
                    for i in range(dimL)])
    
    gmaV_full = np.array([vRe[:,t].reshape(dimL,1)*dVRe[:,:,t] +\
                          vIm[:,t].reshape(dimL,1)*dVIm[:,:,t]\
                         for t in range(pm.T)]).transpose(1,2,0)
        
    ## REMOVE COLUMNS WITH NO PV PRESENT ##
    gmaI = np.delete(gmaI_full,pv.rowDel,1)
    gmaV = np.delete(gmaV_full,pv.rowDel,1)
    
    
    ###########################################################################
    ## UPDATED UNCERTAINTY MARGIN ## 
    ###########################################################################           
    normI = np.array([[np.sqrt(sum(np.square(np.dot(gmaI[i,:,t],\
                                                   pv.covSqrt[:,:,t,iLoad*dt.pvCase+iPV]))))\
                                    for t in range(pm.T)] for i in range(dimL)])
    normV = np.array([[np.sqrt(sum(np.square(np.dot(gmaV[i,:,t],\
                                                   pv.covSqrt[:,:,t,iLoad*dt.pvCase+iPV]))))\
                                    for t in range(pm.T)] for i in range(dimL)])
    
    ### UPDATE UNCERTAINTY MARGIN ###
    # bus voltage
    omV_upd = cc.icdfV[pm.N_PH:].reshape(dimL,1)*normV
    
    # branch current
    omI_upd = cc.icdfI.reshape(dimL,1)*normI
        
    ### RETURN ###
    return omV_upd.reshape(dimL,pm.T,1),omI_upd.reshape(dimL,pm.T,1)  