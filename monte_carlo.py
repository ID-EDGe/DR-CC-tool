# -*- coding: utf-8 -*-
"""
Master Thesis Dominic Scotoni

Monte-Carlo Validation
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
### PACKAGES ###
import numpy as np

### SCRIPTS ###
import param as pm
import data as dt
import results as rlt
import bfs as bfs
from data import bus,branch,pv


###############################################################################
## MONTE-CARLO VALIDATION ## 
###############################################################################
def mc_bfs(solBFS_OPF, iLoad, iPV):
    dimN = dt.n*pm.N_PH
    dimL = dt.l*pm.N_PH
    
    ### PV FORECAST ###
    nSamples = np.size(pv.pvFcst[2],1) # number of samples
    
    # normalized measurements multiplied with installed capacity
    pvReal = np.append(np.zeros((pm.N_PH,pm.T,nSamples)),\
                       np.array([[[pv.pvFcst[2][t,s]*pv.icPhase[i,iLoad*dt.pvCase+iPV]\
                         for s in range(nSamples)] for t in range(pm.T)]\
                       for i in range(dimL)]),axis=0)
    
    ### LOAD RESULTS ###
    solOPF = solBFS_OPF[0][0]
    solBFS = solBFS_OPF[1]
    
    ### DECISION VARIABLES FROM OPF ###
    # renewable inverter
    opf_aCurt = rlt.read_out_3ph('aCurt', dt.n, solOPF) # curtailment
    opf_qR = rlt.read_out_3ph('qR', dt.n, solOPF) # set point inverter
    
    # BESS inverter
    opf_pC = rlt.read_out_3ph('pC', dt.n, solOPF) # charging per phase
    opf_pD = rlt.read_out_3ph('pD', dt.n, solOPF) # discharging per phase
    opf_qB = rlt.read_out_3ph('qB', dt.n, solOPF) # set point inverter

    # load
    opf_pL = rlt.read_out_3ph('pL', dt.n, solOPF) # active load
    opf_qL = rlt.read_out_3ph('qL', dt.n, solOPF) # reactive load
    
    
    # OLTC
    opf_tau = rlt.read_out('tau',pm.N_PH, solOPF) # OLTC tap position
        
    
    ### RUN MC ###
    # initialize arrays for result
    mc_vMag = np.zeros((dimN, pm.T, nSamples)) # voltage magnitude
    mc_iMag = np.zeros((dimL, pm.T, nSamples)) # branch current magnitude
    n_VUp = np.zeros((dimN,pm.T)) # number of upper voltage violations
    n_VLow = np.zeros((dimN,pm.T)) # number of lower voltage violations
    n_iUp = np.zeros((dimL,pm.T)) # number of branch current violations
    
    # net nodal power injections
    mc_P = np.zeros((dimN,pm.T,nSamples)) # active power
    mc_Q = np.zeros((dimN,pm.T,nSamples)) # reactive power
    mc_pNet = np.zeros((dimN,pm.T,nSamples)) # net reactive power
    mc_qNet = np.zeros((dimN,pm.T,nSamples)) # net reactive power
    
    # run bfs for each sample and timestep
    for t in range(pm.T):
        for s in range(nSamples):            
            # active power PV            
            mc_P[:,t,s] = np.array([(1-opf_aCurt[i,t])*\
                                    pvReal[i,t,s]\
                                    for i in range(dimN)])
            # reactive power according to optimal set point
            # reactive power output fixed --> changing power factor
            mc_Q[:,t,s] = opf_qR[:,t]
            
            # net active power
            mc_pNet[:,t,s] =\
                mc_P[:,t,s] -\
                opf_pL[:,t] -\
                opf_pC[:,t] +\
                opf_pD[:,t]
            
            # net reactive power
            mc_qNet[:,t,s] = mc_Q[:,t,s] -\
                        opf_qL[:,t] +\
                        opf_qB[:,t]
                        
            # total net nodal apparent power injection
            mc_sNet = (mc_pNet[pm.N_PH:,t,s] + 1j*mc_qNet[pm.N_PH:,t,s]).reshape(dimL,1)
            
            # run BFS
            solBFS = bfs.bfs_solve(mc_sNet, 1, opf_tau)
            
            # results
            mc_vRe = solBFS[0] 
            mc_vIm = solBFS[1]
            
            mc_iRe = solBFS[2]
            mc_iIm = solBFS[3]
                      
            # update voltage & branch current magnitude
            mc_vMag[:,t,s] = np.absolute(mc_vRe + 1j*mc_vIm).reshape(dimN)
            mc_iMag[:,t,s] = np.absolute(mc_iRe + 1j*mc_iIm).reshape(dimL)
            
            # check for bounds violations
            n_VLow[:,t] += mc_vMag[:,t,s] < bus.vBus_lb[:] # voltage lower bound
            n_VUp[:,t] += mc_vMag[:,t,s] > bus.vBus_ub[:] # voltage upper bound
            n_iUp[:,t] += mc_iMag[:,t,s] > branch.iBr_ub[:] # branch current upper bound
            
    ###########################################################################
    ## SAVE & RETURN RESULTS ## 
    ###########################################################################
    solMC = [mc_vMag, mc_iMag,n_iUp,n_VLow,n_VUp]
        
    ### SAVE TO PKL FILE    
    dt.sol_export('mcr_load%s_pv%s'%(iLoad,iPV),solMC)
    
    
    return solMC

