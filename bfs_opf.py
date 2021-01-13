# -*- coding: utf-8 -*-
"""
Master Thesis Dominic Scotoni

BFS-OPF File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
### PACKAGES ###
import numpy as np

### SCRIPTS ###
import param as pm
import opf as opf
import data as dt
from data import bus,branch
import results as rlt
import bfs as bfs
import cc_margin as marg

###############################################################################
## CHANCE CONSTRAINTS ## 
###############################################################################
def cc_opf(iLoad,iPV):
    dimL = dt.l*pm.N_PH # dimension of all branches * phases
    ###########################################################################
    ## INITIALIZE ## 
    ###########################################################################
    ### OUTER UNCERTAINTY MARGIN LOOP ###
    m = 0 # iteration counter
    
    errMaxI = 1 # initial error for branch current uncertainty margin
    errMaxV = 1 # initial error for upper bus voltage uncertainty margin
    
    # set initial margin to 0
    omI = np.zeros((dimL,pm.T,1)) # branch current
    omV = np.zeros((dimL,pm.T,1)) # bus voltage upper margin
    
    ### FLAT START BFS-OPF ###
    sol_vRe = np.array([bus.vSlackRe,]*pm.T).transpose()
    sol_vIm = np.array([bus.vSlackIm,]*pm.T).transpose()
    
    ###########################################################################
    ## RUN CC-BFS-OPF ## 
    ###########################################################################
    while errMaxV >= pm.ETA_MARG_V:
        #######################################################################
        ## BFS-OPF ## 
        #######################################################################
        # solve BFS-OPF
        solBFSOPF = bfs_opf(sol_vRe, sol_vIm, omI, omV, m, iLoad, iPV) 
        
        # results from BFS-OPF
        sol_vRe = solBFSOPF[1][0] # bfs bus voltage real part
        sol_vIm = solBFSOPF[1][1] # bfs bus voltage imag part        

        #######################################################################
        ## UPDATE UNCERTAINTY MARGIN ## 
        #######################################################################   
        if pm.FLGCC == 1:
            # updated uncertainty margins
            omUpd = marg.cc_upd(solBFSOPF,iLoad,iPV)
            omUpd_V = omUpd[0] # bus voltage
            omUpd_I = omUpd[1] # branch current
            
            # limit margin to avoid overlapping boundaries
            if np.any(omUpd_V > 2.5e-2):
                omUpd_V[omUpd_V > 2.5e-2] = 2.5e-2
            for i in range(dimL):
                for t in range(pm.T):
                    if omUpd_I[i,t] > branch.iBr_ub[i]:
                        omUpd_I[i,t] = branch.iBr_ub[i] - 3e-1
            
            
            ### SMOOTHLY UPDATE MARGIN TO AVOID JUMPING BETWEEN SOLUTIONS ###
            if pm.MARGUPD == 1: 
                omUpd_I = pm.A_MARG*omUpd_I + (1-pm.A_MARG)*omI[:,:,-1]\
                            .reshape(dimL,pm.T,1)
                omUpd_V = pm.A_MARG*omUpd_V + (1-pm.A_MARG)*omV[:,:,-1]\
                            .reshape(dimL,pm.T,1)
        else:
            # no uncertainty margin for deterministic case
            omUpd_V = np.zeros((dimL,pm.T,1))
            omUpd_I = np.zeros((dimL,pm.T,1))
        # bus voltage
        omV = np.append(omV, omUpd_V, axis=2)
        
        # branch current
        omI = np.append(omI, omUpd_I, axis=2)        
        
        #######################################################################
        ## EVALUATE CONVERGENCE CRITERION ## 
        #######################################################################   
        errMaxI = np.max(np.abs(omI[:,:,-1] - omI[:,:,-2])) # branch current
        errMaxV = np.max(np.abs(omV[:,:,-1] - omV[:,:,-2])) # bus voltage          
        if m >= pm.M_MAX:
            errMaxV = 0
            print('NO CONVERGENCE ON UNCERTAINTY MARGIN')
        elif m < pm.M_MIN:
            if pm.FLGCC == 1:
                errMaxV = 1 # miminum m iterations for uncertainty margin
            else:
                errMaxV = 0 # no CC
        else:
            print('Branch current maximum change uncertainty margin: %.3e' %errMaxI)
            print('Bus voltage maximum change uncertainty margin: %.3e' %errMaxV)
        
        ### UPDATE ITERATION COUNTER ###
        m += 1
    
        
    ###########################################################################
    ## SAVE & RETURN RESULTS ## 
    ###########################################################################    
    solCC = [omI,omV] # store results from outer cc-loop
    
    ### SAVE RESULTS ###
    
    # OPF
    dt.sol_export('opf_load%s_pv%s'%(iLoad,iPV),solBFSOPF[0]) 
    # BFS
    dt.sol_export('bfs_load%s_pv%s'%(iLoad,iPV),solBFSOPF[1])
        
    # chance-constraints      
    dt.sol_export('ccs_load%s_pv%s'%(iLoad,iPV),solCC)
    
    ### RETURN ###
    return solBFSOPF


###############################################################################
## BFS-OPF ## 
###############################################################################
def bfs_opf(sol_vRe, sol_vIm, omI, omV, m, iLoad, iPV):
    t_BFSOPF = []
    dimN = dt.n*pm.N_PH # dimension of buses * phases
    dimL = dt.l*pm.N_PH # dimension of branches * phases
    
    # reshape input arrays
    omI = omI[:,:,-1]
    omV = omV[:,:,-1]
    
    ###########################################################################
    ## INITIALIZE ## 
    ###########################################################################
    ### BFS-OPF ###
    b = 0 # iteration counter bfs-opf
    bfsError = 1 # initial bfs error
    
    # voltages from previous outer uncertainty margin loop
    bfs_vRe = sol_vRe.reshape(dimN,pm.T,1)
    bfs_vIm = sol_vIm.reshape(dimN,pm.T,1)
    
    # results for opf
    opf_vRe = (np.array([bus.vSlackRe,]*pm.T).transpose()).reshape(dimN,pm.T,1)
    opf_vIm = (np.array([bus.vSlackIm,]*pm.T).transpose()).reshape(dimN,pm.T,1)
        
    ###########################################################################
    ## BFS-OPF ## 
    ###########################################################################
    while bfsError > pm.ETA_BFSOPF:
        print("\
--------------------\n\
Margin Iteration %s - BFS Iteration %s" %(m,b))
        
        #######################################################################
        ## OPF ## 
        #######################################################################
        ### SOLVE OPF ###
        solOPF = opf.opf_solve(bfs_vRe[:,:,-1],bfs_vIm[:,:,-1], omI, omV,\
                               iLoad, iPV) 
        t_BFSOPF.append(solOPF[2]) # time for OPF
        
        ### READ OUT RESULTS ###
        # net power
        opf_pNet = rlt.read_out('pNet', dimN, solOPF[0])
        opf_qNet = rlt.read_out('qNet', dimN, solOPF[0])
        sNet = opf_pNet[pm.N_PH:,:] + 1j*opf_qNet[pm.N_PH:,:]            
            
        # bus voltages
        opf_vRe = np.append(opf_vRe,\
                            rlt.read_out('vRe', dimN, solOPF[0]).reshape(dimN,pm.T,1),\
                                axis=2)
        opf_vIm = np.append(opf_vIm,\
                            rlt.read_out('vIm', dimN, solOPF[0]).reshape(dimN,pm.T,1),\
                                axis=2)
        opf_vMag = np.abs(opf_vRe + 1j*opf_vIm) # voltage magnitude
        
        
        # OLTC tap position 
        opf_tau = rlt.read_out('tau',pm.N_PH,solOPF[0]) 
        
        #######################################################################
        ## BFS ## 
        #######################################################################
        if pm.BFSUPD == 1:
            ### SMOOTHLY UPDATE SNET TO AVOID JUMPING BETWEEN SOLUTIONS ###
            if b == 0:
                sNet_Upd = sNet.reshape(dimL,pm.T,1)
            else:
                sNet = pm.A_BFS*sNet + (1-pm.A_BFS)*sNet_Upd[:,:,-1]
                sNet_Upd = np.append(sNet_Upd,sNet.reshape(dimL,pm.T,1), axis=2)
            
        ### RUN BFS ###
        solBFS = bfs.bfs_solve(sNet, pm.T, opf_tau)
        t_BFSOPF.append(solBFS[4])
        
        ### READ OUT RESULTS ###
        bfs_vRe = np.append(bfs_vRe, solBFS[0].reshape(dimN,pm.T,1), axis=2)
        bfs_vIm = np.append(bfs_vIm, solBFS[1].reshape(dimN,pm.T,1), axis=2)
        bfs_vMag = np.abs(bfs_vRe + 1j*bfs_vIm) # voltage magnitude
        
        
        #######################################################################
        ## VOLTAGE MISMATCH & CHECK FOR CONVERGENCE ## 
        #######################################################################
        if b >= pm.B_MAX:
            # sys.exit("NO BFS-OPF CONVERGENCE")
            bfsError = 0
            print('NO BFS-OPF CONVERGENCE')
        elif b < 1 and m == 0:
            bfsError = 1 # minimum 2 iterations for first CC iteration
            print('Voltage magnitude mismatch: %.4e\n' \
                  %(np.max(np.abs(bfs_vMag[:,:,-1] - opf_vMag[:,:,-1]))))
        else:
            bfsError = np.max(np.abs(bfs_vMag[:,:,-1] - opf_vMag[:,:,-1]))
            print('Voltage magnitude mismatch: %.4e\n' %bfsError)
        
        # update iteration counter
        b += 1
        
    ###########################################################################
    ## RETURN RESULTS ## 
    ###########################################################################        
    return solOPF,solBFS,t_BFSOPF








