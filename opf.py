# -*- coding: utf-8 -*-
"""
Master Thesis Dominic Scotoni

OPF
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
### PACKAGES ###
import time as tm
import numpy as np
import gurobipy as gp
import sys

### SCRIPTS ###
import param as pm
import data as dt
from data import bus,sets,load,bess,gen,inverter,branch,cost,oltc,pv


###############################################################################
## OPF MODEL ## 
###############################################################################
def opf_solve(bfs_vRe,bfs_vIm, omI, omV, iLoad, iPV):
    # time measurement
    t_OPF = []
    t_s = tm.time()
    
    ###########################################################################   
    ## PREPARE DATA ## 
    ###########################################################################
    dimN = dt.n*pm.N_PH # dimension of n*p [buses*phases]
    dimL = dt.l*pm.N_PH # dimension of l*p [branches*phases]
    
    # reshape input arrays 
    bfs_vRe = bfs_vRe.reshape(dimN,pm.T)
    bfs_vIm = bfs_vIm.reshape(dimN,pm.T)
    bfs_vMag = np.abs(bfs_vRe + 1j*bfs_vIm)
    bfs_vSq = np.square(bfs_vMag) # square voltage magnitude from bfs
        
    ### VOLTAGE DROP BRANCHES ###
    bfs_dVRe = np.zeros((dimL,pm.T))
    bfs_dVIm = np.zeros((dimL,pm.T))
    for t in range(pm.T):
        for j in range(dt.l):
            for k in range(pm.N_PH):
                # real part
                bfs_dVRe[j*pm.N_PH+k,t] = bfs_vRe[pm.N_PH*branch.fbus[j]+k,t] -\
                                        bfs_vRe[pm.N_PH*branch.tbus[j]+k,t] 
                
                # imag part                        
                bfs_dVIm[j*pm.N_PH+k,t] = bfs_vIm[pm.N_PH*branch.fbus[j]+k,t] -\
                                        bfs_vIm[pm.N_PH*branch.tbus[j]+k,t] 
                                        
    ### AVERAGE VOLTAGE MAGNITUDE PER NODE ###
    vAvg = 1/pm.N_PH*np.array([[sum(bfs_vMag[i*pm.N_PH+j,t] for j in range(pm.N_PH))\
                                for t in range(pm.T)] for i in range(dt.n)])
                                        
            
    ###########################################################################   
    ## MODEL START ## 
    ###########################################################################
    m = gp.Model('OPF')
    
    
    ###########################################################################   
    ## VARIABLES ## 
    ###########################################################################
    t_tmp = tm.time() # time for variables
    inf_lb = -gp.GRB.INFINITY # no lower bound
    inf_ub = gp.GRB.INFINITY # no upper bound
    
    ### POWER ###
    # active power
    pR = m.addVars(dimN,pm.T, lb=0, ub=inf_ub, name='pR')
    pNet = m.addVars(dimN,pm.T, lb=inf_lb, ub=inf_ub, name='pNet')
    pInj = m.addVars(dimN,pm.T, lb=inf_lb, ub=inf_ub, name='pInj')
    
    # active power curtailment factor
    aCurt = m.addVars(dimN,pm.T, lb=0, ub=pm.FLGCURT, name='aCurt')
        
    # reactive power
    qR = m.addVars(dimN,pm.T, lb=inf_lb, ub=inf_ub, name='qR')
    qRAbs = m.addVars(dimN,pm.T, lb=0, ub=inf_ub, name='qRAbs')
    qInj = m.addVars(dimN,pm.T, lb=inf_lb, ub=inf_ub, name='qInj')
    qNet = m.addVars(dimN,pm.T, lb=inf_lb, ub=inf_ub, name='qNet')    
    
    # losses
    pLoss = m.addVars(dimL,pm.T, lb=inf_lb, ub=inf_ub, name='pLoss')
    pLossAbs = m.addVars(dimL,pm.T, lb=0, ub=inf_ub, name='pLossAbs')
    qLoss = m.addVars(dimL,pm.T, lb=inf_lb, ub=inf_ub, name='qLoss')
    
    
    ### BRANCH CURRENT ###
    iRe = m.addVars(dimL,pm.T, lb=inf_lb, ub=inf_ub, name='iRe')
    iIm = m.addVars(dimL, pm.T, lb=inf_lb, ub=inf_ub, name='iIm')
    
    
    ### BUS VOLTAGE ###
    vRe = m.addVars(dimN,pm.T, lb=inf_lb, ub=inf_ub, name='vRe')
    vIm = m.addVars(dimN,pm.T, lb=inf_lb, ub=inf_ub, name='vIm')
    vReRot = m.addVars(dimN,pm.T, lb=inf_lb, ub=inf_ub, name='vReRot')
    vImRot = m.addVars(dimN,pm.T, lb=inf_lb, ub=inf_ub, name='vImRot')
    dVRe = m.addVars(dimN,pm.T, lb=inf_lb, ub=inf_ub, name='dVRe')
    dVIm = m.addVars(dimN,pm.T, lb=inf_lb, ub=inf_ub, name='dVIm')
    vDev = m.addVars(dimN,pm.T, lb=inf_lb, ub=inf_ub, name='vDev')
    vDevAbs = m.addVars(dimN,pm.T, lb=0, ub=inf_ub, name='vDevAbs')
    vDevMax = m.addVars(dt.n,pm.T, lb=0, ub=inf_ub, name='vDevMax')
    vuf = m.addVars(dt.n,pm.T, lb=0, ub=inf_ub, name='vuf')
    
    
    ### LOADS ###
    pL = m.addVars(dimN,pm.T, lb=0, ub=inf_ub, name='pL')
    qL = m.addVars(dimN,pm.T, lb=inf_lb, ub=inf_ub, name='qL')
    pShed = m.addVars(dimN,pm.T, lb=0, ub=inf_ub, name='pShed')
    qShed = m.addVars(dimN,pm.T, lb=0, ub=inf_ub, name='qShed')
    sShift=  m.addVars(dimN,pm.T, lb=inf_lb, ub=inf_ub, name='sShift')
    sShiftAbs =m.addVars(dimN,pm.T, lb=0, ub=inf_ub, name='sShiftAbs')
    
    
    ### BESS ###
    eBat = m.addVars(dt.n,pm.T, lb=0, ub=inf_ub, name='eBat')
    pBat = m.addVars(dt.n,pm.T, lb=0, ub=inf_ub, name='pBat')
    pCh = m.addVars(dt.n,pm.T, lb=0, ub=inf_ub, name='pCh')
    pDis = m.addVars(dt.n,pm.T, lb=0, ub=inf_ub, name='pDis')
    pC = m.addVars(dimN,pm.T, lb=0, ub=inf_ub, name='pC')
    pD = m.addVars(dimN,pm.T, lb=0, ub=inf_ub, name='pD')
    pB = m.addVars(dimN,pm.T,lb=0, ub=inf_ub, name='pB')
    qB = m.addVars(dimN,pm.T, lb=inf_lb, ub=inf_ub, name='qB')
    qBAbs = m.addVars(dimN,pm.T, lb=0, ub=inf_ub, name='qBAbs')
    
    
    ### OLTC TRAFO ###
    tau = m.addVars(pm.N_PH,pm.T, lb=inf_lb, ub=inf_ub, vtype=gp.GRB.INTEGER, name='tau')
    tauN = m.addVars(pm.N_PH,pm.T-1, lb=inf_lb, ub=inf_ub,name='tauN')
    tauNAbs = m.addVars(pm.N_PH,pm.T-1, lb=0, ub=inf_ub, name='tauNAbs')
    
    ### OTHER ###
    # slack node
    pImport = m.addVars(pm.N_PH,pm.T, name='pImport')
    pExport = m.addVars(pm.N_PH,pm.T, lb=inf_lb, ub=0, name='pExport')
    
    t_OPF.append(tm.time()-t_tmp) # time for variables
    
    
    
    ###########################################################################   
    ## CONSTRAINTS ## 
    ###########################################################################
    t_tmp = tm.time() # time for constraints
    
    ###########################################################################   
    ## RENEWABLE GENERATORS ## 
    ###########################################################################
    def ren_generator(): 
        ### ACTIVE POWER ###
        m.addConstrs((pR[i,t] ==\
                      pv.pPV[i,t,iLoad*dt.pvCase+iPV]*(1-aCurt[i,t])\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                     name='pR')
           
        ### SLACK NODE ###
        # power curtailment
        m.addConstrs((aCurt[i,t] == 0\
                          for i in range(pm.N_PH) for t in range(pm.T)),\
                     name='aCurtSlack')
            

    ###########################################################################   
    ## RENEWABLE INERTER ## 
    ###########################################################################
    def ren_inverter():            
        ### REACTIVE POWER ###        
        # absolute value three-phase flow
        m.addConstrs((qRAbs[i,t] ==\
                      gp.abs_(qR[i,t])\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                     name='qRAbs')
            
        ### BOUNDS ###        
        # reactive power upper bound
        m.addConstrs((qR[i,t]*qR[i,t] + pR[i,t]*pR[i,t] <=\
                      inverter.capPV[i,iLoad*dt.pvCase+iPV]**2\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                     name='qRenUb')
            
        if pm.FLGPF == 1:
            # power factor upper bound
            m.addConstrs((qR[i,t] <=\
                          gen.prMax[i]*pR[i,t]\
                              for i in range(1,dimN) for t in range(pm.T)),\
                          name='pfMaxRen')
                
            # power factor lower bound
            m.addConstrs((qR[i,t] >=\
                          gen.prMin[i]*pR[i,t]\
                              for i in range(1,dimN) for t in range(pm.T)),\
                          name='pfMinRen')
                      
    
    ###########################################################################   
    ## BESS INVERTER ## 
    ###########################################################################    
    def bess_inverter():
        ### SINGLE PHASE OPERATION ###
        # charging per phase
        m.addConstrs((pC[i,t] ==\
                      pCh[np.floor_divide(i,pm.N_PH),t]*\
                          pm.LOADSHARE[np.mod(i,pm.N_PH)]\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                     name='pC')
            
        # discharging per phase
        m.addConstrs((pD[i,t] ==\
                      pDis[np.floor_divide(i,pm.N_PH),t]*\
                          pm.LOADSHARE[np.mod(i,pm.N_PH)]\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                     name='pD')
            
        # total active power per phase
        m.addConstrs((pB[i,t] ==\
                      pD[i,t] +\
                      pC[i,t]\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                     name='pB')
            
        ### REACTIVE POWER ###    
        # absolute value
        m.addConstrs((qBAbs[i,t] ==\
                      gp.abs_(qB[i,t])\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                     name='qBAbs')
            
        ### BOUNDS ###                
        # reactive power upper bound
        m.addConstrs((qB[i,t]*qB[i,t] + pB[i,t]*pB[i,t] <=\
                      inverter.capBat[i]**2\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                     name='qBUb')
            
        if pm.FLGPF == 1:
            # power factor upper bound
            m.addConstrs((qB[i,t] <=\
                          gen.prMax[i]*pB[i,t]\
                              for i in range(1,dimN) for t in range(pm.T)),\
                          name='pfMaxBat')
                
            # power factor lower bound
            m.addConstrs((qB[i,t] >=\
                          gen.prMin[i]*pB[i,t]\
                              for i in range(1,dimN) for t in range(pm.T)),\
                          name='pfMinBat')
        
    
    ###########################################################################   
    ## NET NODAL POWER INJECTION ## 
    ###########################################################################
    def nodal_balance():
        ### INJECTED CURRENT ###
        # active power 
        m.addConstrs((pInj[i,t] ==\
                      pR[i,t] +\
                      pD[i,t] -\
                      pC[i,t]\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                     name='pInj')
            
        # reactive power
        m.addConstrs((qInj[i,t] ==\
                      qR[i,t] +\
                      qB[i,t]\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                     name='qInj')
        
        ### NET NODAL POWER ###
        # active power
        m.addConstrs((pNet[i,t] ==\
                      pInj[i,t] -\
                      pL[i,t]\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                      name='pNet')
            
        # reactive power
        m.addConstrs((qNet[i,t] ==\
                      qInj[i,t] -\
                      qL[i,t]\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                     name='qNet')
        
    
    ###########################################################################   
    ### BRANCH CURRENT ### 
    ###########################################################################
    def branch_current():
        # real part
        m.addConstrs((iRe[i,t] ==\
                      gp.quicksum(dt.bibc[i,j]*\
                                  (bfs_vRe[j+pm.N_PH,t]/bfs_vSq[j+pm.N_PH,t]*pNet[j+pm.N_PH,t] +\
                                   bfs_vIm[j+pm.N_PH,t]/bfs_vSq[j+pm.N_PH,t]*qNet[j+pm.N_PH,t])\
                                      for j in dt.bibcNZero[i])\
                          for i in range(dimL) for t in range(pm.T)),\
                     name='iRe')
                          
        
        # imag part
        m.addConstrs((iIm[i,t] ==\
                      gp.quicksum(dt.bibc[i,j]*\
                                  (bfs_vIm[j+pm.N_PH,t]/bfs_vSq[j+pm.N_PH,t]*pNet[j+pm.N_PH,t] -\
                                   bfs_vRe[j+pm.N_PH,t]/bfs_vSq[j+pm.N_PH,t]*qNet[j+pm.N_PH,t])\
                                      for j in dt.bibcNZero[i])\
                          for i in range(dimL) for t in range(pm.T)),\
                     name='iIm')
        
            
        ### BOUNDS ###
        # branch current
        m.addConstrs((iRe[i,t]*iRe[i,t] + iIm[i,t]*iIm[i,t] <=\
                      np.square(branch.iBr_ub[i] - omI[i,t])\
                          for i in range(dimL) for t in range(pm.T)),\
                     name='iUb')    
     
        
    ###########################################################################   
    ### BUS VOLTAGE ### 
    ###########################################################################    
    def bus_voltage():
        ### VOLTAGE DIFFERENCE ###
        # real part
        m.addConstrs((dVRe[i,t] ==\
                      gp.quicksum(dt.rTil[i-pm.N_PH,j-pm.N_PH]*\
                                  (bfs_vRe[j,t]/bfs_vSq[j,t]*pNet[j,t] +\
                                   bfs_vIm[j,t]/bfs_vSq[j,t]*qNet[j,t])\
                                      for j in range(pm.N_PH,dimN)) +\
                      gp.quicksum(dt.xTil[i-pm.N_PH,j-pm.N_PH]*\
                                  (bfs_vRe[j,t]/bfs_vSq[j,t]*qNet[j,t] -\
                                   bfs_vIm[j,t]/bfs_vSq[j,t]*pNet[j,t])\
                                      for j in range(pm.N_PH,dimN))\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                     name='dVRe')
            
        # imag part
        m.addConstrs((dVIm[i,t] ==\
                      gp.quicksum(dt.rTil[i-pm.N_PH,j-pm.N_PH]*\
                                  (bfs_vIm[j,t]/bfs_vSq[j,t]*pNet[j,t] -\
                                   bfs_vRe[j,t]/bfs_vSq[j,t]*qNet[j,t])\
                                       for j in range(pm.N_PH,dimN)) +\
                      gp.quicksum(dt.xTil[i-pm.N_PH,j-pm.N_PH]*\
                                  (bfs_vRe[j,t]/bfs_vSq[j,t]*pNet[j,t] +\
                                   bfs_vIm[j,t]/bfs_vSq[j,t]*qNet[j,t])\
                                      for j in range(pm.N_PH,dimN))\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                     name='dVIm')
        
        ### VOLTAGE PHASOR ###
        # real part
        m.addConstrs((vRe[i,t] == \
                      bus.vSlackRe[i] +\
                      dVRe[i,t] +\
                      oltc.dVRe[i]*tau[np.mod(i,pm.N_PH),t]\
                          for i in range(dimN) for t in range(pm.T)),\
                      name='vRe')
             
        # imag part
        m.addConstrs((vIm[i,t] == \
                      bus.vSlackIm[i] +\
                      dVIm[i,t] +\
                      oltc.dVIm[i]*tau[np.mod(i,pm.N_PH),t]\
                          for i in range(dimN) for t in range(pm.T)),\
                      name='vIm')
            
        ### ROTATED VOLTAGE PHASOR ###
        # real part
        m.addConstrs((vReRot[i,t] ==\
                      bus.vSlack +\
                      oltc.dV*tau[np.mod(i,pm.N_PH),t] +\
                      bus.rotRe[i]*dVRe[i,t] -\
                      bus.rotIm[i]*dVIm[i,t]\
                          for i in range(dimN) for t in range(pm.T)),\
                     name='vReRot')
        
        # imag part
        m.addConstrs((vImRot[i,t] ==\
                      bus.rotRe[i]*dVIm[i,t] +\
                      bus.rotIm[i]*dVRe[i,t]\
                          for i in range(dimN) for t in range(pm.T)),\
                     name='vImRot')            
            
        ### VUF (ONLY FOR 3-PH) ###
        if pm.N_PH == 3:
            # voltage deviation from nodal average
            m.addConstrs((vDev[i,t] ==\
                      vReRot[i,t] -\
                      vAvg[np.floor_divide(i,pm.N_PH),t]\
                          for i in range(dimN) for t in range(pm.T)),\
                     name='vDev')
                           
            
            # absolute value voltage deviation from nodal average
            m.addConstrs((vDevAbs[i,t] ==\
                          gp.abs_(vDev[i,t])\
                              for i in range(dimN) for t in range(pm.T)),\
                         name='vDevAbs')
                
            # maximum absolute voltage deviation from nodal average
            m.addConstrs((vDevMax[i,t] ==\
                          gp.max_(vDevAbs[i*pm.N_PH,t],\
                                  vDevAbs[i*pm.N_PH+1,t],\
                                  vDevAbs[i*pm.N_PH+2,t])\
                              for i in range(dt.n) for t in range(pm.T)),\
                         name='vDevMax')
             
            # VUF
            m.addConstrs((vuf[i,t] ==\
                          vDevMax[i,t]/vAvg[i,t]\
                              for i in range(dt.n) for t in range(pm.T)),\
                         name='vuf')
                
                
            # VUF limit
            m.addConstrs((vuf[i,t] <=\
                          bus.vufMax[i]\
                              for i in range(dt.n) for t in range(pm.T)),\
                          name='vufUb')            
            
        ### SLACK BUS ###
        # real part
        m.addConstrs((dVRe[i,t] == 0\
                          for i in range(pm.N_PH) for t in range(pm.T)),\
                     name='vSlackRe')
            
        # imag part
        m.addConstrs((dVIm[i,t] == 0\
                          for i in range(pm.N_PH) for t in range(pm.T)),\
                     name='vSlackIm')
            
        ### BOUNDS ###
        # upper bound   
        m.addConstrs((vReRot[i,t]*vReRot[i,t] + vImRot[i,t]*vImRot[i,t] <=\
                      np.square(bus.vBus_ub[i] - omV[i-pm.N_PH,t])\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                      name='vUb')  
            
        # lower bound
        m.addConstrs((vReRot[i,t] >=\
                      bus.vBus_lb[i] + omV[i-pm.N_PH,t]\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                      name='vLb')
            
            
    ###########################################################################   
    ### OLTC ### 
    ###########################################################################
    def oltc_cnstr():
        if pm.FLGOLTC == 1:
            ### SYMMETRY (if applicable) ###
            if oltc.symmetry == 1 and pm.N_PH == 3:
                m.addConstrs((tau[i,t] ==\
                              tau[i+1,t]\
                                  for i in range(pm.N_PH-1) for t in range(pm.T)),\
                             name='oltcSymmetry')
                    
            ### MAXIMUM SWITCH ACTIONS ###
            # change in tap position per timestep
            m.addConstrs((tauN[i,t] ==\
                          tau[i,t] - tau[i,t+1]\
                              for i in range(pm.N_PH) for t in range(pm.T-1)),\
                         name='tauN')
            
            # absolute value change of tap position
            m.addConstrs((tauNAbs[i,t] ==\
                          gp.abs_(tauN[i,t])\
                              for i in range(pm.N_PH) for t in range(pm.T-1)),\
                         name='tauNAbs')
                
            # limit switch actions
            m.addConstrs((gp.quicksum(tauNAbs[i,t] for t in range(pm.T-1)) <=\
                          oltc.oltcSum\
                              for i in range(pm.N_PH)),\
                         name='oltcSwitchMax')
                
        ### BOUNDS ###
        # upper/lower bound
        m.addConstrs((tau[i,t] <=\
                      oltc.tauMax\
                          for i in range(pm.N_PH) for t in range(pm.T)),\
                     name='tauUb')
        m.addConstrs((tau[i,t] >=\
                      oltc.tauMin\
                          for i in range(pm.N_PH) for t in range(pm.T)),\
                     name='tauLb')
            
            
	###########################################################################   
    ### POWER LOSSES ### 
    ###########################################################################
    def power_losses():
        # active
        m.addConstrs((pLoss[i,t] ==\
                      iRe[i,t]*bfs_dVRe[i,t] +\
                      iIm[i,t]*bfs_dVIm[i,t]\
                          for i in range(dimL) for t in range(pm.T)),\
                      name='pLoss')
            
        # reactive
        m.addConstrs((qLoss[i,t] ==\
                      iRe[i,t]*bfs_dVIm[i,t] -\
                      iIm[i,t]*bfs_dVRe[i,t]\
                          for i in range(dimL) for t in range(pm.T)),\
                      name='qLoss')
            
        # absolute active power losses
        m.addConstrs((pLossAbs[i,t] ==\
                      gp.abs_(pLoss[i,t])\
                          for i in range(dimL) for t in range(pm.T)),\
                     name='pLossAbs')
        
    
    ###########################################################################   
    ### BESS DC SIDE ### 
    ###########################################################################    
    def bess_dc():
        ### SOC ###
        # init
        m.addConstrs((eBat[i,0] == pm.FLGBAT*sets.bat[i]*(\
                      bess.socInit[i]*bess.icBat[i] +\
                      pm.TIMESTEP*(bess.etaBat[i]*pCh[i,0] -\
                                    pDis[i,0]/bess.etaBat[i]))\
                          for i in range(1,dt.n)),\
                      name='socInit')
            
        # end
        m.addConstrs((eBat[i,pm.T-1] ==\
                      bess.socInit[i]*bess.icBat[i]\
                          for i in range(1,dt.n)),\
                      name='socEnd')
            
        # update
        m.addConstrs((eBat[i,t] == pm.FLGBAT*sets.bat[i]*(\
                      eBat[i,t-1] +\
                      pm.TIMESTEP*(bess.etaBat[i]*pCh[i,t] -\
                                    pDis[i,t]/bess.etaBat[i]))\
                          for i in range(1,dt.n) for t in range(1,pm.T)),\
                      name='socUpd')
        
        ### SIMULTANEOUS CHARGING/DISCHARGING ###
        # sum of charging and discharging
        m.addConstrs((pBat[i,t] == pm.FLGBAT*sets.bat[i]*(\
                      pCh[i,t] +\
                      pDis[i,t])\
                          for i in range(1,dt.n) for t in range(pm.T)),\
                      name='pBat')
            
        # sum equal to max of ch/dis
        m.addConstrs((pBat[i,t] ==\
                      gp.max_(pCh[i,t],pDis[i,t])\
                          for i in range(1,dt.n) for t in range(pm.T)),\
                      name='simChar')
            
        ### SLACK BUS ###
        # soc
        m.addConstrs((eBat[0,t] == 0\
                          for t in range(pm.T)),\
                      name='eBatSlack')
            
        # no charging/discharging
        m.addConstrs((pBat[0,t] == 0
                          for t in range(pm.T)),\
                      name='pBatSlack')
            
        ### BOUNDS ###
        # battery energy upper/lower
        m.addConstrs((eBat[i,t] <=\
                      bess.eBat_ub[i]\
                          for i in range(1,dt.n) for t in range(pm.T)),\
                     name='eBatUb')
        m.addConstrs((eBat[i,t] >=\
                      bess.eBat_lb[i]\
                          for i in range(1,dt.n) for t in range(pm.T)),\
                     name='eBatLb')
            
        # charging/discharging
        m.addConstrs((pCh[i,t] <=\
                      bess.pCh_ub[i]\
                          for i in range(1,dt.n) for t in range(pm.T)),\
                     name='pChUb')
        m.addConstrs((pDis[i,t] <=\
                      bess.pDis_ub[i]\
                          for i in range(1,dt.n) for t in range(pm.T)),\
                     name='pDisUb')
            
    
    ###########################################################################   
    ### LOADS ### 
    ###########################################################################      
    def loads():
        # active
        m.addConstrs((pL[i,t] ==\
                      load.pDem[i,t,iLoad]*(1 + pm.FLGSHIFT*sets.flxPhase[i]*\
                                            sShift[i,t]) -\
                      pm.FLGSHED*sets.flxPhase[i]*pShed[i,t]\
                          for i in range(dimN) for t in range(pm.T)),\
                      name='pL')
            
        # reactive
        m.addConstrs((qL[i,t] ==\
                      load.qDem[i,t,iLoad]*(1 + pm.FLGSHIFT*sets.flxPhase[i]*sShift[i,t]) -\
                      pm.FLGSHED*sets.flxPhase[i]*qShed[i,t]\
                          for i in range(dimN) for t in range(pm.T)),\
                     name='qL')
        
        # fullfill daily demand
        m.addConstrs((gp.quicksum(sShift[i,t] for t in range(pm.T)) == 0\
                      for i in range(pm.N_PH,dimN)),\
                      name='dailyDemand')
            
        # absolute value sShift
        m.addConstrs((sShiftAbs[i,t] ==\
                      gp.abs_(sShift[i,t])\
                          for i in range(dimN) for t in range(pm.T)),\
                     name='sShiftAbs')
            
            
        ### SLACK BUS ###
        # load shifting
        m.addConstrs((sShift[i,t] == 0\
                          for i in range(pm.N_PH) for t in range(pm.T)),\
                      name='sShiftSlack')
            
        # load shedding
        m.addConstrs((pShed[i,t] == 0\
                          for i in range(pm.N_PH) for t in range(pm.T)),\
                      name='pShedSlack')
            
        ### BOUNDS ###
        # load shedding active power
        m.addConstrs((pShed[i,t] <=\
                      load.pShed_ub[i,t,iLoad]\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                     name='pShedUb')
            
        # load shedding reactive power
        m.addConstrs((qShed[i,t] <=\
                      load.qShed_ub[i,t,iLoad]\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                     name='qShedUb')
        
        # load shifting upper/lower bound
        m.addConstrs((sShift[i,t] <=\
                      load.sShift_ub[i]\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                     name='sShiftUb')
        m.addConstrs((sShift[i,t] >=\
                      load.sShift_lb[i]\
                          for i in range(pm.N_PH,dimN) for t in range(pm.T)),\
                     name='sShiftLb')
        
        
    ###########################################################################   
    ### SLACK BUS ### 
    ###########################################################################
    def slack():
        ### CONNECTION TO UPPER LEVEL GRID ###
        # net active power
        m.addConstrs((pNet[i,t] ==\
                      - gp.quicksum(pNet[i+j*pm.N_PH,t] for j in range(1,dt.n)) +\
                      gp.quicksum(pLossAbs[i+j*pm.N_PH,t] for j in range(dt.l))\
                          for i in range(pm.N_PH) for t in range(pm.T)),\
                     name='pSlack')
            
        # active power absorption from slack node
        m.addConstrs((pImport[i,t] ==\
                      gp.max_(pNet[i,t],0)\
                          for i in range(pm.N_PH) for t in range(pm.T)),\
                     name='pSlackPos')
        
        # active power injection to slack node
        m.addConstrs((pExport[i,t] ==\
                      gp.min_(pNet[i,t],0)\
                          for i in range(pm.N_PH) for t in range(pm.T)),\
                     name='pSlackNeg')
            
        # net reactive power
        m.addConstrs((qNet[i,t] ==\
                      - gp.quicksum(qNet[i+j*pm.N_PH,t] for j in range(1,dt.n)) +\
                      gp.quicksum(qLoss[i+j*pm.N_PH,t] for j in range(dt.l))\
                          for i in range(pm.N_PH) for t in range(pm.T)),\
                     name='qSlack')
  
    ###########################################################################   
    ### WRITE CONSTRAINTS ### 
    ###########################################################################
    ren_generator()
    ren_inverter()
    bess_inverter()
    nodal_balance()
    branch_current()
    bus_voltage()
    oltc_cnstr()
    power_losses()
    bess_dc()
    loads()
    slack()
    
    
    # time for constraints
    t_OPF.append(tm.time()-t_tmp) 
    
    ###########################################################################   
    ## OBJECTIVE FUNCTION ## 
    ###########################################################################
    t_tmp = tm.time() # time for objective function
    
    obj = gp.quicksum(\
                      gp.quicksum(cost.loss[i]*pLossAbs[i,t]\
                                      for i in range(dimL)) +\
                      gp.quicksum(cost.bat[i]*pBat[i,t]\
                                      for i in range(1,dt.n)) +\
                      gp.quicksum(cost.curt[i]*aCurt[i,t]*\
                                      pv.pPV[i,t,iLoad*dt.pvCase+iPV] +\
                                  cost.qSupport[i]*qRAbs[i,t] +\
                                  cost.qSupport[i]*qBAbs[i,t] +\
                                  cost.shed[i]*pShed[i,t] +\
                                  cost.shed[i]*qShed[i,t] -\
                                  cost.shift[i]*sShiftAbs[i,t] +\
                                  cost.ren[i]*pR[i,t]\
                                      for i in range(pm.N_PH,dimN)) +\
                      gp.quicksum(cost.slackCost*pImport[i,t] +\
                                  cost.slackRev*pExport[i,t] +\
                                  cost.slackQ*qNet[i,t]\
                                      for i in range(pm.N_PH))\
                          for t in range(pm.T))
        
    m.setObjective(obj) # write obj to solver
                                  
    t_OPF.append(tm.time()-t_tmp) # time for objective function
    
    
    
    ###########################################################################   
    ## SOLVE ## 
    ###########################################################################
    t_tmp = tm.time() # time for solver
    m.optimize() # solve model
    t_OPF.append(tm.time()-t_tmp)
    
    # print result
    if m.status != gp.GRB.Status.OPTIMAL:
        m.computeIIS()
        m.write("rlt/model_iis.ilp")
        sys.exit("OPF FAILED")
    else:
        # save results to list
        sol = [(i.varName, i.x) for i in m.getVars()]
        objVal = obj.getValue()
        
    # write model to .lp-file
    m.write('rlt/model.lp')
    
    t_OPF.append(tm.time()-t_s) # time for OPF
    
    return sol,objVal,t_OPF






