# -*- coding: utf-8 -*-
"""
Master Thesis Dominic Scotoni

Data File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
### PACKAGES ###
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.linalg import block_diag
import pickle as pkl
import os

### SCRIPTS ###
import param as pm
import forecast as fcst

###############################################################################
## FUNCTIONS DEFINITIONS ## 
###############################################################################
### EXPORT / IMPORT SOLUTIONS TO PKL FILE ###
def sol_export(filename, data):
    rltDir = 'rlt/case%s_t%s_loadVar%s_pvVar%s_%s_cc%s_drcc%s_flx%s_%s_bat%s/'\
                %(pm.N_BUS,pm.T,pm.FLGVAR_LOAD,pm.FLGVAR_PV,pm.FCSTCASE[0],\
                  pm.FLGCC, pm.FLGDRCC,pm.FLGSHIFT,pm.UNBALANCE,pm.FLGBAT)
    if os.path.exists(rltDir):
        output = open(rltDir + filename + ".pkl", 'wb') # create output file
        pkl.dump(data, output) # write data to output file
        output.close() # close output file
    else:
        os.mkdir(rltDir) # create new directory
        output = open(rltDir + filename + ".pkl", 'wb') # create output file
        pkl.dump(data, output) # write data to output file
        output.close() # close output file
    
def sol_import(filename):
    rltDir = 'rlt/case%s_t%s_loadVar%s_pvVar%s_%s_cc%s_drcc%s_flx%s_%s_bat%s/'\
                %(pm.N_BUS,pm.T,pm.FLGVAR_LOAD,pm.FLGVAR_PV,pm.FCSTCASE[0],\
                  pm.FLGCC, pm.FLGDRCC,pm.FLGSHIFT,pm.UNBALANCE,pm.FLGBAT)
    file = open(rltDir + filename + ".pkl", 'rb') # open results file
    tmp = pkl.load(file) # create arry from file
    file.close() # close file
    
    return tmp


### SYMMETRIC INPUT DATA FOR N PHASES ###
def phase_multiplication(data):
    dim = len(data) # get dimension of input data
    phase = np.ones((pm.N_PH)) # array to multiply input with number of phases
    tmp = []
    for i in range(dim):
        tmp = np.append(tmp,data[i]*phase, axis=0)
        
    return tmp


###############################################################################
## READ DATA FROM CSV FILES ## 
###############################################################################    
def read_data(src):
    srcDir = 'src/case%s/'%pm.N_BUS 
    return pd.read_csv(srcDir + src +"Data.csv", delimiter=',') 

busData = read_data('bus')
branchData = read_data('branch')
costData = read_data('cost')
impData = read_data('imp')
loadData = read_data('load')
batData = read_data('bat')
genData = read_data('gen')
invData = read_data('inv')
oltcData = read_data('oltc')


###############################################################################
## AUXILIARX PARAMETERS ## 
###############################################################################
n = len(busData) # number of nodes
l = len(branchData) # number of branches
loadCase = len(pm.LOADCASE)
pvCase = len(pm.PVCASE)

vBase = busData.values[:,1] # base value voltage [kV]
zBase = (vBase*1e3)**2/(pm.S_BASE*1e6) # base value impedance [Ohm]
iBase = pm.S_BASE*1e6/(vBase[1:]*1e3) # base value current [A]


###############################################################################
## GRID DATA ## 
###############################################################################
### VOLTAGES ###
class bus:
    # reference voltage at slack node magnitude
    vSlack = busData.values[0,4] 
    
    # voltage phasors
    a = np.exp(1j*120*np.pi/180) # symmetrical components operator
    if pm.N_PH == 3:
        phasor_slack = np.array([1,a**2,a]) # slack voltage phasor
        phasor_rot = np.array([1,a,a**2]) # rotation phasor
    else:
        phasor_slack = np.array([1]) # slack voltage phasor
        phasor_rot = np.array([1]) # rotation phasor
    
    # slack voltage real & imag part
    vSlackRe = np.tile(vSlack*np.real(phasor_slack[0:pm.N_PH]),n) 
    vSlackIm = np.tile(vSlack*np.imag(phasor_slack[0:pm.N_PH]),n)
    
    # rotation of voltage phasor real & imag part
    rotRe = np.tile(np.real(phasor_rot[0:pm.N_PH]),n)
    rotIm = np.tile(np.imag(phasor_rot[0:pm.N_PH]),n)
    
    # VUF
    vufMax = busData.values[:,6] # maximum vuf [-]
    
    # bounds   
    vBus_ub = phase_multiplication(busData.values[:,3]) # upper bound  
    vBus_lb = phase_multiplication(busData.values[:,2]) # lower bound


### BRANCHES ###
class branch:
    # stacked impedance matrix
    def z_stack(config):
        zBr = np.zeros((l,pm.N_PH,pm.N_PH), dtype=complex) # pre-allocate
        length = branchData.values[:,5] # length of branch [km]
        data = (impData.values[:,1:].astype(float)) # impedance [Ohm/km]
        for k in range(l):
            idx = int(np.where(impData.values[:,0] == config[k])[0])
            tmp = data[idx:idx+pm.N_PH,:]/zBase[k+1] # array with R & X for branch k [p.u.]
            zBr[k,:,:] = np.array([[tmp[i,j] + 1j*tmp[i,j+1] for j in range(0,2*pm.N_PH,2)]\
                                    for i in range(pm.N_PH)])*length[k] # impedance
            
        return zBr
    
    fbus = branchData.values[:,2].astype(int) # from bus
    tbus = branchData.values[:,3].astype(int) # to bus 
    
    zBrStacked = z_stack(branchData.values[:,1]) # stacked impedance matrix
    
    zBr = block_diag(*zBrStacked) # (block) diagonal matrix with impedances
    rBr = np.real(zBr) # diagonal matrix with resistances
    xBr = np.imag(zBr) # diagonal matrix with reactances
    
    # bounds
    iBr_ub = phase_multiplication(branchData.values[:,4]/iBase) # thermal limit [p.u.]
    
    
### SETS OF NODES ###
class sets:
    bat = list(np.where(batData.values[:,1]>0,1,0)) # battery node
    flx = list(np.where(loadData.values[:,3]!=0,1,0)) # flexible loads
    flxPhase = list(phase_multiplication(flx).astype(int)) # extended for n phases
    ren = list(np.where(loadData.values[:,1]>0,1,0)) # renewable generators
    
    # list with location of sets
    def idx_list(data, rng):
        tmp = [i for i in range(rng) if data[i] == 1]
        return tmp
    
    idxRen = idx_list(phase_multiplication(ren), n*pm.N_PH) # set of PV Buses
    idxBat = idx_list(phase_multiplication(bat), n*pm.N_PH) # set of bat Buses
    idxFlx = idx_list(phase_multiplication(flx), n*pm.N_PH) # set of flexible loads Buses
    

### LOADS ###
class load:
    # normalize load profiles & assign to node
    def load_profile(i):
        profile = pd.read_csv('src/load_profiles/Load_profile_%s.csv'%i)
        load_max = np.max(profile.values[:,1]) # maximum load
        
        # normalized load profile
        profile_norm = (profile.values[:,1]/load_max).astype(float) 
        
        # discretized load profile into T steps
        nMeasure = int(24/pm.TIMESTEP)
        profile_disc = np.array([np.mean(profile_norm[j*int(pm.TIMESTEP*60):\
                                                      (j+1)*int(pm.TIMESTEP*60)])\
                                 for j in range(nMeasure)]) 
            
        ### TAKE VALUES FROM 12:00 +- T/2 ###
        t_middle = 1/pm.TIMESTEP*12
        t_start = int(t_middle - pm.T/2)
        t_end = int(t_middle + pm.T/2)
        
        # export
        profile_load = profile_disc[t_start:t_end]
        
        return profile_load
    
    
    # index of load profile
    profile = loadData.values[:,5].astype(int) 
    
    # peak load and power factor per node
    sPeak = loadData.values[:,1]/(pm.S_BASE*1e3)
    pf = loadData.values[:,2]
    
    # active & reactive power demand [p.u]
    pDem = np.zeros((n*pm.N_PH,pm.T,loadCase))
    qDem = np.zeros((n*pm.N_PH,pm.T,loadCase))
    for c in range(loadCase):
        for i in range(n):
            for j in range(pm.N_PH):
                if pm.FLGLOAD == 1:
                    # active power demand
                    pDem[i*pm.N_PH+j,:,c] = pm.LOADCASE[c]*pm.LOADSHARE[j]*\
                                            sPeak[i]*pf[i]*\
                                            load_profile(profile[i])
                                        
                    # reactive power demand
                    qDem[i*pm.N_PH+j,:,c] = pm.LOADCASE[c]*pm.LOADSHARE[j]*\
                                            sPeak[i]*np.sin(np.arccos(pf[i]))*\
                                            load_profile(profile[i])
                else:
                    pDem[i*pm.N_PH+j,:,c] = pm.LOADCASE[c]*pm.LOADSHARE[j]*\
                                            sPeak[i]*pf[i]
                    qDem[i*pm.N_PH+j,:,c] = pm.LOADCASE[c]*pm.LOADSHARE[j]*\
                                            sPeak[i]*np.sin(np.arccos(pf[i]))            
    
    # bounds
    # max/min load shifting
    sShift_ub = pm.FLGSHIFT*phase_multiplication(sets.flx*loadData.values[:,4]) 
    sShift_lb = pm.FLGSHIFT*phase_multiplication(sets.flx*loadData.values[:,3])
    
    # load shedding
    pShed_ub = pm.FLGSHED*pDem
    qShed_ub = pm.FLGSHED*qDem
    
    
### BESS ###
class bess: 
    icBat = pm.FLGBAT*batData.values[:,1]/pm.S_BASE # installed capacity [p.u.]
    etaBat = batData.values[:,2] # efficiency
    socMin = batData.values[:,3] # soc min 
    socMax = batData.values[:,4] # soc max
    socInit = batData.values[:,5] # initial soc
    e2p = batData.values[:,6] # energy-to-power ratio [MWh/MW]
    
    # bounds
    pCh_ub = pm.FLGBAT*(sets.bat*icBat/e2p) # battery charging
    pDis_ub = pm.FLGBAT*(sets.bat*icBat/e2p) # battery discharging
    eBat_ub = icBat*socMax # soc max 
    eBat_lb = icBat*socMin # soc min


### GENERATORS ###
class gen:
    ### IC PV EITHER FROM INPUT DATA OR FACTOR OF PEAK LOAD ###
    if pm.FLGPV == 0:
        # from input data  - installed capacity [p.u.]
        icPV = []
        for i in range(loadCase):
            for j in range(pvCase):
                icPV.append(pm.PVCASE[j]*genData.values[:,1]/pm.S_BASE)
        icPV = np.array(icPV).transpose()
    else:
        # dependent on load
        icPV = [] # installed capacity [p.u.]
        for i in range(loadCase):
            for j in range(pvCase):
                icPV.append(pm.PVCASE[j]*pm.LOADCASE[i]*load.sPeak)
        
        # create array from list
        icPV = np.array(icPV).transpose()
        
    pfMax = phase_multiplication(genData.values[:,2]) # maximum power factor cos(phi)
    pfMin = -phase_multiplication(genData.values[:,2]) # minimum power factor cos(phi)
    
    prMax = np.sqrt((1-pfMax**2)/pfMax**2) # maximum power ratio gamma
    prMin = -np.sqrt((1-np.square(pfMin))/np.square(pfMin)) # minimum power ratio gamma


### INVERTERS ###
class inverter:
    def phase_selection(data,phase):
        dim = len(data) # get dimension of input data
        nPhase = np.ones((pm.N_PH)) # array to multiply input with number of phases
        tmp = []
        for i in range(dim):
            if phase[i] == 3:
                tmp = np.append(tmp,data[i]*nPhase/pm.N_PH, axis=0)
            else:
                tmp = np.append(tmp,np.zeros((pm.N_PH)), axis=0)
                tmp[i*pm.N_PH + phase[i]] = data[i]
                                
                    
        return tmp
    
    phase_pv = invData.values[:,3].astype(int) # to which phases PV is connected to    
    phase_bat = invData.values[:,4].astype(int) # to which phases bat is connected to
    
    # maximum renewable inverter capacity [p.u]
    capPV = []
    for c in range(pvCase*loadCase):
        capPV.append(phase_selection(invData.values[:,1]*gen.icPV[:,c],phase_pv))
    capPV = np.array(capPV).transpose()
     
    # maximum bat inverter capacity [p.u.]        
    capBat = phase_selection(invData.values[:,2]*bess.icBat/bess.e2p,phase_bat) 


### COSTS ###
class cost:
    def cost_pu(data):
        # calculate costs in [euro/p.u.] and per timestep
        return data*pm.TIMESTEP*pm.S_BASE
    
    curt = cost_pu(phase_multiplication(costData.values[:,1])) # active power curtailment
    ren = cost_pu(phase_multiplication(costData.values[:,2])) # renewable energy source
    bat = cost_pu(costData.values[:,3]) # battery
    shed = cost_pu(phase_multiplication(costData.values[:,4])) # load shedding
    shift = cost_pu(phase_multiplication(costData.values[:,5])) # load shifting
    qSupport = cost_pu(phase_multiplication(costData.values[:,6])) # reactive power injection
    loss = cost_pu(phase_multiplication(costData.values[:-1,7])) # active power losses
    slackRev = cost_pu(costData.values[0,8]) # revenue for selling to upper level grid
    slackCost = cost_pu(costData.values[0,9]) # active power from upper level grid
    slackQ = cost_pu(costData.values[0,10]) # reactive power from upper level grid    
    
    
### OLTC TRAFO ###
class oltc:
    oltc_min = oltcData.values[:,1] # minimum value [p.u.]
    oltc_max = oltcData.values[:,2] # maximum value [p.u.]
    oltc_steps = oltcData.values[:,3] # number of steps [-]
    oltcSum = int(oltcData.values[:,4]) # max number of shifts per time horizon [-]
    symmetry = int(oltcData.values[:,5]) # symmetric = 1, asymmetric = 0
    
    # voltage difference per shift [p.u.]
    dV = float((oltc_max - oltc_min)/oltc_steps) 
    dVRe = dV*bus.vSlackRe # real part [p.u.]
    dVIm = dV*bus.vSlackIm # imag part [p.u.]
    
    # bound
    tauMax = int(pm.FLGOLTC*(oltc_steps/2))
    tauMin = int(pm.FLGOLTC*(-oltc_steps/2))
    
    
###############################################################################
## PV FORECAST ## 
###############################################################################
class pv:
    def pv_phase(data,phase):
        dim = len(data) # get dimension of input data
        nPhase = np.array(pm.PVSHARE) # array to multiply input with number of phases
        tmp = []
        for i in range(dim):
            if phase[i] == 3:
                tmp = np.append(tmp,data[i]*nPhase, axis=0)
            else:
                tmp = np.append(tmp,np.zeros((pm.N_PH)), axis=0)
                tmp[i*pm.N_PH + phase[i]] = data[i]
                                
                    
        return tmp
    
    ### CHECK IF FORECAST FILE EXISTS ###
    fcstFile = 'src/fcst/forecastPV_v%s_%s_t%s.pkl'%(pm.V_FCST,pm.FCSTCASE[0],pm.T)
    if os.path.exists(fcstFile):
        ### READ FCST FILE ###
        file = open(fcstFile, 'rb') # open results file
        pvFcst = pkl.load(file) # create arry from file
        file.close() # close file
    else:
        ### RUN FORECAST ###
        print('Run forecasting script ...')
        pvFcst = fcst.pv_fcst()
        print('... done!')
        

    nSamples = np.size(pvFcst[3],1) # number of samples
    dataFcst = pvFcst[3] # all forecast data
        
    # installed capacity per phase
    icPhase = np.zeros((l*pm.N_PH,loadCase*pvCase))
    for c in range(loadCase*pvCase):
        icPhase[:,c] = pv_phase(gen.icPV[1:,c],inverter.phase_pv[1:])
        
    # forecasted PV infeed per phase
    pPV = np.zeros((n*pm.N_PH,pm.T,pvCase*loadCase))
    for c in range(pvCase*loadCase):
        pPV[:,:,c] = np.append(np.zeros((pm.N_PH,pm.T)),\
                               np.dot(icPhase[:,c].reshape(l*pm.N_PH,1),\
                                      pvFcst[0].reshape(1,pm.T)),axis=0)

    
    ### COVARIANCE MATRIX ###
    # covariance matrix for all timesteps
    cov = np.zeros((l*pm.N_PH,l*pm.N_PH,pm.T,pvCase*loadCase))
    for c in range(pvCase*loadCase):
        for t in range(pm.T):        
            # full covariance matrix
            cov[:,:,t,c] = np.cov(np.dot(icPhase[:,c].reshape(l*pm.N_PH,1),\
                                         dataFcst[t,:].reshape(1,nSamples)))

    # delete empty columnds and rows
    rowDel = []
    for i in range(l*pm.N_PH):
        if np.all(cov[i,:,int(pm.T/2),:] == 0):
            rowDel.append(i)
    covRed = np.delete(cov,rowDel,1)
    covRed = np.delete(covRed,rowDel,0)
    
    
    # independent measurements
    nMeas = np.size(covRed,axis=0)
    covInd = np.zeros((nMeas,nMeas,pm.T,pvCase*loadCase))
    for i in range(nMeas):
        for j in range(nMeas):
            for t in range(pm.T):
                for c in range(pvCase*loadCase):
                    if i != j:
                        covInd[i,j,t,c] = 0
                    else:
                        covInd[i,j,t,c] = covRed[i,j,t,c]
    
    # sqrt of covariance matrix
    covSqrt = np.sqrt(covInd) 
    # covSqrt = np.sqrt(covRed)    

    

###############################################################################
## BIBC MATRIX l x (n-1) | l branches, n buses ## 
###############################################################################     
### FULL BIBC-MATRIX ###
bibc = np.zeros((pm.N_PH*l,pm.N_PH*(n-1))) # pre-allocate matrix

# algorithm to generate bibc matrix
for i in range(l):
    if i == 0:
        bibc[i:pm.N_PH,i:pm.N_PH] = np.identity(pm.N_PH)
    else: 
        bibc[:,(branch.tbus[i]-1)*pm.N_PH:(branch.tbus[i])*pm.N_PH] =\
            bibc[:,(branch.fbus[i]-1)*pm.N_PH:(branch.fbus[i])*pm.N_PH]
        bibc[i*pm.N_PH:(i+1)*pm.N_PH,\
              (branch.tbus[i]-1)*pm.N_PH:(branch.tbus[i])*pm.N_PH] = np.identity(pm.N_PH)
            
### DELETE ENTRIES WHERE NO PHASES PRESENT ###
# list with phases not present
del_phase = sets.idx_list([1 if branch.zBr[i,i] == 0 else 0 \
             for i in range(l*pm.N_PH)],l*pm.N_PH)
    
# delete
bibc[:,del_phase] = 0
        

    

#### CREATE LIST WITH INDICES OF NON-ZERO ELEMENTS OF BIBC ####
bibcNZero = []
for i in range(l*pm.N_PH):
    bibcNZero.append([i for i, j in enumerate(bibc[i,:]) if j != 0])


###############################################################################
## BCBV MATRIX (n-1) x l | n buses, l branches ## 
###############################################################################
bcbv = np.dot(np.transpose(bibc),branch.zBr) # bcbv matrix 
bibv = np.dot(bcbv,bibc) # bibv matrix

# R tilde & X tilde for vBus
rTil = np.real(bibv)
xTil = np.imag(bibv)


###############################################################################
## CHANCE CONSTRAINTS ## 
###############################################################################
class cc:   
    ### VIOLATION PROBABILITIES ###
    eps_I = branchData.values[:,6] # violation probability branch current
    eps_V = busData.values[:,5] # violation probability bus voltage
    conLevel_I = 1-eps_I # confidence level branch current
    conLevel_V = 1-eps_V # confidence level bus voltage
    if pm.FLGDRCC == 0:
        # inverse cummulative distribution function - normal distribution
        icdfI = phase_multiplication(norm.ppf(conLevel_I)) # branch current
        icdfV = phase_multiplication(norm.ppf(conLevel_V)) # bus voltage
    elif pm.FLGDRCC == 1:
        # distributionally robust - assume known mean & covariance
        icdfI = phase_multiplication(np.sqrt(conLevel_I/eps_I))
        icdfV = phase_multiplication(np.sqrt(conLevel_V/eps_V))
        
        # # distributionally robust - assume unimodality
        # icdfI = phase_multiplication(np.sqrt(4/(9*eps_I) -1))
        # icdfV = phase_multiplication(np.sqrt(4/(9*eps_V) -1))


                    









