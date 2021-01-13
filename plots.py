# -*- coding: utf-8 -*-
"""
Master Thesis Dominic Scotoni

Plots File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
### PACKAGES ###
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl

### SCRIPTS ###
import param as pm
import data as dt
import results as rlt

###############################################################################
## READ RESULTS FROM SOLUTION FILE & POST-PROCESSING ## 
###############################################################################
dimN = dt.n*pm.N_PH # dimension of all buses * phases
dimL = dt.l*pm.N_PH # dimension of all branches * phases
nCase = dt.loadCase*dt.pvCase # number of cases



###############################################################################
## EXAMPLE FOR ACTIVE POWER CUTAILMENT ## 
###############################################################################
# import solution from OPF
solOPF = (dt.sol_import('opf'))

# active power curtailment per node, phase, time step
opf_aCurt = rlt.read_out_phase('aCurt', dt.n, solOPF)


# plot for all buses
for p in range(pm.N_PH):
    fig,ax = plt.subplots(tight_layout=True)
    for i in range(dt.n):
        ax.plot(opf_aCurt[i,p,:], label='Bus %s'%i)
    ax.set_xlabel('Time Step')
    plt.title('Curtailment Phase %s'%p)
    plt.legend(loc='upper right')
    plt.show()
    plt.close()

    