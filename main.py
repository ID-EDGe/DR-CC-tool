# -*- coding: utf-8 -*-
"""
Master Thesis Dominic Scotoni

Main File
"""
###############################################################################
## IMPORT PACKAGES & SCRIPTS ## 
###############################################################################
### PACKAGES ###
import time as tm

### SCRIPTS ###
import bfs_opf as bf
import monte_carlo as mc
import param as pm


t_tot = []
tm_tot = tm.time()

###############################################################################
## CC-BFS-OPF ## 
###############################################################################       
### RUN OPTIMIZATION ###
print('########################################\n\
  Start CC-BFS-OPF')
# iterate over load variations iLoad
for iLoad in range(len(pm.LOADCASE)):
    for iPV in range(len(pm.PVCASE)):
        t_tmp = tm.time() # time for bfs-opf
        solBFS_OPF = bf.cc_opf(iLoad,iPV) # run optimization
        t_tot.append(tm.time()-t_tmp)
    
        print('\
          BFS-OPF done in %.2f s\n\
########################################'%t_tot[-1])
        
        
        ###########################################################################
        ## MONTE-CARLO VALIDATION ## 
        ###########################################################################
        print('########################################\n\
          Monte-Carlo Validation ...')
        t_tmp = tm.time() # timing for monte-carlo (mc)
        solMC = mc.mc_bfs(solBFS_OPF, iLoad,iPV) # run mc
        t_tot.append(tm.time()-t_tmp)
        print('\
          Done in %.2f s\n\
########################################'%t_tot[-1])


###############################################################################
## END OF RUN ## 
###############################################################################
t_tot.append(tm.time()-tm_tot)
print('########################################\n\
    Run finished in %.2f s\n\
########################################'%(tm.time()-tm_tot))
        

