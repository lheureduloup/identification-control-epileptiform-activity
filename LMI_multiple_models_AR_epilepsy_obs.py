# -*- coding: utf-8 -*-
"""
@author: João Angelo Ferres Brogin

"""

import numpy as np
import picos as pic
import cvxopt as cvx
from multiple_models_AR_epilepsy import A_cont

A_base = A_cont
n_models = len(A_cont)
Nm = len(A_cont[0])

#%% Optimization problem:
prob = pic.Problem()
                        
alpha = 0
A_base_cvx = []     
CC = np.zeros((1,Nm))

CC[0][0] = 1

I = np.identity(Nm)              
C = pic.new_param('C', cvx.matrix(CC))
II = pic.new_param('I', cvx.matrix(I))

for op in range(0, n_models):
    Am = A_base[op]
    A_base_cvx.append( pic.new_param('A'+str(op+1), cvx.matrix(Am) ) )

M = prob.add_variable('M', (Nm,1) )
P = prob.add_variable('P', (Nm,Nm), vtype='symmetric')

# Restrictions (LMIs):
for qq in range(0, n_models):    
    prob.add_constraint( ( A_base_cvx[qq] + alpha*II ).T * P + P * ( A_base_cvx[qq] + alpha*II ) - M * C - C.T * M.T << 0  )

prob.add_constraint( P >> 0 )

#%% Solver:
prob.solve(verbosity = 1)
print('Status: ' + prob.status)

#%% Variables:
P = np.matrix(P.value)
M = np.matrix(M.value)
C = np.matrix(C.value)
L = P.I * M

P2 = P