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
B = np.identity(Nm)
C = np.zeros((1,Nm))

alpha = 20

#SUB: 32 -- alpha = 40, b = 8.5e-5 (PIS) // alpha = 20, b = 8.5e-05 (input), but general is 8e-5
#SUB: 21 -- alpha = 20  b = 7e-5  (II)   // alpha = 0,  b = 5.0e-05 (input), but general is 7e-5
#DG:  27 -- alpha = 0, b = 9e-6 (PIS) 
#CA2: 29 -- alpha = 10, b = 8e-10  (PIS)
#CA1: 30 -- alpha = 20, b = 1e-4  (PIS)

cte = 0
B[0][0] = 8e-5
B[1][1] = cte
B[2][2] = cte
B[3][3] = cte
B[4][4] = cte
B[5][5] = cte

C[0][0] = 1

#%% Optimization problem:
prob = pic.Problem()
                  
A_base_cvx = []     

BB = pic.new_param('B', cvx.matrix(B))
CC = pic.new_param('C', cvx.matrix(C))
Gx = prob.add_variable('Gx', (Nm,Nm), vtype='symmetric')
X = prob.add_variable('X', (Nm,Nm), vtype='symmetric')

for op in range(0,n_models):  
    Am = A_base[op]
    A_base_cvx.append( pic.new_param('A'+str(op+1), cvx.matrix(Am) ) )

# Restrictions (LMIs):
for qq in range(0,n_models):    
    prob.add_constraint(X *  A_base_cvx[qq].T - Gx.T * BB.T +  A_base_cvx[qq] * X - BB * Gx + 2 * alpha * X << 0 )
    
prob.add_constraint(X >> 0)       

#%% Solver:
prob.solve(verbosity=1)
print('Status: ' + prob.status)

X = np.matrix(X.value)
Gx = np.matrix(Gx.value)

P = X.I
G = Gx.dot(P)
