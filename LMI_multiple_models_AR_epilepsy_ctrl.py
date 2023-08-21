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
B = np.zeros((Nm,Nm))
C = np.zeros((1,Nm))

alpha = 1680 # 1680/2 non-hybrid (SUB), 810/1.2 hybrid (SUB)

cte = 0
B[0][0] = 1
B[1][1] = 1
B[2][2] = 1
B[3][3] = 1
B[4][4] = 1
B[5][5] = 1

C[0][0] = 1

#%% Optimization problem:
prob = pic.Problem()
                  
A_base_cvx = []     

BB = pic.new_param('B', cvx.matrix(B))
CC = pic.new_param('C', cvx.matrix(C))
Gx = prob.add_variable('Gx', (Nm,Nm), vtype='symmetric')
X = prob.add_variable('X', (Nm,Nm), vtype='symmetric')
Q = prob.add_variable('Q', (Nm,Nm), vtype='symmetric')

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
