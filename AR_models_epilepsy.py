# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 15:54:35 2021

@author: João Angelo Ferres Brogin
"""
# Regular libraries:
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import time

rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# ARIMA libraries:
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# Import signals:
from epilepsy_signals import dec_filt_sig, dec_filt_sig2, dt3, n_sig, n_sig2

#%% Parameters of the signal:
Fs = 1/dt3
II_signal = dec_filt_sig
PIS_signal = dec_filt_sig2
N = len(II_signal)
time_vec = np.linspace( 0, dt3*N, N )
n_sig_II = n_sig
n_sig_PIS = n_sig2

#%% Model:
SIG_test = PIS_signal

time0 = time.time()

mse_v = []
aic_v = []
bic_v = []
arp_v = []
fitted_v = []
ar_order = 6
window = 5000

it_idx = N//window
#it_idx = 54 # for CA2, sig 29
#it_idx = 42 # for SUB, sig 35

for i in range(0,it_idx - 1):
    print('Window: ' + str(i + 1) )
    new_series = SIG_test[ window*i:window*(i + 1) ]
    model = ARIMA( new_series, order=(ar_order,0,0))
    model_fit = model.fit(disp=1)    

    AIC = model_fit.aic
    BIC = model_fit.bic
    AR_pars = model_fit.arparams
    fit_data = model_fit.fittedvalues
    residuals = model_fit.resid
    MSE = mean_squared_error(new_series, fit_data)
    
    mse_v.append(MSE)
    aic_v.append(AIC)
    bic_v.append(BIC)
    arp_v.append(list(AR_pars))
    fitted_v.append(fit_data)
    
elapsed = time.time() - time0
print('Elapsed time AR(' + str(ar_order) + '): ' + str(elapsed/60) + ' [min]')

# Preview of fitted model:
model_fit.plot_predict(dynamic=False)
plt.show()
print(model_fit.summary())

#%% Save coefficients in a .txt file:
#filename = 'AR_coefs_PIS_signal_II' + '_' + str(n_sig_II) + '.txt' # Choose a name for the file
#
#import json
#with open(filename,'w') as myfile:
#    json.dump(arp_v,myfile)
#    
#myfile.close()

#%% BIC/AIC criteria: 
# Single windows are used in this case to check the accuracy of the model
# while the order of the model varies from 1 to 15 (for example, as in the paper)

#n_order_max = 15
#vec_aux = np.linspace(1,n_order_max,n_order_max)
#
#plt.figure(2)
#plt.subplot(221)
#plt.plot(vec_aux, aic_v, 'bs-', markersize=15)
#plt.plot(vec_aux, bic_v, 'r*-', markersize=10)
#plt.grid()
#plt.xlim(1,n_order_max)
#plt.ylabel('$AIC/BIC$', fontsize = 35)
#plt.tick_params(axis='both', which='major', labelsize=30)
#
#plt.subplot(223)
#plt.plot(vec_aux, mse_v, 'kv-', markersize=10)
#plt.grid()
#plt.xlim(1,n_order_max)
#plt.xlabel('$k$', fontsize = 35)
#plt.ylabel('$MSE$', fontsize = 35)
#plt.tick_params(axis='both', which='major', labelsize=30)

#%% Plots with zoom:   
plt.figure(3)
plt.subplot(211)
#plt.plot(t2, dec_filt_sig, 'b')
#plt.plot(t2[0 : int(round(0.3*N2))], new_series, 'b', label='$Measured$')
#plt.plot(t2[0 : int(round(0.3*N2))], fitted_v[0], 'r--', label='$AR($'+'$'+str(ar_order)+'$'+'$)$')
plt.plot(time_vec[:5000], new_series, 'b', linewidth=2, label='$Measured$')
plt.plot(time_vec[:5000], fitted_v[-1], 'r--', linewidth=1, label='$AR($'+'$'+str(ar_order)+'$'+'$)$')
plt.ylabel('$mV$', fontsize = 35)
plt.tick_params(axis='both', which='major', labelsize=30) 
plt.grid()
plt.xlim(0,5)
#plt.legend(loc=4, fontsize=20)

plt.subplot(212)
plt.plot(time_vec[:5000], new_series, 'b', linewidth=1, label='$Measured$')
plt.plot(time_vec[:5000], fitted_v[-1], 'r--', linewidth=2, label='$AR($'+'$'+str(ar_order)+'$'+'$)$')
plt.xlabel('$t$ $[s]$', fontsize = 35)
plt.ylabel('$mV$', fontsize = 35)
plt.tick_params(axis='both', which='major', labelsize=30) 
plt.grid()
plt.xlim(3.5,4.5)
#plt.legend(loc=4, fontsize=20)

# linewidth no zoom: 2 e 1.5
# linewidth zoom: 2 and 3

#%% Residuals
from statsmodels.graphics.tsaplots import plot_acf

plot_acf(residuals, lags=4999)
plt.xlim(0,4999)
plt.xlabel('$Lags$',fontsize=35)
plt.ylabel(r'$\rho$',fontsize=35)
plt.title('')
plt.tick_params(axis='both', which='major', labelsize=30) 
plt.grid()

#%% Extension:

full_AR_model = np.zeros(len(dec_filt_sig2))
#AR_II_coef = arp_v[0]
AR_PIS_coef = arp_v[0]

#AR_coef_plot = AR_II_coef
AR_coef_plot = AR_PIS_coef

time_vec = np.linspace( 0, dt3*len(full_AR_model), len(full_AR_model) )

for nn in range(0, 6):
    aux = np.roll( dec_filt_sig2, nn + 1 )
    aux[:nn + 1] = 0
    aux2 = aux*AR_coef_plot[nn]
    full_AR_model = full_AR_model + aux2    
    
n_cell = int(np.floor(len(dec_filt_sig2)/5000))
cell = 5000
MSE_full = []

for jj in range(0, n_cell):
    part1 = full_AR_model[ cell*jj : cell*(jj + 1) ]
    part2 = dec_filt_sig2[ cell*jj : cell*(jj + 1) ]
    MSE_full.append( mean_squared_error( part1,part2 ) )
    
plt.figure(4)
plt.subplot(211)
plt.plot(time_vec, dec_filt_sig2, 'b', linewidth=2)
plt.plot(time_vec, full_AR_model, 'r--', linewidth=1)
#plt.xlabel('$t$ $[s]$',fontsize=35)
plt.ylabel('$mV$',fontsize=35)
plt.grid()
plt.xlim(0,time_vec[-1])
#plt.ylim(-0.8, 0.2)
plt.tick_params(axis='both', which='major', labelsize=30) 

plt.subplot(212)
plt.plot( MSE_full, 'kv-')
plt.plot( np.linspace( 1, n_cell, n_cell ), np.mean(MSE_full)*np.ones(n_cell), 'm:', linewidth=5 )
#plt.plot( np.linspace( 1, n_cell, n_cell ), np.mean(MSE_full)*np.ones(n_cell), 'b--', linewidth=2 )
plt.xlabel('$Cells$',fontsize=40)
plt.ylabel('$MSE$',fontsize=40)
plt.grid()
plt.xlim(1, n_cell)
#plt.ylim(-0.0005,0.0005)
plt.tick_params(axis='both', which='major', labelsize=35) 

#%% State-Space representation
AR_pars = AR_pars
l1 = list(AR_pars)
LAR = len(l1)
ref_line = np.zeros(LAR)
ref_line[0] = 1

M = []

for ii in range(0, LAR - 1):
    aux = np.roll( ref_line, ii )
    M = np.concatenate((M,aux), axis = 0)

MM = np.concatenate((l1,M), axis = 0)
MM = np.reshape(MM, (LAR,LAR))



