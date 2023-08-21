# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 09:59:27 2019

@author: João Angelo Ferres Brogin
"""

#import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from LMI_multiple_models_AR_epilepsy_obs import L, C
from LMI_multiple_models_AR_epilepsy_ctrl import G, B
from epilepsy_signals import dec_filt_sig, dec_filt_sig2
from multiple_models_AR_epilepsy import A_cont

rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# Make the system underactuated:
B[0][0] = 1
B[1][1] = 0
B[2][2] = 0 
B[3][3] = 0 
B[4][4] = 0 
B[5][5] = 0

#%% Parameters of simulation:
Nm = len(A_cont[0])
Fs = 1000
dt = 1/Fs
N = len(dec_filt_sig2)

sig_part1 = dec_filt_sig
sig_part2 = dec_filt_sig2
final_signal_cut = list(sig_part2)
obs_sig = list(sig_part2)

n_sim = len(sig_part2)

X = np.zeros((Nm, n_sim))
Err = np.zeros((Nm, n_sim))
Err[:,0] = np.ones(Nm)*(0.002)
F = np.zeros((Nm, n_sim))
X[:,0] = obs_sig[:Nm]

#%% State-space representation:
def __dXdt__( Xd, Ed, Xc, g, xh, counter ):
    Xd = np.reshape(Xd, (-1,1))
    Ed = np.reshape(Ed, (-1,1))
    
    # Response:
    A = A_cont[counter]    
    p1 = A.dot(Xd) 
    
    # Error dynamics:
    LC = L.dot(C)
    Ae = (A - LC)
    error = Ae.dot(Ed)
    sol_error = np.reshape(error,(-1,))
    
    # Observer:
    y_tilde = X_c - Xd
    p2 = LC.dot(y_tilde)
    
    # Observer + controller:
    BG = B.dot(G)
    Fc = -g[0] * BG.dot(Xd) + g[1] * BG.dot(xh)
    sol = p1 + p2 + Fc
    sol = np.reshape(sol,(-1,))
    
    return sol, sol_error, Fc
    
#%% 4th order Runge-Kutta algorithm:    
window = 5000
n_cells = int(np.floor(n_sim/window))
X_c = np.zeros((Nm,1))    
X_h = np.zeros((Nm,1)) 
aux = np.arange(1,n_cells + 1)
counter = 0
cnt = []
error_v = []

g = [0,0] # Non-hybrid: [1, 1.55] / Hybrid: [1, 2] 
n_sim = aux[-1]*window

for k in range(0,n_sim):   
    print(str(k))  
    if k >= n_sim//2:
        g = [1,1.55]
    
    # Observed signal (seizure):
    X_c[0] = final_signal_cut[k]
    
    # Healthy state (to which drive the system):
    X_h[0] = sig_part1[k]

    k1, ke1, Fn = __dXdt__( X[:,k], Err[:,k], X_c, g, X_h, counter )
    k2, ke2, Fn = __dXdt__( X[:,k] + k1*(dt/2), Err[:,k] + ke1*(dt/2), X_c, g, X_h, counter )
    k3, ke3, Fn = __dXdt__( X[:,k] + k2*(dt/2), Err[:,k] + ke2*(dt/2), X_c, g, X_h, counter )
    k4, ke4, Fn = __dXdt__( X[:,k] + k3*dt, Err[:,k] + ke3*dt, X_c, g, X_h, counter )
    Err[:,k+1] = Err[:,k] + (dt/6)*(ke1 + 2*ke2 + 2*ke3 + ke4)    
    X[:,k+1] = X[:,k] + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    # Input force:
    F[0,k] = Fn[0]
    F[1,k] = Fn[1]
    F[2,k] = Fn[2]
    F[3,k] = Fn[3]
    F[4,k] = Fn[4]
    F[5,k] = Fn[5]
    
    # Switches from model to model over the pre-defined time windows:
    if k >= window and k % window == 0:
        comp = np.where( k == aux*window )
        counter = comp[0][0] + 1
        cnt.append(counter)
    
#%% Norms:
t = np.linspace(0, dt*n_sim, n_sim)

Et = np.sqrt( Err[0,:n_sim]**2 + Err[1,:n_sim]**2 + Err[2,:n_sim]**2 + 
             Err[3,:n_sim]**2 + Err[4,:n_sim]**2 + Err[5,:n_sim]**2 )

sig_part1_aux = np.zeros( len(sig_part1) )
sig_part1_aux[n_sim//2:] = sig_part1[n_sim//2:]
Xt = np.sqrt( (X[0,:n_sim]-sig_part1_aux[:n_sim])**2 + X[1,:n_sim]**2 + X[2,:n_sim]**2 + 
             X[3,:n_sim]**2 + X[4,:n_sim]**2 + X[5,:n_sim]**2 )

#%% General behavior:
plt.figure(1)
plt.subplot(311)
plt.plot(t,obs_sig[:n_sim],'b', linewidth=2, label='$Uncontrolled$')
plt.plot(t,X[0,:n_sim],'r--',linewidth=1, label='$Controlled$')
plt.plot(dt*(n_sim//2)*np.ones(1000), np.linspace(-1.5,1.5,1000), 'k--')
plt.xlim(127,138)
plt.ylim(-1,1)
#plt.xlabel('$t$ $[s]$', fontsize = 25)
plt.ylabel('$mV$', fontsize = 25)
plt.tick_params(axis='both', which='major', labelsize=25) 
plt.grid()

plt.subplot(313)
plt.plot( t[:n_sim//2], np.zeros(len(t))[:n_sim//2], 'k:', linewidth=5 )
plt.plot( t[:n_sim//2], np.zeros(len(t))[:n_sim//2], 'k', linewidth=1 )
plt.plot(dt*(n_sim//2)*np.ones(1000), np.linspace(-1.5,1.5,1000), 'k--')
plt.plot( t[n_sim//2:n_sim], np.ones(len(t))[n_sim//2:n_sim], 'k:', linewidth=5, label='$Ctrl$' )
plt.plot( t[n_sim//2:n_sim], np.ones(len(t))[n_sim//2:n_sim], 'k', linewidth=1 )
plt.ylim(-0.2,1)
plt.plot( t, np.ones(len(t)), 'tab:orange', linewidth=3, label='$Obs$' )
plt.xlim(t[0], t[-1])
plt.xlabel('$t$ $[s]$', fontsize = 25)
plt.ylabel('$Ctrl/Obs$', fontsize = 30)
plt.legend(loc=4, fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=25) 
plt.grid()

y_lab = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]
labels = ['$Off$', '', '', '', '', '$On$', '']
plt.yticks(y_lab, labels)


plt.subplot(312)
plt.plot(t,obs_sig[:n_sim],'b', linewidth=2, label='$Uncontrolled$')
plt.plot(t,X[0,:n_sim],'r--',linewidth=1, label='$Controlled$')
plt.plot(dt*(n_sim//2)*np.ones(1000), np.linspace(-1.5,1.5,1000), 'k--')
plt.xlim(t[0], k*dt)
plt.ylim(-1,1)
#plt.xlabel('$t$ $[s]$', fontsize = 25)
plt.ylabel('$mV$', fontsize = 25)
plt.tick_params(axis='both', which='major', labelsize=25) 
plt.grid()






#%% Plots with norms and controller:
#plt.figure(2)
#plt.subplot(331)
#plt.plot( t, np.zeros(len(t)), 'k:', linewidth=5, label='$Ctrl$' )
#plt.plot( t, np.ones(len(t)), 'tab:orange', linewidth=3, label='$Obs$' )
#plt.xlim(t[0], t[-1])
#plt.xlabel('$t$ $[s]$', fontsize = 25)
##plt.ylabel('$Ctrl/Obs$', fontsize = 30)
#plt.legend(loc=4, fontsize=20)
#plt.tick_params(axis='both', which='major', labelsize=25) 
#plt.grid()
#plt.ylim(-0.2,1.2)
#
#y_lab = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]
#labels = ['$Off$', '', '', '', '', '$On$', '']
#plt.yticks(y_lab, labels)
#
#
#
#plt.subplot(332)
#plt.plot( t, np.ones(len(t)), 'k:', linewidth=5, label='$Ctrl$' )
#plt.plot( t, np.ones(len(t)), 'tab:orange', linewidth=3, label='$Obs$' )
#plt.xlim(t[0], t[-1])
#plt.xlabel('$t$ $[s]$', fontsize = 25)
#plt.title(r'$\mathbf{x}_{h}=0$', fontsize = 25)
#plt.legend(loc=4, fontsize=20)
#plt.tick_params(axis='both', which='major', labelsize=25) 
#plt.grid()
#plt.ylim(-0.2,1.2)
#
#y_lab = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]
#labels = ['$Off$', '', '', '', '', '$On$', '']
#plt.yticks(y_lab, labels)
#
#
#
#plt.subplot(333)
#plt.plot( t, np.ones(len(t)), 'k:', linewidth=5, label='$Ctrk$' )
#plt.plot( t, np.ones(len(t)), 'tab:orange', linewidth=3, label='$Obs$' )
#plt.xlim(t[0], t[-1])
#plt.xlabel('$t$ $[s]$', fontsize = 25)
#plt.title(r'$\mathbf{x}_{h}\neq 0$', fontsize = 25)
#plt.legend(loc=4, fontsize=20)
#plt.tick_params(axis='both', which='major', labelsize=25) 
#plt.grid()
#plt.ylim(-0.2,1.2)
#
#y_lab = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2]
#labels = ['$Off$', '', '', '', '', '$On$', '']
#plt.yticks(y_lab, labels)
#
#plt.subplot(334)
#plt.plot(t,obs_sig[:n_sim],'b', linewidth=2, label='$Uncontrolled$')
#plt.plot(t,X[0,:n_sim],'r--',linewidth=1, label='$Controlled$')
#plt.plot(dt*(n_sim//2)*np.ones(1000), np.linspace(-0.8,0.8,1000), 'k--')
#plt.xlim(t[0], 5)
#plt.ylim(-0.2,0.15)
#plt.xlabel('$t$ $[s]$', fontsize = 25)
#plt.ylabel('$mV$', fontsize = 25)
#plt.tick_params(axis='both', which='major', labelsize=25) 
#plt.grid()
#
## Uncomment to plot controlled signals:
##plt.subplot(335)
##plt.plot(t,obs_sig[:n_sim],'b', linewidth=2, label='$Uncontrolled$')
##plt.plot(t,X[0,:n_sim],'r--',linewidth=1, label='$Controlled$')
##plt.plot(dt*(n_sim//2)*np.ones(1000), np.linspace(-0.8,0.8,1000), 'k--')
##plt.xlim(t[0], 5)
##plt.ylim(-0.2,0.15)
##plt.xlabel('$t$ $[s]$', fontsize = 25)
##plt.ylabel('$mV$', fontsize = 25)
##plt.tick_params(axis='both', which='major', labelsize=25) 
##plt.grid()
#
##plt.subplot(336)
##plt.plot(t,obs_sig[:n_sim],'b', linewidth=2, label='$Uncontrolled$')
##plt.plot(t,X[0,:n_sim],'r--',linewidth=1, label='$Controlled$')
##plt.plot(dt*(n_sim//2)*np.ones(1000), np.linspace(-0.8,0.8,1000), 'k--')
##plt.xlim(t[0], 5)
##plt.ylim(-0.3,0.15)
##plt.xlabel('$t$ $[s]$', fontsize = 25)
##plt.ylabel('$mV$', fontsize = 25)
##plt.tick_params(axis='both', which='major', labelsize=25) 
##plt.grid()



#%% Plots with norms and controller:
#plt.figure(3)
#plt.subplot(331)
#plt.plot(t,obs_sig[:n_sim],'b', linewidth=2, label='$Uncontrolled$')
#plt.plot(t,X[0,:n_sim],'r--',linewidth=2, label='$Controlled$')
#plt.plot(dt*(n_sim//2)*np.ones(1000), np.linspace(-0.8,0.8,1000), 'k--')
#plt.xlim(t[0], 0.1)
#plt.ylim(-0.1,0.1)
#plt.xlabel('$t$ $[s]$', fontsize = 25)
#plt.ylabel('$mV$', fontsize = 25)
#plt.tick_params(axis='both', which='major', labelsize=25) 
#plt.grid()
#
## Uncomment to plot controlled signals:
##plt.subplot(332)
##plt.plot(t,obs_sig[:n_sim],'b', linewidth=2, label='$Uncontrolled$')
##plt.plot(t,X[0,:n_sim],'r--',linewidth=2, label='$Controlled$')
##plt.plot(dt*(n_sim//2)*np.ones(1000), np.linspace(-0.8,0.8,1000), 'k--')
##plt.xlim(t[0], 0.1)
##plt.ylim(-0.1,0.1)
##plt.xlabel('$t$ $[s]$', fontsize = 25)
##plt.ylabel('$mV$', fontsize = 25)
##plt.tick_params(axis='both', which='major', labelsize=25) 
##plt.grid()
#
#plt.subplot(333)
#plt.plot(t,obs_sig[:n_sim],'b', linewidth=2, label='$Uncontrolled$')
#plt.plot(t,X[0,:n_sim],'r--',linewidth=2, label='$Controlled$')
#plt.plot(dt*(n_sim//2)*np.ones(1000), np.linspace(-0.8,0.8,1000), 'k--')
#plt.xlim(t[0], 0.1)
#plt.ylim(-0.1,0.1)
#plt.xlabel('$t$ $[s]$', fontsize = 25)
#plt.ylabel('$mV$', fontsize = 25)
#plt.tick_params(axis='both', which='major', labelsize=25) 
#plt.grid()
#
#
#plt.subplot(334)
#plt.plot(t,Et,'m-.', linewidth=2)
#plt.xlim(t[0], 0.1)
#plt.ylim(0,unc_fac)
#plt.xlabel('$t$ $[s]$', fontsize = 25)
#plt.ylabel(r'$\parallel \mathbf{e}(t) \parallel$', fontsize = 25)
#plt.tick_params(axis='both', which='major', labelsize=25) 
#plt.grid()
#
##plt.subplot(335)
##plt.plot(t,Xt,'m-.', linewidth=2)
##plt.xlim(t[0], 0.1)
##plt.ylim(0,unc_fac)
##plt.xlabel('$t$ $[s]$', fontsize = 25)
##plt.ylabel(r'$\parallel \mathbf{x}(t) \parallel$', fontsize = 25)
##plt.tick_params(axis='both', which='major', labelsize=25) 
##plt.grid()
#
##plt.subplot(336)
##plt.plot(t,Xt,'m-.', linewidth=2)
##plt.xlim(t[0], 0.1)
##plt.ylim(0,unc_fac)
##plt.xlabel('$t$ $[s]$', fontsize = 25)
##plt.ylabel(r'$\parallel \mathbf{x}(t)-\mathbf{x}_{h} \parallel$', fontsize = 25)
##plt.tick_params(axis='both', which='major', labelsize=25) 
##plt.grid()


#%% No swizure:

#plt.plot(3)
#plt.plot(t, dec_filt_sig[:n_sim],'g')
#plt.xlim(t[0], t[-1])
#plt.xlabel('$t$ $[s]$', fontsize = 30)
#plt.ylabel('$mV$', fontsize = 30)
#plt.legend(loc=4, fontsize=20)
#plt.tick_params(axis='both', which='major', labelsize=30) 
#plt.grid()
#plt.ylim(-0.8,0.2)

#%% Spectrogram:

# 0.1, 0.025

window = 0.0125/2
superpos = 0.0125/4

plt.figure(5)
plt.subplot(312)
Pxx, freqs, bins, im = plt.specgram(X[0,:n_sim], NFFT=int(n_sim*window), Fs=Fs, noverlap=int(n_sim*superpos),cmap='jet')
plt.plot(dt*(n_sim//2)*np.ones(1000), np.linspace(0,50,1000), 'k--')
plt.xlabel('$t$ $[s]$', fontsize = 25)
plt.ylabel('$f$ $[Hz]$', fontsize = 25)
#plt.title('$x_{1}$', fontsize = 25)
plt.tick_params(axis='both', which='major', labelsize=20) 
#plt.xlim(0,200)
plt.ylim(0,50)
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=20) 


plt.subplot(313)
Pxx, freqs, bins, im = plt.specgram(X[0,:n_sim], NFFT=int(n_sim*window), Fs=Fs, noverlap=int(n_sim*superpos),cmap='jet')
plt.plot(dt*(n_sim//2)*np.ones(1000), np.linspace(0,50,1000), 'k--')
plt.xlabel('$t$ $[s]$', fontsize = 25)
plt.ylabel('$f$ $[Hz]$', fontsize = 25)
#plt.title('$x_{1}$', fontsize = 25)
plt.tick_params(axis='both', which='major', labelsize=20) 
#plt.xlim(0,200)
plt.ylim(0,50)
#cbar = plt.colorbar()
#cbar.ax.tick_params(labelsize=20) 

#plt.subplot(132)
#Pxx, freqs, bins, im = plt.specgram(sig_part1, NFFT=int(n_sim*window), Fs=Fs, noverlap=int(n_sim*superpos),cmap='jet')
#    
#plt.xlabel('$t$ $[s]$', fontsize = 25)
#plt.ylabel('$f$ $[Hz]$', fontsize = 25)
#plt.title('$x_{1}$', fontsize = 25)
#plt.tick_params(axis='both', which='major', labelsize=20) 
##plt.xlim(0,200)
#plt.ylim(0,100)
#
#plt.subplot(133)
#Pxx, freqs, bins, im = plt.specgram(X[0,:], NFFT=int(n_sim*window), Fs=Fs, noverlap=int(n_sim*superpos),cmap='jet')
#    
#plt.xlabel('$t$ $[s]$', fontsize = 25)
#plt.ylabel('$f$ $[Hz]$', fontsize = 25)
#plt.title('$x_{1}$', fontsize = 25)
#plt.tick_params(axis='both', which='major', labelsize=20) 
##plt.xlim(0,200)
#plt.ylim(0,100)

#%% Input:
#n_interval_inp = 20000
#n_sim_ini_inp = n_sim//2
#n_sim_fin_inp = n_sim//2 + n_interval_inp
#t_inp = np.linspace(dt*(n_sim_ini_inp - 1), dt*n_sim_fin_inp, n_interval_inp)

n_interval_inp = 20000
n_sim_ini_inp = 332000
n_sim_fin_inp = n_sim_ini_inp + n_interval_inp
t_inp = np.linspace(dt*(n_sim_ini_inp - 1), dt*n_sim_fin_inp, n_interval_inp)

# Controlled activity:
plt.figure(6)
plt.subplot(221)
plt.plot(t_inp, sig_part2[n_sim_ini_inp - 1 : n_sim_fin_inp - 1],'b',linewidth=1)
plt.plot(t_inp, X[0,n_sim_ini_inp - 1 : n_sim_fin_inp - 1],'r--',linewidth=1)
plt.xlim(340,350)
plt.ylim(-0.35,0.1)
plt.ylabel('$mV$', fontsize = 25)   
plt.tick_params(axis='both', which='major', labelsize=25) 
plt.grid()

plt.subplot(222)
plt.plot(t_inp, sig_part2[n_sim_ini_inp - 1 : n_sim_fin_inp - 1],'b',linewidth=1)
plt.plot(t_inp, X[0,n_sim_ini_inp - 1 : n_sim_fin_inp - 1],'r--',linewidth=1)
plt.xlim(345,347)
plt.ylim(-0.35,0.1)
plt.ylabel('$mV$', fontsize = 25)
plt.tick_params(axis='both', which='major', labelsize=25) 
plt.grid()

# Input:
plt.subplot(223)
plt.plot(t_inp, F[0,n_sim_ini_inp - 1 : n_sim_fin_inp - 1],'k', linewidth=1)
plt.xlim(340,350)
plt.ylim(-1500,800)
plt.xlabel('$t$ $[s]$', fontsize = 25)
plt.ylabel(r'$F_{1}(t)$ $[mV]$', fontsize = 25)
plt.tick_params(axis='both', which='major', labelsize=25) 
plt.grid()

plt.subplot(224)
#plt.plot( t_inp, -*np.ones(len(t_inp)),  ':', color='darkorange', linewidth=2 )
#plt.plot( t_inp, -*np.ones(len(t_inp)), ':', color='darkorange', linewidth=2 )
plt.plot(t_inp, F[0,n_sim_ini_inp - 1 : n_sim_fin_inp - 1],'k', linewidth=1)
plt.xlim(345,347)
plt.ylim(-1500,800)
plt.xlabel('$t$ $[s]$', fontsize = 25)
plt.ylabel(r'$F_{1}(t)$ $[mV]$', fontsize = 25)
plt.tick_params(axis='both', which='major', labelsize=25) 
plt.grid()

# Hybrud:       
# Non-hybrid: 

# hybrid:       -850 --> -595
# non-hybrid: -1480 --> -740

#%% PSD:
window = 5000

# References:
ref_II  = sig_part1[n_sim//2:n_sim] 
ref_PIS = np.array(final_signal_cut[:n_sim//2]) 

# Tests:
signal1 = X[0,n_sim//2:n_sim] # II (controlled)
signal2 = X[0,:n_sim//2]      # PIS (uncontrolled)

psd_vec1 = []
psd_vec2 = []
psd_vec3 = []
psd_vec4 = []

n_psd = int(len(signal2)//window)
NpS = 2500
Nlap = 500

# Comparisons:
import scipy

for qq in range(0, n_psd):
    f1, Pxx1 = scipy.signal.welch(ref_II[  qq*window:(qq + 1)*window ], fs=Fs, window='hann', nperseg=NpS, noverlap=Nlap, nfft=None, scaling='density', average='mean')
    f2, Pxx2 = scipy.signal.welch(ref_PIS[ qq*window:(qq + 1)*window ], fs=Fs, window='hann', nperseg=NpS, noverlap=Nlap, nfft=None, scaling='density', average='mean')

    f3, Pxx3 = scipy.signal.welch(signal1[ qq*window:(qq + 1)*window ], fs=Fs, window='hann', nperseg=NpS, noverlap=Nlap, nfft=None, scaling='density', average='mean')
    f4, Pxx4 = scipy.signal.welch(signal2[ qq*window:(qq + 1)*window ], fs=Fs, window='hann', nperseg=NpS, noverlap=Nlap, nfft=None, scaling='density', average='mean')
    
    psd_vec1.append( Pxx1 )
    psd_vec2.append( Pxx2 )
    psd_vec3.append( Pxx3 )
    psd_vec4.append( Pxx4 )
    
# Means and standard deviations:   
mean_psd1 = np.mean(psd_vec1,axis=0)
mean_psd2 = np.mean(psd_vec2,axis=0)
mean_psd3 = np.mean(psd_vec3,axis=0)
mean_psd4 = np.mean(psd_vec4,axis=0)

std_psd1 = np.std(psd_vec1,axis=0)
std_psd2 = np.std(psd_vec2,axis=0)
std_psd3 = np.std(psd_vec3,axis=0)
std_psd4 = np.std(psd_vec4,axis=0)

plt.figure()
plt.subplot(221)
for pp in range(0, n_psd):
    plt.semilogy(f1, psd_vec1[pp], 'gainsboro', linewidth=0.1)
#    plt.xlabel('$f$ $[Hz]$', fontsize = 25)
    plt.ylabel('$PSD$ $[mV^2/Hz]$', fontsize = 25)
    plt.xlim(0,50)
    plt.ylim(1e-8,1e-2)
    plt.tick_params(axis='both', which='major', labelsize=25) 
plt.semilogy(f1, mean_psd1, 'r', linewidth=3, label='$Ref.$ $II$')
plt.semilogy(f1, mean_psd1 + 1.96*std_psd1/np.sqrt(n_psd), 'k:', linewidth=3)
plt.semilogy(f1, mean_psd1 - 1.96*std_psd1/np.sqrt(n_psd), 'k:', linewidth=3)
plt.grid()    
plt.legend(loc=1, fontsize=20)

plt.subplot(222)
for pp in range(0, n_psd):
    plt.semilogy(f3, psd_vec2[pp], 'gainsboro', linewidth=0.1)
#    plt.xlabel('$f$ $[Hz]$', fontsize = 25)
    plt.ylabel('$PSD$ $[mV^2/Hz]$', fontsize = 25)
    plt.xlim(0,50)
    plt.ylim(1e-8,1e-2)
    plt.tick_params(axis='both', which='major', labelsize=25) 
plt.semilogy(f1, mean_psd2, 'b', linewidth=3, label='$Ref.$ $PIS$')
plt.semilogy(f1, mean_psd2 + 1.96*std_psd2/np.sqrt(n_psd), 'k:', linewidth=3)
plt.semilogy(f1, mean_psd2 - 1.96*std_psd2/np.sqrt(n_psd), 'k:', linewidth=3)
plt.grid()   
plt.legend(loc=1, fontsize=20)

plt.subplot(223)
for pp in range(0, n_psd):
    plt.semilogy(f2, psd_vec3[pp], 'gainsboro', linewidth=0.1)
    plt.xlabel('$f$ $[Hz]$', fontsize = 25)
    plt.ylabel('$PSD$ $[mV^2/Hz]$', fontsize = 25)
    plt.xlim(0,50)
    plt.ylim(1e-8,1e-2)
    plt.tick_params(axis='both', which='major', labelsize=25) 
plt.semilogy(f1, mean_psd3, 'r--', linewidth=3, label='$Controlled$')
plt.semilogy(f1, mean_psd3 + 1.96*std_psd3/np.sqrt(n_psd), 'k:', linewidth=3)
plt.semilogy(f1, mean_psd3 - 1.96*std_psd3/np.sqrt(n_psd), 'k:', linewidth=3)
plt.grid()   
plt.legend(loc=1, fontsize=20)
    
plt.subplot(224)
for pp in range(0, n_psd):
    plt.semilogy(f4, psd_vec4[pp], 'gainsboro', linewidth=0.1)
    plt.xlabel('$f$ $[Hz]$', fontsize = 25)
    plt.ylabel('$PSD$ $[mV^2/Hz]$', fontsize = 25)
    plt.xlim(0,50)
    plt.ylim(1e-8,1e-2)
    plt.tick_params(axis='both', which='major', labelsize=25) 
plt.semilogy(f1, mean_psd4, 'b--', linewidth=3, label='$Uncontrolled$')
plt.semilogy(f1, mean_psd4 + 1.96*std_psd4/np.sqrt(n_psd), 'k:', linewidth=3)
plt.semilogy(f1, mean_psd4 - 1.96*std_psd4/np.sqrt(n_psd), 'k:', linewidth=3)
plt.grid()   
plt.legend(loc=1, fontsize=20)

plt.figure()
plt.subplot(221)
plt.semilogy(f1, mean_psd1, 'r', linewidth=3, label='$Ref.$ $II$')
plt.semilogy(f2, mean_psd2, 'b', linewidth=3, label='$Ref.$ $PIS$')
plt.semilogy(f3, mean_psd3, 'r--', linewidth=3, label='$Controlled$')
plt.semilogy(f4, mean_psd4, 'b--', linewidth=3, label='$Uncontrolled$')
plt.xlabel('$f$ $[Hz]$', fontsize = 25)
plt.ylabel('$PSD$ $[mV^2/Hz]$', fontsize = 25)
plt.xlim(0,50)
plt.ylim(1e-8,1e-2)
plt.grid()
plt.legend(loc=1, fontsize=20)
plt.tick_params(axis='both', which='major', labelsize=25) 

#%% PCA:
psd_refII  = np.array(psd_vec1)
psd_refPIS = np.array(psd_vec2)
psd_sig1   = np.array(psd_vec3)
psd_sig2   = np.array(psd_vec4)

signals = np.vstack( (psd_refII,psd_refPIS, psd_sig1, psd_sig2) )
cvt = np.cov(signals.T)
eig_val, eig_vec = np.linalg.eig(cvt)
idx = eig_val.argsort()[::-1]
eig_val = eig_val[idx]
eig_vec = eig_vec[:,idx]

# Set 1:
#PCA1 = eig_vec[:,0].dot(signals.T)
#PCA2 = eig_vec[:,1].dot(signals.T)
#PCA3 = eig_vec[:,2].dot(signals.T)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('$PC_{1}$', fontsize=20)
ax.set_ylabel('$PC_{2}$', fontsize=20)
ax.set_zlabel('$PC_{3}$', fontsize=20)
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.zaxis.set_tick_params(labelsize=10)

#ax.azim = 60 # Rotation
#ax.dist = 10 # Zoom
#ax.elev = 30 # Elevation

# Ref II:
PCA1_test1 = eig_vec[:,0].dot(psd_refII.T)
PCA2_test1 = eig_vec[:,1].dot(psd_refII.T)
PCA3_test1 = eig_vec[:,2].dot(psd_refII.T)
ax.plot(PCA1_test1,PCA2_test1,PCA3_test1,'ro', linewidth=5, label='$Ref.$ $II$')
plt.show()

# Ref PIS:
PCA1_test2 = eig_vec[:,0].dot(psd_refPIS.T)
PCA2_test2 = eig_vec[:,1].dot(psd_refPIS.T)
PCA3_test2 = eig_vec[:,2].dot(psd_refPIS.T)
ax.plot(PCA1_test2,PCA2_test2,PCA3_test2,'bo', linewidth=5, label='$Ref.$ $PIS$')
plt.show()

# Ref signal1:
PCA1_test3 = eig_vec[:,0].dot(psd_sig1.T)
PCA2_test3 = eig_vec[:,1].dot(psd_sig1.T)
PCA3_test3 = eig_vec[:,2].dot(psd_sig1.T)
ax.plot(PCA1_test3,PCA2_test3,PCA3_test3,'rv', linewidth=5, label='$Controlled$')
plt.show()

# Ref signal2:
PCA1_test4 = eig_vec[:,0].dot(psd_sig2.T)
PCA2_test4 = eig_vec[:,1].dot(psd_sig2.T)
PCA3_test4 = eig_vec[:,2].dot(psd_sig2.T)
ax.plot(PCA1_test4,PCA2_test4,PCA3_test4,'b^', linewidth=5, label='$Uncontrolled$')
plt.show()

plt.legend(loc=1,fontsize=15)



# Normality test (two-sided):
norm1 = scipy.stats.normaltest(np.real(PCA2_test1))
norm2 = scipy.stats.normaltest(np.real(PCA2_test2))
norm3 = scipy.stats.normaltest(np.real(PCA2_test3))
norm4 = scipy.stats.normaltest(np.real(PCA2_test4))

# Kruskal-Wallis + Dunn's test:
import scikit_posthocs as sp

KW_stat, p_value_KW = scipy.stats.kruskal( np.real(PCA2_test1), np.real(PCA2_test2), np.real(PCA2_test3), np.real(PCA2_test4)  )
dunns_test = [np.real(PCA2_test1), np.real(PCA2_test2), np.real(PCA2_test3), np.real(PCA2_test4)]
p_values_dunn = sp.posthoc_dunn(dunns_test)

# ANOVA + Tukey's test:
ANOVA_test, p_value_ANOVA = scipy.stats.f_oneway( np.real(PCA2_test1), np.real(PCA2_test2), np.real(PCA2_test3), np.real(PCA2_test4)  )
tukeys_test = [np.real(PCA2_test1), np.real(PCA2_test2), np.real(PCA2_test3), np.real(PCA2_test4)]
p_values_tukey = sp.posthoc_tukey(tukeys_test)

#%% Cross-correlation (time domain):
from scipy import signal

cor1 = []
cor2 = []
cor3 = []
cor4 = []

for gg in range(0, n_psd):
    ref_II_cor  = ref_II[  gg*window:(gg + 1)*window ]
    ref_PIS_cor = ref_PIS[ gg*window:(gg + 1)*window ]
    signal1_cor = signal1[ gg*window:(gg + 1)*window ]
    signal2_cor = signal2[ gg*window:(gg + 1)*window ]

    ref_II_cor  = ( ref_II_cor  - np.mean(ref_II_cor)  ) / np.std(ref_II_cor)
    ref_PIS_cor = ( ref_PIS_cor - np.mean(ref_PIS_cor) ) / np.std(ref_PIS_cor)
    signal1_cor = ( signal1_cor - np.mean(signal1_cor) ) / np.std(signal1_cor)
    signal2_cor = ( signal2_cor - np.mean(signal2_cor) ) / np.std(signal2_cor)

    # Sem controle vs II
    correlation1 = signal.correlate(ref_II_cor, signal2_cor, mode="full")/len(ref_II_cor)
    lags1 = signal.correlation_lags(ref_II_cor.size, signal2_cor.size, mode="full")
    lag1_max = np.argmax(correlation1)
    cor1.append(max(correlation1))

    # Sem controle vs PIS
    correlation2 = signal.correlate(ref_PIS_cor, signal2_cor, mode="full")/len(ref_PIS_cor)
    lags2 = signal.correlation_lags(ref_PIS_cor.size, signal2_cor.size, mode="full")
    lag2_max = np.argmax(correlation2)
    cor2.append(max(correlation2))

    # Com controle vs II
    correlation3 = signal.correlate(ref_II_cor, signal1_cor, mode="full")/len(ref_II_cor)
    lags3 = signal.correlation_lags(ref_II_cor.size, signal1_cor.size, mode="full")
    lag3_max = np.argmax(correlation3)
    cor3.append(max(correlation3))

    # Com controle vs PIS
    correlation4 = signal.correlate(ref_PIS_cor, signal1_cor, mode="full")/len(ref_PIS_cor)
    lags4 = signal.correlation_lags(ref_PIS_cor.size, signal1_cor.size, mode="full")
    lag4_max = np.argmax(correlation4)
    cor4.append(max(correlation4))

## Mean, standard deviation and confidence intervals vectors:
Mcor = [np.mean(cor2),np.mean(cor1),np.mean(cor4),np.mean(cor3)]

ICs_cor = [1.96*np.std(cor2)/len(cor2), 1.96*np.std(cor1)/len(cor1),
           1.96*np.std(cor4)/len(cor4), 1.96*np.std(cor3)/len(cor3)]

plt.figure()
#plt.bar( ('$A$','$B$', '$C$', '$D$'), Mcor, yerr=ICs_cor, ecolor='black', color='lightsteelblue', width=0.8, bottom=None, align='center', capsize=30)

# ICs:
plt.errorbar( [1,2], [np.mean(cor1),np.mean(cor3)], yerr=[1.96*np.std(cor1)/len(cor1),1.96*np.std(cor3)/len(cor3)], 
             ecolor='black', color='red', capsize=30, fmt='--s', markersize = 8, label='$II$')
plt.errorbar( [1,2], [np.mean(cor2),np.mean(cor4)], yerr=[1.96*np.std(cor2)/len(cor2),1.96*np.std(cor4)/len(cor4)], 
             ecolor='black', color='blue', capsize=30, fmt='-o', markersize = 8, label='$PIS$')

plt.title('$Confidence$ $Intervals$', fontsize = 25)
plt.ylabel(r'$max(\rho)$', fontsize = 25)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=25) 
labels = ['$Uncontrolled$', '$Controlled$']
plt.xticks([1,2], labels)
plt.xlim(0.75,2.25)
plt.legend(loc=1, fontsize=20)

