# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 10:07:04 2021

@author: João Angelo Ferres Brogin
"""

import numpy as np
import matplotlib.pyplot as plt
import re
from scipy import signal
from scipy.signal import butter,filtfilt
from matplotlib import rc

rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

#%% Info about the channels:
# Interictal: 
    
#   [1,1] = DG(II)
#   [1,2] = DG(II)
#   [1,3] = DG(II)
#   [1,4] = DG(II)

#   [1,5] = CA4(II)
#   [1,6] = CA4(II)
#   [1,7] = CA4(II)
#   [1,8] = CA4(II)
#   [1,9] = CA4(II)

#   [1,10] = CA3(II)
#   [1,11] = CA3(II)
#   [1,12] = CA3(II)
#   [1,13] = CA3(II)
#   [1,14] = CA3(II)

#   [1,15] = CA2(II)

#   [1,16] = CA1(II)
#   [1,17] = CA1(II)

#   [1,18] = SUB(II)
#   [1,19] = SUB(II)
#   [1,20] = SUB(II)
#   [1,21] = SUB(II)
#   [1,22] = SUB(II)

# --------------------------------------------

# Ictal:

#   [1,23] = DG(PIS)
#   [1,24] = DG(PIS)
#   [1,25] = DG(PIS)
#   [1,26] = DG(PIS)
#   [1,27] = DG(PIS)

#   [1,28] = CA2(PIS)
#   [1,29] = CA2(PIS)

#   [1,30] = CA1(PIS)

#   [1,31] = SUB(PIS)
#   [1,32] = SUB(PIS)
#   [1,33] = SUB(PIS)
#   [1,34] = SUB(PIS)
#   [1,35] = SUB(PIS)

#%% Filters:
def butter_lowpass_filter(data_low, cutoff_low, Fs, order_low):
    nyq = 0.5 * Fs
    cutoff_low = cutoff_low / nyq
    b_low, a_low = butter(order_low, cutoff_low, btype='low', analog=False)
    y_low = filtfilt(b_low, a_low, data_low)
    return y_low

def butter_highpass_filter(data_high, cutoff_high, Fs, order_high):
    nyq = 0.5 * Fs
    cutoff_high = cutoff_high / nyq
    b_high, a_high = butter(order_high, cutoff_high, btype='high', analog=False)
    y_high = filtfilt(b_high, a_high, data_high)
    return y_high

def notch_filter(data_notch, Fs, notch_freq, quality_factor):
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, Fs)
    freq, h = signal.freqz(b_notch, a_notch, fs = Fs)
    notched_y = filtfilt(b_notch, a_notch, data_notch)
    return notched_y

def __filter__and__downsampling__( sign, samp_freq ):
    cutoff_low = 1000
    cutoff_high = 0.5      
    order_low = 5
    order_high = 1                
    notch_freq = 54
    quality_factor = 30

    # High-pass filter:
    filt_sig = butter_highpass_filter(sign, cutoff_high, samp_freq, order_high)
    
    # Low-pass filter:
    filt_sig2 = butter_lowpass_filter(filt_sig, cutoff_low, samp_freq, order_low)

    # Notch filter:    
    n_freqs = 10
    aux = filt_sig2
    for ii in range(0, n_freqs):
        notched_sig = notch_filter(aux, samp_freq, notch_freq*(ii + 1), quality_factor)
        aux = notched_sig
    
    # Downsampling:
    notched_dec_filt_sig = signal.decimate(notched_sig, 10)
    new_size = len(notched_dec_filt_sig)
    new_samp_freq = 1000
    new_dt = 1/new_samp_freq
    time_vec = np.linspace(0, new_size*new_dt, new_size)
    return time_vec, new_size, new_dt, notched_dec_filt_sig

#%% Import signals:
    
# Signals used in the paper:
# SUB: 18-32
# DG:   3-27
# CA2: 15-29
# CA1: 16-30
# SUB2: 18-35
    
# Interictal:
n_sig = 18
n_sig_ = n_sig - 1 

for k in range(n_sig_, n_sig):
    kk = k + 1
    print(kk)
    filename = 's'+str(kk)+'.txt'
    f1 = open(filename, 'r')
    sinal = re.findall(r"\S+", f1.read()) 
    f1.close()

sig = np.zeros(len(sinal))
    
for j in range(0, len(sinal)):
    sig[j] = round(float(sinal[j]),14)

Fs = 10000 
dt1 = 1/Fs
N1 = len(sig)
T1 = dt1*len(sig)
t1 = np.linspace(0, T1, N1)

#################################################
# Seizure:
n_sig2 = 32
n_sig_2 = n_sig2 - 1 

for k in range(n_sig_2, n_sig2):
    kk = k + 1
    print(kk)
    filename = 's'+str(kk)+'.txt'
    f1 = open(filename, 'r')
    sinal2 = re.findall(r"\S+", f1.read()) 
    f1.close()

sig2 = np.zeros(len(sinal2))
    
for j in range(0, len(sinal2)):
    sig2[j] = round(float(sinal2[j]),14)
    
Fs2 = 10000 
dt2 = 1/Fs2
N2 = len(sig2)
T2 = dt1*len(sig2)
t2 = np.linspace(0, T2, N2)

#%% Filtered downsampled signals:
t3, N3, dt3, dec_filt_sig  = __filter__and__downsampling__( sig,  Fs  )
t4, N4, dt4, dec_filt_sig2 = __filter__and__downsampling__( sig2, Fs2 )

#%% Plots:
import scipy 

NpS = 5000
Nlap = NpS//2 

# Check if harmonics were removed:
plt.figure(1)
freq1, Pot1 = scipy.signal.welch(sig, fs=1/dt1, window='hann', nperseg=NpS*10, noverlap=Nlap*5, nfft=None, scaling='density', average='mean')
freq2, Pot2 = scipy.signal.welch(sig2, fs=1/dt2, window='hann', nperseg=NpS*10, noverlap=Nlap*5, nfft=None, scaling='density', average='mean')
plt.semilogy(freq1, Pot1, 'r--', linewidth=1, label='$II$')
plt.semilogy(freq2, Pot2, 'b', linewidth=1, label='$PIS$')
plt.xlabel('$f$ $[Hz]$', fontsize = 25)
plt.ylabel('$PSD$ $[mV^2/Hz]$', fontsize = 25)
plt.xlim(0,500)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=25) 
plt.legend(loc=1,fontsize=20)

plt.figure(2)
freq1, Pot1 = scipy.signal.welch(dec_filt_sig, fs=1/dt3, window='hann', nperseg=NpS, noverlap=Nlap, nfft=None, scaling='density', average='mean')
freq2, Pot2 = scipy.signal.welch(dec_filt_sig2, fs=1/dt4, window='hann', nperseg=NpS, noverlap=Nlap, nfft=None, scaling='density', average='mean')
plt.semilogy(freq1, Pot1, 'r--', linewidth=1, label='$II$')
plt.semilogy(freq2, Pot2, 'b', linewidth=1, label='$PIS$')
plt.xlabel('$f$ $[Hz]$', fontsize = 25)
plt.ylabel('$PSD$ $[mV^2/Hz]$', fontsize = 25)
plt.xlim(0,500)
plt.grid()
plt.tick_params(axis='both', which='major', labelsize=25) 
plt.legend(loc=1,fontsize=20)


