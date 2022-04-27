#!/usr/bin/python3
from numpy import *
from matplotlib.pyplot import *
from scipy import signal

racc = 200

direc = loadtxt('../'+'dir_names.txt',dtype=str)
n_dpts = len(direc)
ec = ['cornflowerblue', 'greenyellow', 'indianred', 'violet']
lc = ['darkblue', 'darkgreen', 'maroon', 'purple']

fig, ax = subplots()
ax.set_xlabel("Frequency", fontsize=16)
ax.set_ylabel("Power spectral density", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)

n_times = 1024
auto_corr_acc = zeros((n_dpts, n_times))
for l in range(n_dpts):
    acc = loadtxt('../'+direc[l]+'/loss.txt')
    acc = acc[racc:-2]
    nacc = len(acc)
    print('nacc is', nacc) 
    macc = sum(acc)/nacc
    for k in range(n_times):
        i = 0
        while k+i < nacc:
            auto_corr_acc[l, k] += acc[i]*acc[k+i]
            i = i + 1
        if i != 0:
            auto_corr_acc[l,k] /= i
            auto_corr_acc[l,k] -= macc*macc
        auto_corr_acc[l] = abs(auto_corr_acc[l])
    freq, psd_acc = signal.welch(auto_corr_acc[l], 1000)
    if l < 6:
        if l == 0:
            ax.semilogy(freq, psd_acc, lw=2.0, color=lc[0], label="0 % noise")
        else:
            ax.semilogy(freq, psd_acc, lw=2.0, color=lc[0]) 
    elif l < 11: 
        if l == 6:
            ax.semilogy(freq, psd_acc, lw=2.0, color=lc[1], label="10 % noise")
        else:
            ax.semilogy(freq, psd_acc, lw=2.0, color=lc[1])
    elif l < 17:
        if l == 11:
            ax.semilogy(freq, psd_acc, lw=2.0, color=lc[2], label="25 % noise")
        else:
            ax.semilogy(freq, psd_acc, lw=2.0, color=lc[2])
    else:
        if l == 17:
            ax.semilogy(freq, psd_acc, lw=2.0, color=lc[3], label="50 % noise")
        else:
            ax.semilogy(freq, psd_acc, lw=2.0, color=lc[3])




ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/power_spectrum.png')


