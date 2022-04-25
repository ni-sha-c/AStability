#!/usr/bin/python3
from numpy import *
from matplotlib.pyplot import *
from scipy import signal

runup = 200
dacc = 40
racc = 4
skip = 1


direc = loadtxt('../'+'dir_names.txt',dtype=str)
n_dpts = 6 
n_data = int(n_dpts*(n_dpts-1)//2)
n_noise = int(len(direc)//n_dpts)

ec = ['cornflowerblue', 'greenyellow', 'indianred']
lc = ['darkblue', 'darkgreen', 'maroon']

fig, ax = subplots()
ax.set_xlabel("Frequency", fontsize=16)
ax.set_ylabel("Power spectral density", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)
for l in range(n_noise):
    acc = loadtxt('../'+direc[6*l]+'/test_acc.txt')
    acc = acc[racc:-1]
    nacc = len(acc)
    for i, di in enumerate(direc[6*l:6*(l+1)]):
        acc = loadtxt('../'+di+'/test_acc.txt')
        acc = acc[racc:-1]
        #acc = cumsum(acc)/arange(1, nacc+1)
        freq, psd_acc = signal.welch(acc, 1/40)
        ax.semilogy(freq, psd_acc, lw=2.0, color=lc[l])


ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/power_spectrum.png')


