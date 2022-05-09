#!/usr/bin/python3
from numpy import *
from matplotlib.pyplot import *

racc = 200
skip = 1


direc = loadtxt('../'+'list_stst.txt',dtype=str)
n_dpts = 5 
n_data = int(n_dpts*(n_dpts-1)//2)
n_noise = int(len(direc)//n_dpts)

ec = ['silver', 'cornflowerblue', 'violet', 'greenyellow', 'indianred']
lc = ['black', 'darkblue', 'purple', 'darkgreen', 'maroon']
noise = [0, 10, 17, 25, 50]
stcoeff = zeros(n_noise)
fig, ax = subplots()
ax.set_xlabel("% Noise", fontsize=16)
ax.set_ylabel("Stat stab coeff", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)

dacces = zeros((n_noise, n_data))
for l in range(n_noise):
    acc = loadtxt('../'+direc[n_dpts*l]+'/test_loss.txt')
    acc = acc[racc:]
    nacc = len(acc)
    acces = zeros(n_dpts)
    r_dpts = range(n_dpts*l,n_dpts*(l+1))
    for i, di in enumerate(direc[r_dpts]):
        acc = loadtxt('../'+di+'/test_loss.txt')
        acc = acc[racc:]
        acc = cumsum(acc)/arange(1, nacc+1)
        acces[i] = acc[-1]




    k = 0
    for i in range(n_dpts):
        for j in range(i+1, n_dpts):
            dacces[l,k] = abs(acces[i] - acces[j])
            k = k + 1
stcoeff = sum(dacces, axis=1)/n_data
err_stc = sum((dacces.T - stcoeff)**2, axis=0)/n_data
err_stc = sqrt(err_stc/n_data)
ax.errorbar(noise, stcoeff, yerr=(err_stc), fmt="o", ms=10, ecolor=ec[1], color=lc[1], elinewidth=2.0, label='{} % noise'.format(noise[l]))
#ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/stst_coeff.png')


