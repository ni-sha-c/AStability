#!/usr/bin/python3

from numpy import *
from matplotlib.pyplot import *

racc = 200
skip = 1

direc = loadtxt('../'+'list_stst.txt',dtype=str)
n_dpts = 10 
n_data = int(n_dpts*(n_dpts-1)//2)
n_noise = int(len(direc)//n_dpts)


ec = ['silver', 'cornflowerblue', 'violet', 'greenyellow', 'indianred']
lc = ['black', 'darkblue', 'purple', 'darkgreen', 'maroon']
noise = [0, 10, 17, 25, 50]
fig, ax = subplots()
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel(r"$|\Delta $ (test loss)|", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)
for l in range(n_noise):
    acc = loadtxt('../'+direc[n_dpts*l]+'/test_loss.txt')
    acc = acc[racc:]
    nacc = len(acc)
    acces = zeros((n_dpts, nacc))
    r_dpts = range(n_dpts*l,n_dpts*(l+1))
    print(l)
    for i, di in enumerate(direc[r_dpts]):
        print(i)
        acc = loadtxt('../'+di+'/test_loss.txt')
        acc = acc[racc:]
        nacc = len(acc)
        acc = cumsum(acc)/arange(1, nacc+1)
        acces[i] = acc

    k = 0
    dacces = zeros((n_data, nacc))
    m_dacc = zeros(nacc)
    for i in range(n_dpts):
        for j in range(i+1, n_dpts):
            dacces[k] = abs(acces[i] - acces[j])
            m_dacc += dacces[k]/n_data
            k = k + 1

    err_dacc = sum((dacces - m_dacc)**2, axis=0)/n_data
    err_dacc = sqrt(err_dacc/n_data)
    ax.errorbar(range(nacc)[::skip], m_dacc[::skip], yerr=(err_dacc[::skip]), lw=2.5, ecolor=ec[l], color=lc[l], elinewidth=2.0, label='{} % noise'.format(noise[l]))
ax.legend(fontsize=16)
tight_layout()
#fig.savefig('../plots/stst_resnet.png')

