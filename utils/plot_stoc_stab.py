#!/usr/bin/python3
from numpy import *
from matplotlib.pyplot import *

runup = 200
dacc = 1
racc = 4
skip = 10


direc = loadtxt('../'+'dir_names.txt',dtype=str)
n_dpts = 6 
n_data = int(n_dpts*(n_dpts-1)//2)
n_noise = int(len(direc)//n_dpts)

ec = ['cornflowerblue', 'greenyellow', 'indianred', 'violet']
lc = ['darkblue', 'darkgreen', 'maroon', 'purple']
noise = [0, 10, 25, 50]
n_d = [6, 4, 6, 6]
fig, ax = subplots()
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel(r"$|\Delta $ Accuracy| in %", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)
for l in range(n_noise-1):
    acc = loadtxt('../'+direc[6*l]+'/test_acc.txt')
    acc = acc[racc:-1]
    nacc = len(acc)
    acces = zeros((n_dpts, nacc))
    r_dpts = range(n_dpts*l,n_dpts*(l+1))
    if l == 1:
        r_dpts =  range(n_dpts*l,n_dpts*(l+1)-2)
    for i, di in enumerate(direc[r_dpts]):
        acc = loadtxt('../'+di+'/test_acc.txt')
        acc = acc[racc:-1]
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
    ax.plot(range(0,dacc*nacc,dacc)[::skip], m_dacc[::skip], color=lc[l], linewidth=3.0, label='{} % noise'.format(noise[l]))
    #ax.errorbar(range(0,dacc*nacc,dacc)[::skip], log10(m_dacc[::skip]), yerr=log10(err_dacc[::skip]), fmt="o", ms=10, ecolor=ec[l], color=lc[l], elinewidth=2.0, label='{} % noise'.format(noise[l]))
ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/stochastic_stability.png')


