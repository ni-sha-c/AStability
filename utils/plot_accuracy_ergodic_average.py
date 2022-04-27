#!/usr/bin/python3
from numpy import *
from matplotlib.pyplot import *

runup = 50
dacc = 40
racc = 4
direc = loadtxt('../'+'dir_names.txt',dtype=str)
fig, ax = subplots()

ec = ['cornflowerblue', 'greenyellow', 'indianred']
lc = ['darkblue', 'darkgreen', 'maroon']
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel("Test accuracy %", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)
for i, di in enumerate(direc):
    acc = loadtxt('../'+di+'/test_acc.txt')
    acc = 100*acc[racc:-2]
    lacc = len(acc)
    acc = cumsum(acc)/arange(1,lacc+1)
    if i < 3:
        if i==0:
            ax.plot(range(lacc), acc, color=lc[int(i//3)], lw=3.0,label='0% noise')
        else:
            ax.plot(range(lacc), acc, color=lc[int(i//3)], lw=3.0)
    elif i < 6:
        if i==3:
            ax.plot(range(lacc), acc, lw=3.0, color=lc[int(i//3)], label='25% noise')
        else:
            ax.plot(range(lacc), acc, lw=3.0,  color=lc[int(i//3)])
    else:
        if i==6:
            ax.plot(range(lacc), acc,lw=3.0, label='50% noise',  color=lc[int(i//3)])
        else:
            ax.plot(range(lacc), acc, lw=3.0,  color=lc[int(i//3)])
ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/test_acc_erg_avg.png')

