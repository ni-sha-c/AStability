#!/usr/bin/python3
from numpy import *
from matplotlib.pyplot import *
import matplotlib.cm as cm
from scipy import signal

runup = 200
dacc = 40
racc = 4
skip = 1


direc = loadtxt('../'+'dir_names.txt',dtype=str)
n_dirs = len(direc)
cmap_list = ['winter', 'winter', 'autumn', 'autumn', 'viridis', 'viridis']
fig, ax = subplots()
ax.set_xlabel("Test accuracy", fontsize=16)
ax.set_ylabel("Loss", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)
for l, d_name in enumerate(direc):
    acc = loadtxt('../'+str(d_name)+'/test_acc.txt')
    acc = 100*acc[runup:-2]
    n_pts = len(acc)
    loss = loadtxt('../'+str(d_name)+'/loss.txt')
    loss = loss[runup:]
    crs = cm.get_cmap(cmap_list[l])
    for i in range(n_pts):
        c = crs(i/n_pts)
        ax.plot(acc[i], loss[i], '.', ms=4.0, color=c)
        if i == int(0.4*n_pts):
            if l == 0:
                ax.plot(acc[i], loss[i], '.', ms=4.0, color=c, label="0 % noise")

            if l == 2:
                ax.plot(acc[i], loss[i], '.', ms=4.0, color=c, label="25% noise")

            if l == 4:
                ax.plot(acc[i], loss[i], '.', ms=4.0, color=c, label="50 % noise")
ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/bifurcation_orig.png')


