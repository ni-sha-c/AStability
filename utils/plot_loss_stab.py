#!/usr/bin/python3
from numpy import *
from matplotlib.pyplot import *

epsi = 1.e-3
runup = 200
dacc = 40
racc = 4
skip = 1
nn = 1
n_lvl = [0, 10, 17, 25, 50]

clr  = ['silver', 'cornflowerblue', 'violet', 'greenyellow', 'indianred']
#clr = ['black', 'darkblue', 'purple', 'darkgreen', 'maroon']

fig, ax = subplots()
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel("Time-averaged loss", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)

for l in range(nn):
    direc = loadtxt('../'+'dir_names' + str(l) + '.txt',dtype=str)
    n_dpts = len(direc)
    loss = loadtxt('../'+direc[0]+'/loss.txt')
    loss = loss[runup:]
    num_el = len(loss)
    loss = cumsum(loss)/arange(1, num_el+1)
    #ax.plot(range(num_el)[::skip], loss[::skip], color=clr[l], linewidth=3.0, label='{}% noise'.format(n_lvl[l]))
    for i, di in enumerate(direc):
        loss = loadtxt('../'+di+'/loss.txt')
        loss = loss[runup:]
        loss = cumsum(loss)/arange(1, num_el+1)
        loss /= mean(loss)
        ax.plot(range(num_el)[::skip], loss[::skip], linewidth=3.0)
#ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/loss_stab_loss.png')



fig, ax = subplots()
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel("Time-avgd test acc", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)

for l in range(nn):
    direc = loadtxt('../'+'dir_names' + str(l) + '.txt',dtype=str)
    acc = loadtxt('../'+direc[0]+'/test_acc.txt')
    acc = acc[racc:-1]
    nacc = len(acc)
    acc = cumsum(acc)/arange(1, nacc+1)

    #ax.plot(range(0,dacc*nacc,dacc), acc, color=clr[l], lw = 3.0, label='{}% noise'.format(n_lvl[l])) 
    for i, di in enumerate(direc):
        acc = loadtxt('../'+di+'/test_acc.txt')
        acc = acc[racc:-1]
        acc = cumsum(acc)/arange(1, nacc+1)
        acc /= mean(acc)
        ax.plot(range(0,dacc*nacc,dacc), acc, lw = 3.0)
#ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/loss_stab_acc.png')


fig, ax = subplots()
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel("Time-avgd ||1st lr wt||", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)

for l in range(nn):
    direc = loadtxt('../'+'dir_names' + str(l) + '.txt',dtype=str)
    n_dpts = len(direc)
    n = loadtxt('../'+direc[0]+'/norm.txt')
    n = n[runup:]
    n = cumsum(n)/arange(1, num_el+1)
    #ax.plot(range(num_el)[::skip], n[::skip], color=clr[l], linewidth=3.0, label='{}% noise'.format(n_lvl[l]))
    for i, di in enumerate(direc):
        n = loadtxt('../'+di+'/norm.txt')
        n = n[runup:]
        n = cumsum(n)/arange(1, num_el+1)
        n /= mean(n)
        ax.plot(range(num_el)[::skip], n[::skip], linewidth=3.0)
#ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/loss_lyap_norm.png')


fig, ax = subplots()
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel(r"Time-avgd w[0]", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)
for l in range(nn):
    direc =  loadtxt('../'+'dir_names' + str(l) + '.txt',dtype=str)
    c = loadtxt('../'+direc[0]+'/norm_comp.txt')
    c = c[runup:]
    c = cumsum(c)/arange(1, num_el+1)
    #ax.plot(range(num_el)[::skip], c[::skip], color=clr[l], linewidth=3.0, label='{}% noise'.format(n_lvl[l]))
    for i, di in enumerate(direc):
        c = loadtxt('../'+di+'/norm_comp.txt')
        c = c[runup:]
        c = cumsum(c)/arange(1, num_el+1)
        c /= mean(c)
        ax.plot(range(num_el)[::skip], c[::skip], linewidth=3.0)
#ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/loss_stab_comp.png')


