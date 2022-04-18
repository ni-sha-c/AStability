#!/usr/bin/python3
from numpy import *
from matplotlib.pyplot import *

runup = 200
dacc = 40
racc = 4
skip = 1
direc = loadtxt('../'+'dir_names.txt',dtype=str)

fig, ax = subplots()
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel(r"$|\Delta $ Loss|", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)
n_dpts = len(direc)
loss = loadtxt('../'+direc[0]+'/loss.txt')
loss = loss[runup:]
num_el = len(loss)
losses = zeros((n_dpts, num_el))
for i, di in enumerate(direc):
    loss = loadtxt('../'+di+'/loss.txt')
    loss = loss[runup:]
    loss = cumsum(loss)/arange(1, num_el+1)
    losses[i] = loss

n_data = int(n_dpts*(n_dpts - 1)//2)
k = 0
dlosses = zeros((n_data, num_el))
m_dloss = zeros(num_el)
for i in range(n_dpts):
    for j in range(i+1, n_dpts):
        dlosses[k] = abs(losses[i] - losses[j])
        m_dloss += dlosses[k]/n_data
        k = k + 1
err_dloss = sum((dlosses - m_dloss)**2, axis=0)/n_data
err_dloss = sqrt(err_dloss)
ax.errorbar(range(num_el)[::skip], m_dloss[::skip], yerr=err_dloss[::skip], fmt="o", ms=6, ecolor="dodgerblue", color="midnightblue", elinewidth=0.4, label='Stoc stab')
ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/loss_sens_orig.png')



fig, ax = subplots()
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel(r"$|\Delta $ Accuracy| in %", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)
acc = loadtxt('../'+direc[0]+'/test_acc.txt')
acc = acc[racc:-1]
nacc = len(acc)
acces = zeros((n_dpts, nacc))
for i, di in enumerate(direc):
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
err_dacc = sqrt(err_dacc)
ax.errorbar(range(0,dacc*nacc,dacc)[::skip], m_dacc[::skip], yerr=err_dacc[::skip], fmt="o", ms=6, ecolor="dodgerblue", color="midnightblue", elinewidth=2.0, label='Stoc stab')
ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/acc_sens_orig.png')




"""
fig, ax = subplots()
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel("Test accuracy %", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)
for i, di in enumerate(direc):
    acc = loadtxt('../'+di+'/test_acc.txt')
    acc = 100*acc[racc:-1]
    lacc = len(acc)
    acc = cumsum(acc)/arange(1,lacc+1)
    timeacc = range(racc*dacc, (racc+lacc)*dacc, dacc)
    if i < len(direc)-1:
        ax.plot(timeacc, acc, lw=3.0,label='i/p pert at ind {}'.format(i))
    else:
        ax.plot(timeacc, acc,lw=3.0, label='orig, first pt rmvd')
ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/test_acc_perts.png')

fig, ax = subplots()
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel("Norm of first layer weights", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)
for i, di in enumerate(direc):
    n = loadtxt('../'+di+'/norm.txt')
    n = n[runup:]
    n = cumsum(n)/arange(1,num_el+1)
    if i < len(direc)-1:
        ax.plot(n,lw=3.0,label='i/p pert at ind {}'.format(i))
    else:
        ax.plot(n,lw=3.0,label='orig, first pt rmvd')
ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/weight_norm_perts.png')

fig, ax = subplots()
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel("First component of weight vector", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)
for i, di in enumerate(direc):
    c = loadtxt('../'+di+'/norm_comp.txt')
    c = c[runup:]
    c = cumsum(c)/arange(1,num_el+1)
    if i < len(direc)-1:
        ax.plot(c,lw=3.0,label='i/p pert at ind {}'.format(i))
    else:
        ax.plot(c,lw=3.0,label='orig, first pt rmvd')
ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/weight_comp_perts.png')
"""
