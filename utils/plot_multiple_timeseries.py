#!/usr/bin/python3
from numpy import *
from matplotlib.pyplot import *

runup = 50
dacc = 40
racc = 4
direc = loadtxt('../'+'dir_names.txt',dtype=str)
fig, ax = subplots()
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel("Training loss", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)

for i, di in enumerate(direc):
    loss = loadtxt('../'+di+'/loss.txt')
    loss = loss[runup:]
    num_el = len(loss)
    loss = cumsum(loss)/arange(1, num_el+1)
    if i < len(direc)-1:
        ax.plot(loss,lw=3.0,label='i/p pert at ind {}'.format(i))
    else:
        ax.plot(loss,lw=3.0,label='orig, first pt rmvd')
ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/loss_perts.png')

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

