#!/usr/bin/python3
from numpy import *
from matplotlib.pyplot import *

runup = 50
fs = 16
n_eps = 3
n_per_eps = 4
clr = ["b", "r", "g"]

direc = loadtxt('../'+'dir_names.txt',dtype=str)
fig, ax = subplots()
ax.set_xlabel("Time", fontsize=fs)
ax.set_ylabel("Absolute diff in loss", fontsize=fs)
ax.xaxis.set_tick_params(labelsize=fs)
ax.yaxis.set_tick_params(labelsize=fs)
ax.grid(True)
loss_orig = loadtxt('../'+direc[0]+'/loss.txt')
loss_orig = loss_orig[runup:]
num_el = len(loss_orig)
loss_orig = cumsum(loss_orig)/arange(1, num_el+1)
dloss = zeros((n_eps, num_el))
err_dloss = zeros((n_eps, num_el))
for k in range(n_eps):
    for i, di in enumerate(direc[1+n_per_eps*k:1+n_per_eps*(k+1)]):
        loss = loadtxt('../'+di+'/loss.txt')
        loss = loss[runup:]
        num_el = len(loss)
        loss = cumsum(loss)/arange(1, num_el+1)
        dloss[k] += abs(loss_orig - loss)/n_per_eps 
    ax.plot(dloss[k],lw=3.0,color=clr[k],label=r"$\epsilon = 10^{{{}}}$".format(-(k+1)))
ax.legend(fontsize=fs)
tight_layout()
fig.savefig('../plots/dloss_avg.png')

fig, ax = subplots()

ax.set_xlabel("Time", fontsize=fs)
ax.set_ylabel("Abs diff in test accuracy %", fontsize=fs)
ax.xaxis.set_tick_params(labelsize=fs)
ax.yaxis.set_tick_params(labelsize=fs)
ax.grid(True)
acc_orig = loadtxt('../'+direc[0]+'/test_acc.txt')
acc_orig = acc_orig[5:-1]
num_el = len(acc_orig)
acc_orig = cumsum(acc_orig)/arange(1,num_el+1)
acc_orig = 100*acc_orig
dacc = zeros((n_eps, num_el))
for k in range(n_eps):
    for i, di in enumerate(direc[1+n_per_eps*k:1+n_per_eps*(k+1)]):
        acc = loadtxt('../'+di+'/test_acc.txt')
        acc = acc[5:-1]
        acc = cumsum(acc)/arange(1,len(acc)+1)
        acc = 100*acc
        timeacc = range(len(acc))
        dacc[k] += abs(acc - acc_orig)/n_per_eps
    ax.plot(timeacc, dacc[k],lw=3.0,color=clr[k],label=r"$\epsilon = 10^{{{}}}$".format(-(k+1)))
ax.legend(fontsize=fs)
tight_layout()
fig.savefig('../plots/dacc_avg.png')

fig, ax = subplots()
ax.set_xlabel("Time", fontsize=fs)
ax.set_ylabel("Abs diff in norm of first layer weights", fontsize=fs)
ax.xaxis.set_tick_params(labelsize=fs)
ax.yaxis.set_tick_params(labelsize=fs)
ax.grid(True)
n_orig = loadtxt('../'+direc[0]+'/norm.txt')
n_orig = n_orig[runup:]
num_el = len(n_orig)
n_orig = cumsum(n_orig)/arange(1,num_el+1)
dn = zeros((n_eps, num_el))
for k in range(n_eps):
    for i, di in enumerate(direc[1+k*n_per_eps:1+(k+1)*n_per_eps]):
        n = loadtxt('../'+di+'/norm.txt')
        n = n[runup:]
        n = cumsum(n)/arange(1,len(n)+1)
        dn[k] += abs(n - n_orig)/n_per_eps 
    ax.plot(dn[k],lw=3.0,color=clr[k],label=r"$\epsilon = 10^{{{}}}$".format(-(k+1)))
ax.legend(fontsize=fs)
tight_layout()
fig.savefig('../plots/dweight_norm_avg.png')

fig, ax = subplots()
ax.set_xlabel("Time", fontsize=fs)
ax.set_ylabel("Absolute diff in w[0]", fontsize=fs)
ax.xaxis.set_tick_params(labelsize=fs)
ax.yaxis.set_tick_params(labelsize=fs)
ax.grid(True)
c_orig = loadtxt('../'+direc[0]+'/norm_comp.txt')
c_orig = c_orig[runup:]
num_el = len(c_orig)
c_orig = cumsum(c_orig)/arange(1,num_el+1)
dc = zeros((n_eps, num_el))
for k in range(n_eps):
    for i, di in enumerate(direc[1+k*n_per_eps:1+(k+1)*n_per_eps]):

        c = loadtxt('../'+di+'/norm_comp.txt')
        c = c[runup:]
        c = cumsum(c)/arange(1,len(c)+1)
        dc[k] += abs(c - c_orig)/n_per_eps
    ax.plot(dc[k],lw=3.0,color=clr[k],label=r"$\epsilon = 10^{{{}}}$".format(-(k+1)))
ax.legend(fontsize=fs)
tight_layout()
fig.savefig('../plots/dweight_comp_avg.png')

