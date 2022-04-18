#!/usr/bin/python3
from numpy import *
from matplotlib.pyplot import *


direc = sys.argv[1]
loss = loadtxt('../'+direc+'/loss.txt')
acc = loadtxt('../'+direc+'/test_acc.txt')
c = loadtxt('../'+direc+'/norm_comp.txt')
n = loadtxt('../'+direc+'/norm.txt')
runup = 200
racc = 4
dacc = 40
loss = loss[runup:]
acc = 100*acc[racc:-1]
c = c[runup:]
n = n[runup:]
num_el = len(loss)
time_arr = range(1,num_el+1)
loss_ca = cumsum(loss)/time_arr
acc_ca = cumsum(acc)/range(1,len(acc)+1)
c_ca = cumsum(c)/time_arr
n_ca = cumsum(n)/time_arr



fig, ax = subplots()
ax.plot(loss_ca,lw=3.0,color='b')
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel("Training loss", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)
tight_layout()
fig.savefig("../plots/loss_no_data_aug.png")

fig, ax = subplots()
ax.plot(range(racc*dacc,dacc*(len(acc_ca)+racc),dacc),acc_ca,lw=3.0,color='b')
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel("Test accuracy %", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)
tight_layout()
fig.savefig("../plots/test_acc_no_data_aug.png")

fig, ax = subplots()
ax.plot(n_ca,lw=3.0,color='b')
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel("Norm of first layer weights", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)
tight_layout()
fig.savefig("../plots/weight_norm_no_data_aug.png")


fig, ax = subplots()
ax.plot(c_ca,lw=3.0,color='b')
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel("First component of weight vector", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)
tight_layout()
fig.savefig("../plots/weight_comp_no_data_aug.png")

