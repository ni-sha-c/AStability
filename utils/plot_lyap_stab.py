#!/usr/bin/python3
from numpy import *
from matplotlib.pyplot import *

epsi = 1.e-3
runup = 200
dacc = 1
racc = 1
skip = 1
nn = 5
n_lvl = [0, 10, 17, 25, 50]

clr  = ['silver', 'cornflowerblue', 'violet', 'greenyellow', 'indianred']
clr = ['black', 'darkblue', 'purple', 'darkgreen', 'maroon']

fig, ax = subplots()
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel(r"$|\Delta $ Loss|", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)

for l in range(nn):
    direc = loadtxt('../'+'dir_names' + str(l) + '.txt',dtype=str)
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
    ax.semilogy(range(num_el)[::skip], m_dloss[::skip]/epsi, color=clr[l], linewidth=2.0, label='{}% noise'.format(n_lvl[l]))
ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/loss_lyap_stab.png')



fig, ax = subplots()
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel(r"$|\Delta $ Accuracy| in %", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)

for l in range(nn):
    direc = loadtxt('../'+'dir_names' + str(l) + '.txt',dtype=str)
    acc = loadtxt('../'+direc[0]+'/test_acc.txt')
    acc = acc[runup:-2]
    nacc = len(acc)
    n_dpts = len(direc)
    acces = zeros((n_dpts, nacc))
    for i, di in enumerate(direc):
        acc = loadtxt('../'+di+'/test_acc.txt')
        acc = acc[runup:-2]
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
    ax.semilogy(range(0,dacc*nacc,dacc)[::skip], m_dacc[::skip]/epsi, color=clr[l], lw = 3.0, label='{}% noise'.format(n_lvl[l]))
ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/acc_lyap_stab.png')


fig, ax = subplots()
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel(r"$|\Delta $ ||1st lr wt|| |", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)

for l in range(nn):
    direc = loadtxt('../'+'dir_names' + str(l) + '.txt',dtype=str)
    n_dpts = len(direc)
    nes = zeros((n_dpts, num_el))
    for i, di in enumerate(direc):
        n = loadtxt('../'+di+'/norm.txt')
        n = n[runup:]
        n = cumsum(n)/arange(1, num_el+1)
        nes[i] = n

    k = 0
    dnes = zeros((n_data, num_el))
    m_dn = zeros(num_el)
    for i in range(n_dpts):
        for j in range(i+1, n_dpts):
            dnes[k] = abs(nes[i] - nes[j])
            m_dn += dnes[k]/n_data
            k = k + 1
    err_dn = sum((dnes - m_dn)**2, axis=0)/n_data
    err_dn = sqrt(err_dn/n_data)
    ax.semilogy(range(num_el)[::skip], m_dn[::skip]/epsi, color=clr[l], linewidth=2.0, label='{}% noise'.format(n_lvl[l]))
ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/norm_lyap_stab.png')


fig, ax = subplots()
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel(r"$|\Delta $ w[0] |", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)
for l in range(nn):
    direc =  loadtxt('../'+'dir_names' + str(l) + '.txt',dtype=str)
    n_dpts = len(direc)
    ces = zeros((n_dpts, num_el))
    for i, di in enumerate(direc):
        c = loadtxt('../'+di+'/norm_comp.txt')
        c = c[runup:]
        c = cumsum(c)/arange(1, num_el+1)
        ces[i] = c

    k = 0
    dces = zeros((n_data, num_el))
    m_dc = zeros(num_el)
    for i in range(n_dpts):
        for j in range(i+1, n_dpts):
            dces[k] = abs(ces[i] - ces[j])
            m_dc += dces[k]/n_data
            k = k + 1
    err_dc = sum((dces - m_dc)**2, axis=0)/n_data
    err_dc = sqrt(err_dc/n_data)
    ax.semilogy(range(num_el)[::skip], m_dc[::skip]/epsi, color=clr[l], linewidth=2.0, label='{}% noise'.format(n_lvl[l]))
ax.legend(fontsize=16)
tight_layout()
fig.savefig('../plots/comp_lyap_stab.png')


