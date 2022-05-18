#!/usr/bin/python3
from numpy import *
from matplotlib.pyplot import *
from scipy import signal

racc = 200

direc = loadtxt('../'+'list_stst_resnet.txt',dtype=str)
n_pts_tot = len(direc)
n_pts = 10
ec = ['silver', 'cornflowerblue', 'violet', 'greenyellow', 'indianred']
lc = ['black', 'darkblue', 'purple', 'darkgreen', 'maroon']

fig, ax = subplots()
ax.set_xlabel("Time", fontsize=16)
ax.set_ylabel("Test loss correlation", fontsize=16)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.grid(True)
n_times = 700
auto_corr_acc = zeros((n_pts_tot, n_times))
for l in range(0, n_pts_tot):
    l1 = l
    acc = loadtxt('../'+direc[l1]+'/test_loss.txt')
    acc = acc[racc:]
    nacc = len(acc)
    #print('nacc is', nacc) 
    macc = sum(acc)/nacc
    #vacc = abs(sum((acc - macc)*(acc - macc))/nacc)
    vacc = macc*macc
    #vacc = 1.0
    for k in range(n_times):
        i = 0
        while k+i < nacc:
            auto_corr_acc[l, k] += acc[i]*acc[k+i]
            i = i + 1
        if i != 0:
            auto_corr_acc[l,k] /= i
            auto_corr_acc[l,k] -= macc*macc
    auto_corr_acc[l] = abs(auto_corr_acc[l])/vacc
    #freq, psd_acc = signal.welch(auto_corr_acc[l], 1)
    freq = range(n_times)

    #psd_acc = auto_corr_acc[l]
    #freq = range(nacc)
    #psd_acc = acc
    fun = lambda x, a, b: b*exp(-a*x) 
    if l < n_pts:
        if l == n_pts-1:
            j = 0
            psd_acc = sum(auto_corr_acc[j:j+n_pts],axis=0)/n_pts
            print("hmm?")
            ax.plot(freq, psd_acc, "o", ms=3.0, color=lc[0], label="0 % noise")
            #ax.plot(freq, fun(freq/1800, 8, 5e-14), '--', lw=3.0, color=lc[0], label="fit")
        #else:
        #    ax.plot(freq, psd_acc, lw=3.0, color=lc[0]) 
    elif l < 2*n_pts: 
        if l == 2*n_pts-1:
            j = n_pts
            psd_acc = sum(auto_corr_acc[j:j+n_pts],axis=0)/n_pts

            ax.plot(freq, psd_acc, "v", ms = 3.0, color=lc[1], label="10 % noise")
        #else:
         #   ax.plot(freq, psd_acc, lw=3.0, color=lc[1])
    elif l < 3*n_pts:
        if l == 3*n_pts-1:
            j = 2*n_pts
            psd_acc = sum(auto_corr_acc[j:j+n_pts],axis=0)/n_pts

            ax.plot(freq, psd_acc, "p", ms = 3.0, color=lc[2], label="17 % noise")
        #else:
        #    ax.plot(freq, psd_acc, lw=3.0, color=lc[2])
    elif l < 4*n_pts:
        if l == 4*n_pts-1:
            j = 3*n_pts
            psd_acc = sum(auto_corr_acc[j:j+n_pts],axis=0)/n_pts

            ax.plot(freq, psd_acc, "*", ms = 3.0, color=lc[3], label="25 % noise")
        #else:
        #    ax.plot(freq, psd_acc, lw=3.0, color=lc[3])
    else:
        if l == 5*n_pts-1:
            j = 4*n_pts
            psd_acc = sum(auto_corr_acc[j:j+n_pts],axis=0)/n_pts

            ax.plot(freq, psd_acc, "D", ms = 3.0, color=lc[4], label="50 % noise")
        #else:
        #    ax.plot(freq, psd_acc, lw=3.0, color=lc[4])




ax.legend(fontsize=14, mode="expand", ncol=3)
tight_layout()
fig.savefig('../plots/test_loss_corr_resnet.png')


