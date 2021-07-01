#!/usr/bin/env python
# coding: utf-8

# In[3]:


x = [0.68,0.35,0.44,0.49,-0.04,1.25]
y = [11.48,2.84,2.62,3.33,3.99,3.37]
yerr= [3.30,2.55,3.20,1.12,2.11,1.17]
xerr = None

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from matplotlib import rc
import pandas as pd
from scipy import stats

x = np.asanyarray(x)
y = np.asanyarray(y)
xerr = np.asanyarray(xerr)
yerr = np.asanyarray(yerr)

def spearman_stats(x,y):
    d = {'x': x, 'y': y}
    df = pd.DataFrame(data=d)
    df["Rx"] = df['x'].rank()
    df["Ry"] = df['y'].rank()
    df["d_sqr"] = (df["Rx"]-df["Ry"])**2
    N = len(df["Rx"])
    rs = 1 - (6*np.sum(df["d_sqr"]))/(N**3-N)
    z_score = np.arctanh(rs)*np.sqrt((N-3)/1.06)
    t_score = rs*np.sqrt((N-2)/(1-rs**2))
    return rs, z_score, t_score; 

#Spearman Stattistics without Errors
rs_mean,z_score_mean,t_score_mean  = spearman_stats(x,y)
print('Spearman Stattistics without Errors:')
print(r'Mean Spearman Rank Correlation: %f'%(rs_mean))
print('Mean z-score: %f'%(z_score_mean))
print('Mean t-score: %f'%(t_score_mean))

#Spearman Stattistics with Errors
print('\nSpearman Stattistics with Errors:')
Nboot = int(input('Number of Bootstrapping Step:')); mu = 0; sigma = 1

data_ids = np.arange(0,len(y),1)
resamp_ids = np.random.choice(data_ids,replace=True,size=(Nboot,len(data_ids)))
Yj = np.array([np.array(y)[resamp_ids[item]] for item in range(Nboot)])
Xj = np.array([np.array(x)[resamp_ids[item]] for item in range(Nboot)])

if xerr.all() == None:
    Xerri = 0
    print('No X error provided.')
else:
    Xerri = [np.array(xerr)[resamp_ids[item]] for item in range(Nboot)]
    
if yerr.all() == None:
    Yerri = 0
    print('No Y error provided.')
else:
    Yerri = [np.array(yerr)[resamp_ids[item]] for item in range(Nboot)]
    
G = np.random.normal(loc=0.0, scale=1.0, size=(Nboot,len(data_ids)))
yi = Yj + G*Yerri
xi = Xj + G*Xerri
rs_resample = []; z_score_resample = []; t_score_resample = [];
for item in range(Nboot):
    rs, z_score, t_score = spearman_stats(xi[item],yi[item]) 
    rs_resample.append(rs)
    z_score_resample.append(z_score)
    t_score_resample.append(t_score)
rs_resample = np.array(rs_resample)
z_score_resample = np.array(z_score_resample)
t_score_resample = np.array(t_score_resample)

rs_2sigma_l,rs_1sigma_l,rs_median,rs_1sigma_u,rs_2sigma_u = np.percentile(rs_resample,[2.5,15.85,50,84.15,97.5])
z_score_2sigma_l,z_score_1sigma_l,z_score_median,z_score_1sigma_u,z_score_2sigma_u = np.percentile(z_score_resample,[2.5,15.85,50,84.15,97.5])
t_score_2sigma_l,t_score_1sigma_l,t_score_median,t_score_1sigma_u,t_score_2sigma_u = np.percentile(t_score_resample,[2.5,15.85,50,84.15,97.5])

print(r'Spearman Rank Correlation Confidence (95%%): [%f, %f]'%(rs_2sigma_l,rs_2sigma_u))
print('z-score Confidence (95%%): [%f, %f]'%(z_score_2sigma_l,z_score_2sigma_u))
print('t-score Confidence (95%%): [%f, %f]'%(t_score_2sigma_l,t_score_2sigma_u))

z_critical  = stats.norm.ppf(1-.05/2)
print('Critical z-score: %f'%z_critical)
t_critical  = stats.t.ppf(q=1-0.025, df=len(x)-2)
print('Critical t-score: %f\n'%t_critical)

if z_score_2sigma_u>z_critical or z_score_2sigma_l<(-z_critical):
    print('The Null Hypothesis is Rejected (based on z-score).')
else:
    print('We Cannot Reject the Null Hypothesis (based on z-score).')

if t_score_2sigma_u>t_critical or t_score_2sigma_l<(-t_critical):
    print('The Null Hypothesis is Rejected (based on t-score).')
else:
    print('We Cannot Reject the Null Hypothesis (based on t-score).')
    
#plotting
mu_rs = np.mean(rs_resample)
sigma_rs = np.std(rs_resample)
mu_z_score = np.mean(z_score_resample)
sigma_z_score = np.std(z_score_resample)
mu_t_score = np.mean(t_score_resample)
sigma_t_score = np.std(t_score_resample)

resamples = [rs_resample,z_score_resample,t_score_resample]

mu = [mu_rs, mu_z_score, mu_t_score]
sigma = [sigma_rs, sigma_z_score, sigma_t_score]

sigma_l = [rs_2sigma_l,z_score_2sigma_l,t_score_2sigma_l]
sigma_u = [rs_2sigma_u,z_score_2sigma_u,t_score_2sigma_u]

rows= 1
columns = 3
loc = [2,1,1]
title = ["Spearman's Rank Correlation Coefficient","z-score","t-score"]
labels = ['','> critical z','> critical t']
colors = ['C0','aqua','aqua']

plt.rcParams['axes.edgecolor'] = "k"
plt.rcParams['axes.linewidth'] = 4
plt.rcParams.update({'font.size': 28})
legend_properties = {'weight':'normal'}

fig, axs = plt.subplots(nrows=rows,ncols=columns, figsize=[columns*10,rows*10],dpi=200,sharey=False)
axs =axs.flatten()

for i in range(3):
    count, bins, ignored = axs[i].hist(resamples[i],30,color= colors[i],density=True,label=labels[i],alpha=1)
    axs[i].plot(bins, 1/(sigma[i] * np.sqrt(2 * np.pi))*np.exp( - (bins - mu[i])**2 / (2 * sigma[i]**2) ),
                linewidth=7, color='r',label='PDF')
    axs[i].vlines(x=mu[i],ymin=0,ymax=count.max()*0.9,ls='--',lw=6.5,color='lime',label='Mean')
    axs[i].vlines(x=sigma_l[i],ymin=0,ymax=count.max()*0.3,ls='-.',lw=6.5,color='fuchsia',label='95% Confidence')
    axs[i].vlines(x=sigma_u[i],ymin=0,ymax=count.max()*0.3,ls='-.',lw=6.5,color='fuchsia')
    axs[i].set_title('%s'%(title[i]),fontsize=26,fontweight='bold')
    axs[i].legend(loc=loc[i],fontsize=23,prop=legend_properties,markerscale=1,handlelength=0.5,labelspacing=0.2)

    axs[i].tick_params(axis = 'both', which = 'major',direction='out', length=12, width=2)
    axs[i].tick_params(axis = 'both', which= 'minor', direction='out', length=8, width=1.5, color='k')
    axs[i].xaxis.set_ticks_position('bottom')
    axs[i].yaxis.set_ticks_position('left')
    axs[i].xaxis.set_minor_locator(AutoMinorLocator())
    axs[i].yaxis.set_minor_locator(AutoMinorLocator())

hist_z, bins_z = np.histogram(resamples[1], bins=30,density=True)
thresh_z = z_critical
mask_z = bins_z < thresh_z
below_thresh_z = np.array(bins_z[mask_z].tolist() + [thresh_z])
axs[1].bar(below_thresh_z[:-1], hist_z[mask_z[:-1]], width=np.diff(below_thresh_z), color='C0',align='edge')

hist_t, bins_t = np.histogram(resamples[2], bins=30,density=True)
thresh_t = t_critical
mask_t = bins_t < thresh_t
below_thresh_t = np.array(bins_t[mask_t].tolist() + [thresh_t])
axs[2].bar(below_thresh_t[:-1], hist_t[mask_t[:-1]], width=np.diff(below_thresh_t), color='C0',align='edge')

plt.tight_layout()
plt.show()

