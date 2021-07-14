#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from scipy.optimize import curve_fit
from scipy.stats.distributions import chi2

def flat_p_optimizer(xdata,ydata,yerr,bmin=-5,bmax=10):
    """
    This subroutine fits a zero slope line by minimizing the chi-square value to the data under the assumption that there is
    no correlation between the x and y values. To reject this null hypothesis, it also calculates the p-vaue from the 
    chi-square distribution, where <0.05 means we can reject our null hypotheis. The soubroutine returns the p-value with 
    estimated 1sigma erros in p and also plots the data with the best-fit zero slope line.

    """


    ## Defining a function that calculates chi-square value
    def chisqr(obs, exp, error):
        chisqr = 0
        for i in range(len(obs)):
            chisqr = chisqr + ((obs[i]-exp[i])**2)/(error[i]**2)
        return chisqr
   
    def line_model(pars, x):
        """
        Evaluate a straight line model at the input x values.

        Parameters
        ----------
        pars : list, array
            This should be a length-2 array or list containing the
            parameter values (a, b) for the (slope, intercept).
        x : numeric, list, array
            The coordinate values.

        Returns
        -------
        y : array
            The computed y values at each input x.
        """
        return pars[0]*np.array(x) + pars[1]

    ## Generating p-values from chi-squares
    X = np.vander([0]*len(xdata), N=2, increasing=True)
    Cov = np.diag(yerr**2)
    Cinv = np.linalg.inv(Cov) # we need the inverse covariance matrix

    # using the new Python matrix operator
    best_pars_linalg = np.linalg.pinv(X.T @ Cinv @ X) @ (X.T @ Cinv @ ydata)

    # we can also get the parameter covariance matrix
    pars_Cov = np.linalg.pinv(X.T @ Cinv @ X)
   
    m = best_pars_linalg[1]
    m_err = np.sqrt(np.diag(pars_Cov))[1]
    b = best_pars_linalg[0]
    b_err = np.sqrt(np.diag(pars_Cov))[0]
   
    expected = [b]*(len(xdata))
    expected_err = [b_err]*(len(xdata))
    chisqr_val_min = chisqr(ydata, expected, yerr)
    chisqr_val_err = chisqr(ydata, expected_err, yerr)
    p = chi2.sf(chisqr_val_min,len(ydata)-2)
    p_err = chi2.sf(chisqr_val_err,len(ydata)-2)
    print('P-value: %s+/-%s'%(p,p_err))
   
    ## Plotting
    plt.figure(figsize=(6,5),dpi=100)
    datastyle = dict(linestyle='none', marker='o', markersize=10,color='k', ecolor='#666666',elinewidth=2)
    plt.errorbar(xdata, ydata, yerr, **datastyle,zorder=4)
   
    x_grid = np.arange(xdata.min()-0.1, xdata.max()+0.1, 0.01)
    plt.plot(x_grid, line_model(np.array([0,b]), x_grid), marker='', color='forestgreen',
             linestyle='--',lw=3,label='horizontal Fit (p-value = %s)'%(np.round(p,2)),zorder=3)
               
    plt.xlabel('$x$',fontsize=20)
    plt.ylabel('$y$',fontsize=20)
    plt.legend(loc='best',fontsize=12)
    plt.tight_layout()
    plt.show()
   
    return p, p_err;


# In[ ]:




