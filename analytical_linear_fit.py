#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

def analytical_linear_fit(x,y,y_err):
    '''
    This subroutine fits a linear model (y=ax+b) to the data by solving linear algebra and minimized chi-square to contrain model 
    parameters. It returns the parameter values with 1 sigma errors and the reduced chi-square value. As a plus, it also 
    polts the data and show the best fit line with the slope value. 
    
    Caution: This model is only appropriate for a linear model.
    
    '''
    # create matrices and vectors:
    X = np.vander(x, N=2, increasing=True)
    Cov = np.diag(y_err**2)
    Cinv = np.linalg.inv(Cov) # we need the inverse covariance matrix

    # using the new Python matrix operator
    best_pars_linalg = np.linalg.inv(X.T @ Cinv @ X) @ (X.T @ Cinv @ y)

    # we can also get the parameter covariance matrix
    pars_Cov = np.linalg.inv(X.T @ Cinv @ X)
   
    m = best_pars_linalg[1]
    m_err = np.sqrt(np.diag(pars_Cov))[1]
    b = best_pars_linalg[0]
    b_err = np.sqrt(np.diag(pars_Cov))[0]
   
    m_range = np.linspace(m-m_err,m+m_err,25)
    b_range = np.linspace(b-b_err,b+b_err,25)
   
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
   
    # Defining a function that can calculate chi-square value
    def chisqr(obs, exp, error):
        chisqr = 0
        for i in range(len(obs)):
            chisqr = chisqr + ((obs[i]-exp[i])**2)/(error[i]**2)
        return chisqr
   
    expected = line_model([m,b],x)
    chisqr_val_min = chisqr(y, expected, y_err)
    print('Minimized Chi-Square: %s'%(chisqr_val_min))
    reduced_chisqr_val_min = chisqr_val_min/(len(y)-3)
    sigma = np.sqrt(2/(len(y)-3))
    nsig = (reduced_chisqr_val_min-1)/sigma
    print('Reduced minimized Chi-Square: %s (%s sigma)'%(reduced_chisqr_val_min,nsig))
   
    datastyle = dict(linestyle='none', marker='o', markersize=10,color='k', ecolor='#666666',elinewidth=2)
   
    plt.figure(figsize=(6,5),dpi=100)
    plt.errorbar(x, y, y_err, **datastyle,zorder=4)
   
    x_grid = np.arange(x.min()-0.1, x.max()+0.1, 0.01)
    plt.plot(x_grid, line_model(best_pars_linalg[::-1], x_grid), marker='', color='crimson',linestyle='-',
             lw=3,label='best-fit (slope = %s)'%(np.round(m,2)),zorder=3)
   
    for i in range(len(m_range)):
        for j in range(len(b_range)):
            plt.plot(x_grid, line_model(np.array([m_range[i],b_range[j]]), x_grid), color='#3182bd', marker='',
                     linestyle='-',zorder=2,alpha=0.1)
               
    plt.xlabel('$x$',fontsize=20)
    plt.ylabel('$y$',fontsize=20)
    plt.legend(loc='best',fontsize=12)
    plt.tight_layout()
   
    return m, m_err, b, b_err, reduced_chisqr_val_min;

