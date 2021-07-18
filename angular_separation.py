import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy . wcs import WCS
from astropy.io import ascii
from astropy.table import Table
from astropy import units as u

def angular_separation(z,l):
    """
    Calcuates the angular separation (in degree) from input redshift and linear_separation;
    z=redshift,l=linear_separation/proper length (in Mpc).
    """
    from astropy import units as u
    from astropy.cosmology import LambdaCDM
    cosmo = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)
    d_A=cosmo.angular_diameter_distance(z).value #Calculates the angular diameter distance, d_A(in Mpc) from redshift
    theta = (180*l/(np.pi*d_A))*u.degree
    return theta;