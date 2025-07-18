import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import camb
from camb import model, initialpower
import scipy.integrate as integ
from scipy.special import factorial
from scipy.integrate import quad
from scipy.special import erf, erfc
from scipy.spatial import KDTree
from scipy.optimize import root_scalar
import scipy.special as sp
from tqdm import tqdm

import h5py
import yt
from yt.extensions.astro_analysis.halo_analysis import HaloCatalog
import sys, platform, os

from astropy.cosmology import FlatLambdaCDM
from astropy import units as u


#Lambda CDM parameters used here
H0=67.00 # km/s/Mpc
h=H0/100. # reduced hubble parameter
TCMB=2.7255 # CMB temperature
s8=0.8344 # sigma 8
ns=0.96 # slope of primordial power spectrum
As = 2e-9

ob=0.049 # lambdaCDM parameter

om=0.3175
ov=1.0-om 
omk=0.0

z=0.0 #we needed, we'll do the study at z=0

maxkh=1e3
minkh=1e-4

G=6.67e-11 #m^3 kg^−1 s^−2
M_sun = 1.989e30  # kg
Mpc_to_m = 3.086e22  # meters
delta_c = 1.68647 # critical overdensity (Sheth, Tormen, 2002)

rho_0 = om * 3 * (H0*1e3/Mpc_to_m)**2 / 8 / np.pi / G # mean density in kg/m^-3
rho_0_Msun_Mpc3 = rho_0/(M_sun)*(Mpc_to_m)**3 # mean density in Msun/Mpc^3 
rho_0_h2_Msun_Mpc3 = rho_0/(M_sun/h)*(Mpc_to_m/h)**3 # mean density in h^2*Msun/Mpc^3 

L=164 #Mpc/h

def compute_age_Gyr(z, H0=H0, om=om):
    """
    Compute the age of the universe at a given redshift in Gyr.
    
    Inputs : 
    z : Redshift
    H0 : Hubble constant in km/s/Mpc
    om : Matter density parameter (Omega_m)
    
    Outputs : 
    age : Age of the universe at redshift z in Gyr
    """
    H0_astropy = H0 * u.km / u.s / u.Mpc
    cosmo = FlatLambdaCDM(H0=H0_astropy, Om0=om)
    age = cosmo.age(z).to(u.Gyr).value
    return age

def compute_mpart(rho, Npart, L):
    """
    Compute the mass of a particle given the density, number of particles and box size.
    
    Inputs : 
    rho : Density in Msun/h/Mpc^3
    Npart : Number of particles
    L : Box size in Mpc/h
    
    Outputs :
    mp : Mass of a particle in Msun/h
    """
    return rho*(L/h*3.086e22)**3/Npart

def compute_init_power_spectrum(z, H0, TCMB, om, ob ,h, ns, As, maxkh=maxkh, omk=0.0):
    """
    Compute the initial power spectrum using CAMB.
    
    Inputs :
    z : Redshift 
    H0 : Hubble constant in km/s/Mpc
    TCMB : CMB temperature in K
    om : Total matter density parameter (Omega_m)
    ob : Baryon density parameter (Omega_b)
    h : Reduced Hubble parameter (H0/100)
    ns : Spectral index of primordial power spectrum
    As : Amplitude of primordial power spectrum
    maxkh : Maximum k/h value for the power spectrum
    omk : Curvature density parameter (default is 0 for flat universe)
    
    Outputs :
    pars : CAMB parameters object
    results : CAMB results object
    s8_fid : Fiducial sigma8 value
    """
    pars = camb.set_params(H0=H0, TCMB=TCMB, ombh2=ob*h**2, omch2=(om-ob)*h**2, ns=ns, As=As, omk=omk, )
    pars.set_matter_power(redshifts=[0., z+1e-6], kmax=maxkh)

    results = camb.get_results(pars)
    s8_fid = results.get_sigma8_0()
    return pars,results,s8_fid

def compute_matter_power_spectrum(z, pars, s8_fid, ns=ns, s8=s8, minkh=minkh, maxkh=maxkh, npoints=500):
    """
    Compute the matter power spectrum using CAMB
    
    Inputs :
    z : Redshift 
    pars : CAMB parameters object
    s8_fid : Fiducial sigma8 value
    ns : Spectral index of primordial power spectrum
    s8 : Current sigma8 value
    minkh : Minimum k/h value for the power spectrum
    maxkh : Maximum k/h value for the power spectrum
    npoints : Number of points in the power spectrum
    
    Outputs :
    kh : Array of k/h values in h/Mpc
    zcamb : Redshift at which the power spectrum is computed (should be close to z)
    pk : Array of power spectrum values in (Mpc/h)^3
    """
    
    As_corr = As*s8**2/s8_fid**2
    pars.InitPower.set_params(As=As_corr, ns=ns)
    pars.set_matter_power(redshifts=[0., z+1e-6], kmax=maxkh)
    
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, zcamb, pk = results.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, npoints=npoints)
    
    return kh, zcamb, pk

def compute_sig_camb(R, kh, pk):
    '''
    Compute the variance of the density field at scale R using CAMB power spectrum
    
    Inputs :
    R : radius of the region in Mpc/h
    kh : Array of k/h values in h/Mpc
    pk : Array of power spectrum values in (Mpc/h)^3
    
    Outputs :
    sig_camb is unitless
    '''
    x = kh * R
    w = 3 * (np.sin(x) - x * np.cos(x)) / x**3
    pconv = pk[-1, :] / (2 * np.pi)**3
    sig_camb = np.sqrt(integ.trapezoid((pconv) * 4 * np.pi * kh**2 * w**2, kh))
    return sig_camb

def derivative_of_sig_camb(R, kh, pk, h=1e-5):
    '''
    Computes the derivative of the variance of the density field at scale R using finite differences
    
    Inputs :
    R : radius of the region in Mpc/h
    kh : Array of k/h values in h/Mpc
    pk : Array of power spectrum values in (Mpc/h)^3
    h is the step size for finite difference
    
    Outputs :
    dsig_dR : Derivative of the variance with respect to R
    '''
    sig_plus = compute_sig_camb(R + h, kh, pk)
    sig_minus = compute_sig_camb(R - h, kh, pk)
    dsig_dR = (sig_plus - sig_minus) / (2 * h)
    return dsig_dR

def M_to_R(M):
    '''
    Converts mass to radius using the mean density of the Universe (always assumed to be true) for a spherical collapse
    
    M in M_sun/h
    R in Mpc/h
    '''
    return (3 * M / (4 * np.pi * rho_0_h2_Msun_Mpc3))**(1/3)

def R_to_M(R):
    '''
    Converts radius to mass using the mean density of the Universe (always assumed to be true) for a spherical collapse
    
    R in Mpc/h
    M in M_sun/h
    '''
    return 4 / 3 * np.pi * rho_0_h2_Msun_Mpc3 * R**3 

def dR_dM_fct(M):
    '''
    Computes the derivative of radius with respect to mass for a spherical collapse
    
    M in M_sun/h
    R in Mpc/h
    '''
    return (3/(4*np.pi*rho_0_h2_Msun_Mpc3))**(1/3) * M**(-2/3) / 3


def compute_n_PS(M, kh, pk):
    '''
    Computes the Press-Schechter mass function
    
    Inputs :
    M : Mass of the region in Msun/h
    kh : Array of k/h values in h/Mpc
    pk : Array of power spectrum values in (Mpc/h)^3
    
    Outputs :
    n_PS : Press-Schechter mass function value at mass M in h^4/Msun/Mpc^-3
    
    formula : $\displaystyle n_{PS}(M)=\sqrt{\frac{2}{\pi}}\left| \frac{d\sigma}{dM} \right|\frac{\rho_0}{M}\frac{\delta_c}{\sigma(M,z)^2}\exp(-\frac{\delta_c^2}{2\sigma(M,z)^2})$

    with :

    $\displaystyle \frac{d\sigma}{dM}=\frac{d\sigma}{dR}\frac{dR}{dM}=\frac{d\sigma}{dR}\cdot\frac{1}{3}(\frac{3}{4\pi\rho_0})^{\frac{1}{3}}M^{-\frac{2}{3}}$
    '''
    R = M_to_R(M)
    sig_camb = compute_sig_camb(R, kh, pk)

    #dsig/dM = dsig/dR * dR/dM
    dsig_dR = derivative_of_sig_camb(R, kh, pk)
    dR_dM =  dR_dM_fct(M)
    
    dsig_dM = dsig_dR * dR_dM
    
    n_PS = np.sqrt(2/np.pi) * np.abs(dsig_dM) * (rho_0_h2_Msun_Mpc3 / M) * (delta_c / sig_camb**2) * np.exp(-delta_c**2 / 2 / sig_camb**2)
    return n_PS

def compute_n_ST(M,kh,pk):
    '''
    Computes the Sheth-Tormen mass function
    
    Inputs :
    M : Mass of the region in Msun/h
    kh : Array of k/h values in h/Mpc
    pk : Array of power spectrum values in (Mpc/h)^3
    
    Outputs :
    n_ST : Sheth-Tormen mass function value at mass M in h^4/Msun/Mpc^-3
    
    formula : $\displaystyle n_{ST}(M)=A\sqrt{\frac{2a}{\pi}}\left| \frac{d\sigma}{dM} \right|\frac{\rho_0}{M}\left[1+\left(\frac{\sigma^2}{a\delta_c^2}\right)^{0.3}\right]\frac{\delta_c}{\sigma^2}\exp\left(-\frac{a\delta_c^2}{2\sigma^2}\right)$

    with :

    $\displaystyle \frac{d\sigma}{dM}=\frac{d\sigma}{dR}\frac{dR}{dM}$
    '''
    a=0.707
    A=0.3222
    
    R = M_to_R(M)
    
    sig_camb = compute_sig_camb(R, kh, pk)
    
    #dsig/dM = dsig/dR * dR/dM
    dsig_dR = derivative_of_sig_camb(R, kh, pk)
    dR_dM =  dR_dM_fct(M)
    dsig_dM = dsig_dR * dR_dM    
    
    n_ST = A * np.sqrt(2*a/np.pi) * np.abs(dsig_dM) * (rho_0_h2_Msun_Mpc3 / M) * (1+(sig_camb**2/a/delta_c**2)**0.3) * (delta_c / sig_camb**2) * np.exp(-a*delta_c**2 / 2 / sig_camb**2)
    return n_ST

def hmfcalc(M,kh,pk,model):
    '''
    Computes the HMF depending on the chosen model
    
    Inputs :
    M : Mass of the region in Msun/h
    kh : Array of k/h values in h/Mpc
    pk : Array of power spectrum values in (Mpc/h)^3
    model : "PS" for Press-Schechter or "ST" for Sheth-Tormen
    
    Outputs :
    hmf : HMF value at mass M in h^4/Msun/Mpc^-3
    '''
    if model == "PS" :
        hmf = compute_n_PS(M, kh, pk)
    elif model == "ST":
        hmf = compute_n_ST(M, kh, pk)
    else :
        print("Model not recognized")
        hmf = None
    return hmf


def n_EPS(M, kh, pk, R0, delta_L):
    '''
    Computes the CMF using the Extended Press-Schechter (EPS) formalism on a spherical region of radius R0
    
    Inputs :
    M : Mass of the region in Msun/h
    kh : Array of k/h values in h/Mpc
    pk : Array of power spectrum values in (Mpc/h)^3
    R0 : size of the spherical region in Mpc/h
    delta_L : linear overdensity (unitless)
    
    Outputs :
    hmf : HMF value at mass M in h^4/Msun/Mpc^-3
    '''
    R = M_to_R(M)

    
    sig_camb = compute_sig_camb(R, kh, pk)
    sig_camb_0 = compute_sig_camb(R0, kh, pk)
    
    delta_sig = sig_camb**2-sig_camb_0**2

    dsig_dR = derivative_of_sig_camb(R, kh, pk)
    dR_dM =  (3/4/np.pi/rho_0_h2_Msun_Mpc3)**(1/3) * M**(-2/3) / 3
    dsig_dM = dsig_dR * dR_dM

    first_term = np.sqrt(2/np.pi) * np.abs(dsig_dM)
    second_term = rho_0_h2_Msun_Mpc3 / M
    third_term = sig_camb * (delta_c-delta_L) / (delta_sig)**(3/2)
    exp_argument = -(delta_c-delta_L)**2 / 2 / delta_sig
    fourth_term = np.exp(exp_argument)
    
    n_EPS_val = first_term * second_term * third_term * fourth_term
    # n_EPS_val = np.sqrt(2/np.pi) * np.abs(dsig_dM) * (rho_0_h2_Msun_Mpc3 / M) * (sig_camb*(delta_c-delta_0) /(sig_camb**2-sig_camb_0**2)**(3/2)) * np.exp(-delta_c**2 / 2 / (sig_camb**2-sig_camb_0**2))
    return n_EPS_val

def n_EPS_scaled(M, kh, pk, R0, delta_L):
    '''
    Computes the CMF using the Extended Press-Schechter (EPS) formalism on a spherical region of radius R0
    but sigma is scaled up by 5% on average as in Meriot & Semelin (2024)
    
    Inputs :
    M : Mass of the region in Msun/h
    kh : Array of k/h values in h/Mpc
    pk : Array of power spectrum values in (Mpc/h)^3
    R0 : size of the spherical region in Mpc/h
    delta_L : linear overdensity (unitless)
    
    Outputs :
    CMF : CMF value at mass M in h^4/Msun/Mpc^-3
    '''
    R = M_to_R(M)
    
    sig_camb = compute_sig_camb(R, kh, pk) 
    sig_camb_0 = compute_sig_camb(R0, kh, pk)
    
    ###
    ### fig 2. Meriot & Semelin (2024) : "the σ in the EPS model has been scaled up by 5% on average."
    sig_camb = sig_camb*1.05
    ###
    
    delta_sig = sig_camb**2-sig_camb_0**2
    
    dsig_dR = derivative_of_sig_camb(R, kh, pk)
    dR_dM =  (3/4/np.pi/rho_0_h2_Msun_Mpc3)**(1/3) * M**(-2/3) / 3
    dsig_dM = dsig_dR * dR_dM
    
    # breaking terms down to find my error
    first_term = np.sqrt(2/np.pi) * np.abs(dsig_dM)
    second_term = rho_0_h2_Msun_Mpc3 / M
    third_term = sig_camb * (delta_c-delta_L) / (delta_sig)**(3/2)
    exp_argument = -(delta_c-delta_L)**2 / 2 / delta_sig
    fourth_term = np.exp(exp_argument)
    
    n_EPS_val = first_term * second_term * third_term * fourth_term
    # n_EPS_val = np.sqrt(2/np.pi) * np.abs(dsig_dM) * (rho_0_h2_Msun_Mpc3 / M) * (sig_camb*(delta_c-delta_0) /(sig_camb**2-sig_camb_0**2)**(3/2)) * np.exp(-delta_c**2 / 2 / (sig_camb**2-sig_camb_0**2))
    return n_EPS_val

# To compute the CMF using the ST formalism, we need to define a few functions first
def B_func(sig2):
    '''
    Mass dependant threshold function
    input is SQUARED
    
    Inputs :
    sig2 : Variance of the density field at scale R in (Mpc/h)^2
    
    Outputs :
    res : B function value at sig2 in unitless
    '''
    a=0.707
    alpha=0.615
    beta=0.485
    res = np.sqrt(a)*delta_c*(1+beta*(a*delta_c**2/sig2)**(-alpha))
    return res

def B_nth_derivative(sig2, n):
    '''
    Computes the nth derivative of the B function with respect to sig2
    Formula :
    $\frac{\partial^n B}{\partial y^n} = \beta(a\delta_c^2)^{-\alpha+1/2}\frac{\alpha!}{(\alpha-n)!}y^{\alpha-n}$
    
    Inputs :
    sig2 : Variance of the density field at scale R in (Mpc/h)^2
    n : Order of the derivative to compute (integer)
    
    Outputs :
    res : nth derivative of the B function at sig2 in unitless
    '''
    a=0.707
    alpha=0.615
    beta=0.485
    if n==0:
        return B_func(sig2)
    res = beta*(a*delta_c**2)**(-alpha+0.5) * factorial(alpha)/factorial(alpha - n, extend='complex') * (sig2)**(alpha-n) #extend = 'complex' to be able to put negative values inside the factorial function
    return res

def T_func(sig, sig0, delta_L, order_of_taylor_series=5):
    """
    Computes the Taylor expansion of the T function around sig0 up to the order_of_taylor_series
    
    Inputs :
    sig : Variance of the density field at scale R in (Mpc/h)^2
    sig0 : Variance of the density field at scale R0 in (Mpc/h)^2
    delta_0 : linear overdensity (unitless)
    order_of_taylor_series : Order of the Taylor series expansion (default is 5)
    
    Outputs :
    res : Taylor expansion of the T function at sig in unitless
    """
    
    res = 0
    delta_sig = sig**2 - sig0**2
    for n in range(order_of_taylor_series+1):
        if n == 0:
            der = B_func(sig**2)-delta_L
        else :
            der = B_nth_derivative(sig**2,n)
        res += (-delta_sig)**n * der / factorial(n) 
    return res

def n_cST(M, kh, pk, R0, delta_L):
    '''
    Computes the CMF using the conditional Sheth-Tormen (cST) formalism on a spherical region of radius R0
    
    Inputs :
    M : Mass of the region in Msun/h
    kh : Array of k/h values in h/Mpc
    pk : Array of power spectrum values in (Mpc/h)^3
    R0 : size of the spherical region in Mpc/h
    delta_L : linear overdensity (unitless)
    
    Outputs :
    CMF : CMF value at mass M in h^4/Msun/Mpc^-3
    '''
    R = M_to_R(M)
    
    sig_camb = compute_sig_camb(R, kh, pk)
    sig_camb_0 = compute_sig_camb(R0, kh, pk)
    delta_sig = sig_camb**2 - sig_camb_0**2
    epsilon = 1e-10
    delta_sig = delta_sig + epsilon  #for delta_sig no never be 0
    
    dsig_dR = derivative_of_sig_camb(R, kh, pk)
    dR_dM =  (3/4/np.pi/rho_0_h2_Msun_Mpc3)**(1/3) * M**(-2/3) / 3
    dsig_dM = dsig_dR * dR_dM
    
    T_res = T_func(sig_camb, sig_camb_0, delta_L)
    B_res = B_func(sig_camb**2)
    
    first_term = np.sqrt(2/np.pi) * np.abs(dsig_dM)
    second_term = rho_0_h2_Msun_Mpc3 / M
    third_term = np.abs(T_res)
    fourth_term = sig_camb/delta_sig**(3/2)
    
    exp_argument = -(B_res-delta_L)**2 / 2 / delta_sig  
    fifth_term = np.exp(exp_argument)
    
    n_cST_val = first_term * second_term * third_term * fourth_term * fifth_term
    # n_cST_val = np.sqrt(2/np.pi) * np.abs(dsig_dM) * (rho_0_h2_Msun_Mpc3 / M) * np.abs(T_res)*sig_camb/delta_sig**(3/2) * np.exp(-(B_res-delta_0)**2 / 2 / delta_sig)
    return n_cST_val


def n_EPS2(M, kh, pk, R0, delta_L, delta_NL):
    '''
    Computes the CMF using the Extended Press-Schechter (EPS) formalism on a spherical region of radius R0
    DIFFERENCE HERE : we take into account the non-linear evolution of the overdensity by scaling R0 
    with (1+delta_NL)**(1/3), this new radius is called the Lagrangian radius
    
    Inputs :
    M : Mass of the region in Msun/h
    kh : Array of k/h values in h/Mpc
    pk : Array of power spectrum values in (Mpc/h)^3
    R0 : size of the spherical region in Mpc/h
    delta_L : linear overdensity (unitless)
    
    Outputs :
    CMF : CMF value at mass M in h^4/Msun/Mpc^-3
    '''
    R = M_to_R(M)

    R0 = R0 * (1+delta_NL)**(1/3)
    
    sig_camb = compute_sig_camb(R, kh, pk)
    sig_camb_0 = compute_sig_camb(R0, kh, pk)
    
    delta_sig = sig_camb**2-sig_camb_0**2
    epsilon = 1e-10
    delta_sig = delta_sig + epsilon  #for delta_sig to never be 0
    
    dsig_dR = derivative_of_sig_camb(R, kh, pk)
    dR_dM =  (3/4/np.pi/rho_0_h2_Msun_Mpc3)**(1/3) * M**(-2/3) / 3
    dsig_dM = dsig_dR * dR_dM

    
    # breaking terms down to find my error
    first_term = np.sqrt(2/np.pi) * np.abs(dsig_dM)
    second_term = rho_0_h2_Msun_Mpc3 / M
    third_term = sig_camb * (delta_c-delta_L) / (delta_sig)**(3/2)
    exp_argument = -(delta_c-delta_L)**2 / 2 / delta_sig
    fourth_term = np.exp(exp_argument)
    
    n_EPS_val = first_term * second_term * third_term * fourth_term
    # n_EPS_val = np.sqrt(2/np.pi) * np.abs(dsig_dM) * (rho_0_h2_Msun_Mpc3 / M) * (sig_camb*(delta_c-delta_0) /(sig_camb**2-sig_camb_0**2)**(3/2)) * np.exp(-delta_c**2 / 2 / (sig_camb**2-sig_camb_0**2))
    return n_EPS_val

def n_cST2(M, kh, pk, R0, delta_L, delta_NL):
    '''
    Computes the CMF using the conditional Sheth-Tormen (cST) formalism on a spherical region of radius R0
    DIFFERENCE HERE : we take into account the non-linear evolution of the overdensity by scaling R0 
    with (1+delta_NL)**(1/3), this new radius is called the Lagrangian radius

    Inputs :
    M : Mass of the region in Msun/h
    kh : Array of k/h values in h/Mpc
    pk : Array of power spectrum values in (Mpc/h)^3
    R0 : size of the spherical region in Mpc/h
    delta_L : linear overdensity (unitless)
    
    Outputs :
    CMF : CMF value at mass M in h^4/Msun/Mpc^-3
    '''
    R = M_to_R(M)
    
    R0 = R0 * (1+delta_NL)**(1/3)
    
    sig_camb = compute_sig_camb(R, kh, pk)
    sig_camb_0 = compute_sig_camb(R0, kh, pk)
    delta_sig = sig_camb**2 - sig_camb_0**2
    epsilon = 1e-10
    delta_sig = delta_sig + epsilon  #for delta_sig to never be 0
    
    dsig_dR = derivative_of_sig_camb(R, kh, pk)
    dR_dM =  (3/4/np.pi/rho_0_h2_Msun_Mpc3)**(1/3) * M**(-2/3) / 3
    dsig_dM = dsig_dR * dR_dM
    
    T_res = T_func(sig_camb, sig_camb_0, delta_L)
    B_res = B_func(sig_camb**2)
    
    # breaking terms down to find an error
    first_term = np.sqrt(2/np.pi) * np.abs(dsig_dM)
    second_term = rho_0_h2_Msun_Mpc3 / M
    third_term = np.abs(T_res)
    fourth_term = sig_camb/delta_sig**(3/2)
    
    exp_argument = -(B_res-delta_L)**2 / 2 / delta_sig  
    fifth_term = np.exp(exp_argument)
    
    n_cST_val = first_term * second_term * third_term * fourth_term * fifth_term
    # n_cST_val = np.sqrt(2/np.pi) * np.abs(dsig_dM) * (rho_0_h2_Msun_Mpc3 / M) * np.abs(T_res)*sig_camb/delta_sig**(3/2) * np.exp(-(B_res-delta_0)**2 / 2 / delta_sig)
    # if np.isnan(n_cST_val) or np.isinf(n_cST_val):
    #     print(f"NaN value found with parameters : delta_L = {delta_L:.2f} | M = {M:.2e} Msun/h | R = {R:.2e} Mpc/h | R0 = {R0:.2e} Mpc/h")
    return n_cST_val

def cmfcalc(M, kh, pk, R0, delta_L, delta_NL, model):
    '''
    Inputs :
    M : Mass of the region in Msun/h
    kh : Array of k/h values in h/Mpc
    pk : Array of power spectrum values in (Mpc/h)^3
    R0 : size of the spherical region in Mpc/h
    delta_L : linear overdensity (unitless)
    model : "EPS" for Extended Press-Schechter, "EPS_scaled" for EPS with a 5% increase in sigma, "cST" for conditional Sheth-Tormen, "QEPS" for EPS with Lagrangian radius, "QcST" for cST with Lagrangian radius
    
    Outputs :
    CMF : CMF value at mass M in h^4/Msun/Mpc^-3
    '''
    if model == "EPS":
        cmf = n_EPS(M, kh, pk, R0, delta_L)
    elif model == "EPS_scaled":
        cmf = n_EPS_scaled(M, kh, pk, R0, delta_L)
    elif model == "cST":
        cmf = n_cST(M, kh, pk, R0, delta_L)
    elif model == "QEPS" :
        cmf = n_EPS2(M, kh, pk, R0, delta_L ,delta_NL)
    elif model == "QcST":
        cmf = n_cST2(M, kh, pk, R0, delta_L, delta_NL)
    else :
        print("Model not recognized")
        cmf = None
    if np.isnan(cmf):
        return 0
    return cmf


def Mcoll(Mmin, Mregion, kh, pk, R0, delta_L, delta_NL, model):
    """
    Computes the collapsed mass in a spherical region of radius R0 using the CMF calculated with cmfcalc
    
    Inputs :
    Mmin : Minimum mass to integrate from in Msun/h
    Mregion : Mass of the region in Msun/h
    kh : Array of k/h values in h/Mpc
    pk : Array of power spectrum values in (Mpc/h)^3
    R0 : Size of the spherical region in Mpc/h
    delta_L : Linear overdensity (unitless)
    delta_NL : Non-linear overdensity (unitless)
    model : "EPS" for Extended Press-Schechter, "EPS_scaled" for EPS with a 5% increase in sigma, "cST" for conditional Sheth-Tormen, "QEPS" for EPS with Lagrangian radius, "QcST" for cST with Lagrangian radius
    
    Outputs :
    Mcoll : Collapsed mass in the region in Msun/h
    """
    
    Vreg = 4/3*np.pi*(1+delta_NL)*R0**3

    def integrand(x):
        cmf_value = cmfcalc(x, kh, pk, R0,  delta_L, delta_NL, model)
        return cmf_value * x

    integral, abserr = quad(integrand, Mmin, Mregion, limit=100)
    
    return Vreg*integral

def fcoll(Mmin, Mregion, kh, pk, R0, delta_L, delta_NL, model):
    """
    Computes the fraction of collapsed mass in a spherical region of radius R0
    
    Inputs :
    Mmin : Minimum mass to integrate from in Msun/h
    Mregion : Mass of the region in Msun/h
    kh : Array of k/h values in h/Mpc
    pk : Array of power spectrum values in (Mpc/h)^3
    R0 : Size of the spherical region in Mpc/h
    delta_L : Linear overdensity (unitless)
    delta_NL : Non-linear overdensity (unitless)
    model : "EPS" for Extended Press-Schechter, "EPS_scaled" for EPS with a 5% increase in sigma, "cST" for conditional Sheth-Tormen, "QEPS" for EPS with Lagrangian radius, "QcST" for cST with Lagrangian radius

    Outputs :
    res : Fraction of collapsed mass in the region (unitless)
    """
    res = Mcoll(Mmin, Mregion, kh, pk, R0, delta_L, delta_NL, model) / Mregion
    if res < 0:
        print(f"Warning! fcoll < 0 for Mcoll = {res:.2e} Msun/h")
        res = 0
    return res

def fcoll_EPS(Mmin, kh, pk, R0, delta_L, delta_NL):
    """
    Computes the fraction of collapsed mass in a spherical region of radius R0 
    using the EPS formalism since an analytical formula exists
    
    Inputs :
    Mmin : Minimum mass to integrate from in Msun/h
    kh : Array of k/h values in h/Mpc
    pk : Array of power spectrum values in (Mpc/h)^3
    R0 : Size of the spherical region in Mpc/h
    delta_L : Linear overdensity (unitless)
    delta_NL : Non-linear overdensity (unitless)

    Outputs :
    res : Fraction of collapsed mass in the region (unitless)
    """    

    Rmin=M_to_R(Mmin)
    
    R0 = R0 * (1+delta_NL)**(1/3)
    
    num = delta_c-delta_L
    denom = np.sqrt(2*(compute_sig_camb(Rmin, kh, pk)**2 - compute_sig_camb(R0, kh, pk)**2))
    res = erfc(num/denom)
    return res


def compute_delta_linear(delta_NL, delta_c = delta_c):
    """
    Computes the linear overdensity from the non-linear overdensity using the fitting formula from MBW (1996)
    
    Inputs :
    delta_NL : Non-linear overdensity (unitless)
    delta_c : Critical density for collapse (default is 1.68647)
    
    Outputs :
    res : Linear overdensity (unitless)
    """
    res = delta_c / 1.68647  * (1.68647 - 1.35/(1+delta_NL)**(2/3) - 1.12431/(1+delta_NL)**(1/2) + 0.78785/(1+delta_NL)**0.58661)
    return res

def compute_delta_NL(delta_L, delta_c = delta_c):
    """
    Computes the non-linear overdensity from the linear overdensity
    
    Inputs :
    delta_L : Linear overdensity (unitless)
    delta_c : Critical density for collapse (default is 1.68647)
    
    Outputs :
    res : Non-linear overdensity (unitless)
    """
    def func(delta_NL):
        return compute_delta_linear(delta_NL, delta_c = delta_c) - delta_L
    
    sol = root_scalar(func, bracket=[-0.99, 200], method='brentq') 
    if sol.converged:
        return sol.root
    else:
        return None
    
def norm2dist(norm_val, grid_size):
    '''
    Computes the distance in Mpc from a normalized value between 0 and 1
    
    Inputs :
    norm_val : Normalized value between 0 and 1
    grid_size : Size of the grid in Mpc in each direction (should be the same
    in all directions)
    
    Outputs : 
    dist : Distance in Mpc corresponding to the normalized value
    '''
    for i in range(len(grid_size)-1):
        if grid_size[i]!=grid_size[i+1]:
            print("The grid is does not have the same size in all directions.")
            return -1
    dist = norm_val * grid_size[0]
    return dist

def dist2norm(dist, grid_size):
    '''
    Computes the distance in Mpc from a normalized value between 0 and 1
    
    Inputs :
    dist : Distance in Mpc corresponding to the normalized value
    grid_size : Size of the grid in Mpc in each direction (should be the same
    in all directions)
    
    Outputs : 
    norm_val : Normalized value between 0 and 1
    '''
    for i in range(len(grid_size)-1):
        if grid_size[i]!=grid_size[i+1]:
            print("The grid is does not have the same size in all directions.")
            return -1
    norm_val = dist / grid_size[0]
    return norm_val

def idx_of_part_in_shape(center_sphere, size, grid_size, pos_mpc, shape="sphere"):
    """
    Computes the indices of particles in a spherical or cubic region centered on center_sphere with a given size.
    
    Inputs :
    center_sphere : Center of the sphere or cube in Mpc (x, y, z)
    size : Size of the sphere (radius) or cube (half-width) in Mpc
    grid_size : Size of the grid in Mpc in each direction (should be the same
    in all directions)
    pos_mpc : Positions of the particles in Mpc (N, 3) array
    shape : Shape of the region to consider, either "sphere" or "cube"
    
    Outputs : 
    w : Indices of the particles that are inside the sphere or cube
    """
    #size is radius (for sphere) or half-width (for cube)
    cx, cy, cz = center_sphere
    x, y, z = pos_mpc[:,0], pos_mpc[:,1], pos_mpc[:,2]
    xsize, ysize, zsize = grid_size
    
    dx = np.minimum(abs(x - cx), xsize - abs(x - cx)) #either normal distance or substracting the grid_size 
    dy = np.minimum(abs(y - cy), ysize - abs(y - cy))
    dz = np.minimum(abs(z - cz), zsize - abs(z - cz))
    
    if shape == "sphere":
        d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        w = np.where(d < size) 
    elif shape == "cube":
        inside_x = dx < size
        inside_y = dy < size
        inside_z = dz < size
        w = np.where(inside_x & inside_y & inside_z)  
    else :
        print("Choose 'sphere' or 'cube' in the shape arg.")
    return np.array(w[0])

def shift_to_sphere_center(sphere_center, grid_size, points):
    '''
    Shifts the points such that the sphere is centered at the center of the grid.
    
    Inputs :
    sphere_center : Center of the sphere in Mpc (x, y, z)
    grid_size : Size of the grid in Mpc in each direction (should be the same
    in all directions)
    points : List of points in Mpc to shift, should be a list of tuples [(x1,y1,z1), (x2,y2,z2), ...]
    
    Outputs : 
    shifted_points : List of points shifted such that the sphere is centered at the center of the grid
    '''
    nx, ny, nz = grid_size
    x, y, z = sphere_center
    
    xc, yc, zc = nx / 2, ny / 2, nz / 2 #domain center
    dx, dy, dz = xc - x, yc - y, zc - z
    
    shifted_points = np.array([( (X + dx) % nx, (Y + dy) % ny, (Z + dz) % nz ) for X, Y, Z in points])
    
    return shifted_points


def idx_halos(center_sphere, radius_sphere, grid_size, halo_pos_mpc, method="Manual", shape="sphere"):
    """
    Computes the indices of halos in a spherical region centered on center_sphere with a given radius.
    
    Inputs :
    center_sphere : Center of the sphere in Mpc (x, y, z)
    radius_sphere : Radius of the sphere in Mpc
    grid_size : Size of the grid in Mpc in each direction (should be the same
    in all directions)
    halo_pos_mpc : Positions of the halos in Mpc (N, 3)
    method : Method to use for finding the halos in the sphere, either "KDTree" or "Manual"
    shape : Shape of the region to consider, either "sphere" or "cube"
    
    Outputs :
    idx_of_halos_in_sphere : Indices of the halos that are inside the shape
    
    """
    
    if method == "KDTree":
        #computing the shifted postion such tha tthe sphere is at the center
        shifted_halos_to_sphere = shift_to_sphere_center(center_sphere, grid_size, halo_pos_mpc)
        
        #creating instances of KDTrees to use their methods
        tree_halo = KDTree(shifted_halos_to_sphere)
        
        nx,ny,nz = grid_size
        shifted_center_sphere = [nx / 2, ny / 2, nz / 2] #new center in the center of the box
        
        #finding the number of halos in the sphere using query_ball_point 
        idx_of_halos_in_sphere_kdtree = np.array(tree_halo.query_ball_point(shifted_center_sphere,radius_sphere))
        
        return idx_of_halos_in_sphere_kdtree
    elif method == "Manual":
        #idx of halos and particles in the sphere respectively
        w_halos = idx_of_part_in_shape(center_sphere, radius_sphere, grid_size, halo_pos_mpc, shape=shape)
        
        return w_halos
    else : 
        print(r"Method not recognized ! Choose between 'KDTree' and 'Manual'.")
        return -1
    
def sphere_density(Rsphere, nb_part, mpart):
    '''
    Computes the density in a spherical region of radius Rsphere containing nb_part particles of mass mpart.
    
    Inputs :
    Rsphere : Radius of the sphere in Mpc
    nb_part : Number of particles in the sphere
    mpart : Mass of each particle in kg
    
    Outputs :
    rho_sphere : Density in the sphere in Msun/Mpc^3
    '''
    Msphere = mpart / 2e30 * nb_part #in Msun
    Vsphere = 4/3*np.pi*Rsphere**3 #in Mpc**3
    rho_sphere = Msphere / Vsphere #Msun/Mpc**3
    return rho_sphere

def density_in_shape(center_sphere, size_shape, grid_size, part_pos_mpc, mpart, shape="sphere"):
    """
    Computes the density in a spherical or cubic region centered on center_sphere with a given size.
    
    Inputs :
    center_sphere : Center of the sphere or cube in Mpc (x, y,z)
    size_shape : Size of the sphere (radius) or cube (half-width) in Mpc
    grid_size : Size of the grid in Mpc in each direction (should be the same
    in all directions)
    part_pos_mpc : Positions of the particles in Mpc (N, 3)
    mpart : Mass of each particle in kg
    shape : Shape of the region to consider, either "sphere" or "cube"
    
    Outputs :
    rho_shape : Density in the sphere or cube in Msun/Mpc^3
    """
    
    w_part = idx_of_part_in_shape(center_sphere, size_shape, grid_size, part_pos_mpc, shape=shape)
    nb_part_in_shape = len(w_part)
    if shape == "sphere":
        rho_shape = sphere_density(size_shape, nb_part_in_shape, mpart)
    elif shape == "cube":
        rho_shape = mpart/2e30 * nb_part_in_shape / (2*size_shape)**3
    else :
        print("Choose 'sphere' or 'cube' in the shape arg.")
    return rho_shape

def paving_domain(N, grid_size):
    """
    Segments the domain into N^3 subregions, each with a spherical shape.
    
    Inputs :
    N : Number of cut in each direction (there will be N^3 subregions)
    grid_size : Size of the grid in Mpc in each direction (should be the same
    in all directions)
    
    Outputs :
    centers : Array of shape (N^3, 3) containing the centers of the subregions in Mpc
    radius : Radius of the spheres in Mpc
    """
    nx, ny, nz = grid_size
    dx, dy, dz = nx / N, ny / N, nz / N  # diameter  in each coordinate
    radius = min(dx, dy, dz) / 2  # min to not have spheres that overlap

    i, j, k = np.meshgrid(np.arange(0,N,1), np.arange(0,N,1), np.arange(0,N,1), indexing='ij')

    centers = np.stack(((i+0.5) * dx, (j+0.5) * dy, (k+0.5) * dz), axis=-1) #+0.5 at each i j k?

    centers = np.reshape(centers, (-1,3))
    
    return centers, radius

class subregion:
    """
    This class allows to easily manipulate the subregions of a domain.
    """
    def __init__(self, center, radius, shape="sphere"):
        self.center = np.array(center) 
        self.radius = radius #radius for sphere / half_width for cube in Mpc
        self.shape = shape #sphere or cube
        self.density = None #Msun/Mpc^3
        self.delta_NL = None
        self.delta_L = None # = delta0
        self.part_indices = []
        self.halo_indices = []
        self.num_halos = 0
        self.cmf_emp = None
        self.err_emp = None
        self.bcen_emp = None
        self.cmf_th_cST = None
        self.cmf_th_EPS = None
        self.bins_emp = None
    
    def get_idx_particles_inside(self, part_pos_mpc):
        """
        Computes the indices of particles inside the subregion defined by its center and radius.
        
        Inputs :
        part_pos_mpc : Positions of the particles in Mpc (N, 3)
        
        Outputs :
        w : Indices of the particles that are inside the subregion
        """
        cx, cy, cz = self.center
        if self.shape == "cube":
            w = np.where((part_pos_mpc[:, 0] >= cx - self.radius) & (part_pos_mpc[:, 0] < cx + self.radius) &
                         (part_pos_mpc[:, 1] >= cy - self.radius) & (part_pos_mpc[:, 1] < cy + self.radius) &
                         (part_pos_mpc[:, 2] >= cz - self.radius) & (part_pos_mpc[:, 2] < cz + self.radius))[0]
        elif self.shape=="sphere": 
            distances = np.sqrt((part_pos_mpc[:, 0] - cx) ** 2 +(part_pos_mpc[:, 1] - cy) ** 2 +(part_pos_mpc[:, 2] - cz) ** 2)
            # distances = np.linalg.norm(part_pos_mpc - self.center, axis=1)
            w = np.where(distances <= self.radius)[0]
        self.part_indices = w
        return w
    
    def compute_density(self, part_pos_mpc, mpart, rho_mean):
        """
        Computes the density in a spherical or cubic region centered on center_sphere with a given size.
        
        Inputs :
        part_pos_mpc : Positions of the particles in Mpc (N, 3)
        mpart : Mass of each particle in kg
        rho_mean : Average density of the universe in Msun/Mpc^3
        
        Outputs :
        num_particles : Number of particles inside the subregion
        w : Indices of the particles that are inside the subregion
        """
        w = self.get_idx_particles_inside(part_pos_mpc)
        num_particles = len(w)
        
        if self.shape == "sphere":
            self.density = sphere_density(self.radius, num_particles, mpart)
        elif self.shape == "cube" : 
            self.density = (mpart / 2e30) * num_particles / (2 * self.radius) ** 3
        
        self.delta_NL = (self.density - rho_mean) / rho_mean
        self.delta_L = compute_delta_linear(self.delta_NL)
        
        return num_particles, w
    
    def compute_halo_stat(self, halo_pos_mpc, mh, kh, pk, M_list, grid_size, model = "both", nbins=20):
        """
        Computes the halo statistics in the subregion defined by its center and radius.
        
        Inputs :
        halo_pos_mpc : Positions of the halos in Mpc (N, 3)
        mh : Masses of the halos in Msun (N,)
        kh : Array of k/h values in h/Mpc
        pk : Array of power spectrum values in (Mpc/h)^3
        M_list : List of masses in Msun/h for which to compute the CMF
        grid_size : Size of the grid in Mpc in each direction (should be the same
        in all directions)
        model : Model to use for the CMF calculation
        
        Outputs :
        self.halo_indices : Indices of the halos that are inside the subregion
        self.num_halos : Number of halos inside the subregion
        self.bins_emp : Bins for the empirical CMF
        self.bcen_emp : Centers of the bins for the empirical CMF
        self.cmf_emp : Empirical CMF values
        self.err_emp : Errors on the empirical CMF values
        self.cmf_th_cST : Theoretical CMF values for the cST model
        self.cmf_th_EPS : Theoretical CMF values for the EPS model
        """
        self.halo_indices = idx_halos(self.center, self.radius, grid_size, halo_pos_mpc, method="Manual", shape=self.shape)
        self.num_halos = len(self.halo_indices)
        
        if self.num_halos > 0:
            mh_in_subreg = mh[self.halo_indices]
            lightest_halo = np.min(mh_in_subreg)
            heaviest_halo = np.max(mh_in_subreg)
            
            # self.bins_emp = np.logspace(np.log10(lightest_halo), np.log10(heaviest_halo), num=int(np.sqrt(self.num_halos)))
            # self.bins_emp = np.logspace(np.log10(lightest_halo), np.log10(heaviest_halo), num=50)
            self.bins_emp = np.logspace(11, 16, num=nbins)
            
            self.bcen_emp = 0.5 * (self.bins_emp[1:] + self.bins_emp[:-1])
            db = self.bins_emp[1:] - self.bins_emp[:-1]
            cmf, _ = np.histogram(mh_in_subreg * h, bins=self.bins_emp)
            if self.shape == "sphere":
                Vshape = 4/3 * np.pi * (self.radius * h * (1+self.delta_NL)**(1/3)) ** 3 # (Mpc/h)^3
            else :
                Vshape = (self.radius * h * (1+self.delta_NL)**(1/3)) ** 3 # (Mpc/h)^3
            
            self.cmf_emp = cmf / db / Vshape
            
            err_upper = self.cmf_emp + 3 * np.sqrt(cmf) / db / Vshape
            err_lower = self.cmf_emp - 3 * np.sqrt(cmf) / db / Vshape
            self.err_emp = [err_upper, err_lower]
        else:
            self.bins_emp = None
            self.bcen_emp = None
            self.cmf_emp = None
            self.err_emp = None
        
        if model == "QcST" :
            self.cmf_th_cST = [cmfcalc(M_list[i], kh, pk, self.radius * h, self.delta_L, self.delta_NL, "QcST") for i in range(len(M_list))]   
        elif model == "cST" :
            self.cmf_th_cST = [cmfcalc(M_list[i], kh, pk, self.radius * h, self.delta_L, self.delta_NL, "cST") for i in range(len(M_list))]
        elif model == "QEPS_scaled" :
            self.cmf_th_EPS = [cmfcalc(M_list[i], kh, pk, self.radius * h, self.delta_L, self.delta_NL, "QEPS_scaled") for i in range(len(M_list))]
        elif model == "EPS" :
            self.cmf_th_EPS = [cmfcalc(M_list[i], kh, pk, self.radius * h, self.delta_L, self.delta_NL, "EPS") for i in range(len(M_list))]     
        elif model == "QEPS" :
            self.cmf_th_EPS = [cmfcalc(M_list[i], kh, pk, self.radius * h, self.delta_L, self.delta_NL, "QEPS") for i in range(len(M_list))]
        elif model == "EPS_scaled":
            self.cmf_th_EPS = [cmfcalc(M_list[i], kh, pk, self.radius * h, self.delta_L, self.delta_NL, "QEPS_scaled") for i in range(len(M_list))]
        else : 
            # self.cmf_th_cST = [mycmf.cmfcalc(M_list[i], kh, pk, self.radius * h, self.delta_L, "cST") for i in range(len(M_list))]
            # self.cmf_th_EPS = [mycmf.cmfcalc(M_list[i], kh, pk, self.radius * h, self.delta_L, "EPS_scaled") for i in range(len(M_list))]
            self.cmf_th_cST = [cmfcalc(M_list[i], kh, pk, self.radius * h, self.delta_L, self.delta_NL, "QcST") for i in range(len(M_list))]
            self.cmf_th_EPS = [cmfcalc(M_list[i], kh, pk, self.radius * h, self.delta_L, self.delta_NL, "QEPS") for i in range(len(M_list))]


def analyze_subregions(Ncut, shape, grid_size, halo_pos_mpc, part_pos_mpc, mpart, rho_mean, mh, kh, pk, M_list, model = "cST", nbins=20):
    """
    Analyzes the subregions of a domain by computing the density and halo statistics in each subregion.
    
    Inputs : 
    Ncut : Number of cuts in each direction (there will be Ncut^3 subregions)
    shape : Shape of the subregions, either "sphere" or "cube"
    grid_size : Size of the grid in Mpc in each direction (should be the same
    in all directions)
    halo_pos_mpc : Positions of the halos in Mpc (N, 3)
    part_pos_mpc : Positions of the particles in Mpc (N, 3)
    mpart : Mass of each particle in kg
    rho_mean : Average density of the universe in Msun/Mpc^3
    mh : Masses of the halos in Msun (size N)
    kh : Array of k/h values in h/Mpc
    pk : Array of power spectrum values in (Mpc/h)^3
    M_list : List of masses in Msun/h for which to compute the CMF
    model : Model to use for the CMF calculation, default is "cST"
    nbins : Number of bins to use for the empirical CMF calculation, default is 20
    
    Outputs : 
    subregions : List of subregion objects containing the computed statistics for each subregion
    """
    
    paved_centers, paved_radius = paving_domain(Ncut, grid_size)
    subregions = [subregion(center, paved_radius, shape = shape) for center in paved_centers]
    
    print(f"Computing values for {len(subregions)} spheres of radius {paved_radius:.2f} Mpc...")
    current_nb_part_tot = 0
    current_nb_halos = 0
    for current_subregion in tqdm(subregions):
        num_particles, w = current_subregion.compute_density(part_pos_mpc, mpart, rho_mean)
        part_pos_mpc = np.delete(part_pos_mpc, w, axis=0)  # remove found elements to speed up next iterations
        current_nb_part_tot += num_particles
        current_subregion.compute_halo_stat(halo_pos_mpc, mh, kh, pk, M_list, grid_size, model = model, nbins=nbins)
        current_nb_halos += current_subregion.num_halos
        
    
    print(f"Found {current_nb_part_tot} particles and {current_nb_halos} halos in {Ncut}^3 = {len(subregions)} {shape}s of radius/half-width {paved_radius:.2f} Mpc.")
    return np.array(subregions)

def plot_sub_cmf(subreg_sorted,  arg, delta_NL_paved, Ncut, mh, nbins, M_list, shape = "sphere", method="range"):
    '''
    Plots the average CMF and the range of overdensities for each subregion.
    
    Inputs :
    subreg_sorted : Sorted list of subregion objects
    arg : Argument for the method, either a range of overdensities or a count of
    subregions to consider
    delta_NL_paved : Array of non-linear overdensities for each subregion
    Ncut : Number of cuts in each direction (there will be Ncut^3 subregions)
    mh : Masses of the halos in Msun (size N)
    nbins : Number of bins to use for the empirical CMF calculation
    M_list : List of masses in Msun/h for which to compute the CMF
    shape : Shape of the subregions, either "sphere" or "cube"
    method : Method to use for the analysis, either "range" for a range of overd
    '''
    cmap = cm.viridis
    norm = plt.Normalize(vmin=-1, vmax=2.5)
    color = cmap(norm(delta_NL_paved))
    fig, ax = plt.subplots(2, 1, figsize=(13, 10))
    
    if method == "range":
        dNL_range = arg
        lower_bound = dNL_range[0]
        upper_bound = dNL_range[-1]
        indiv_range = dNL_range[1] - dNL_range[0]
        
        nb_iter = len(dNL_range) - 1
        ax[0].set_title(f"Avg of CMF every for different ranges of $\delta_{{NL}}\in [{arg[0]},{arg[-1]}]$")
        ax[1].set_title(f"Range in overdensities in $\delta_{{NL}}\in [{arg[0]},{arg[-1]}]$")
    elif method == "number":
        count_cut = arg
        nb_iter = len(subreg_sorted) // count_cut + 1
        # if count_cut == (Ncut**3):
        #     ax[0].plot(bcen_hmf,mydndm_hmf,'-o',markersize=5, color = "k", label=f"HMF",alpha=0.3)
        ax[0].set_title(f"Avg of CMF every {count_cut} out of {Ncut**3} {shape}s")
        ax[1].set_title(f"Range in overdensities, every {count_cut} out of {Ncut**3} {shape}s")


    print("nb_iter", nb_iter)

    # first plot
    for i in range(nb_iter):
        if method == "range":
            # if i != nb_iter-1 :
            #     w = np.where((delta_NL_paved >= dNL_range[i]) & (delta_NL_paved < dNL_range[i+1]))[0]
            #     sr_of_interest = subreg_sorted[w]
            #     color_idx = int((w[0]+w[-1])/2)
            # else :
            #     w = np.where(delta_NL_paved >= dNL_range[i])[0]
            #     sr_of_interest = subreg_sorted[w]
            #     color_idx = int((w[0]+len(subreg_sorted))/2)
            w = np.where((delta_NL_paved >= dNL_range[i]) & (delta_NL_paved < dNL_range[i+1]))[0]
            sr_of_interest = subreg_sorted[w]
            if len(w)>0:
                color_idx = int((w[0]+w[-1])/2)
            else : 
                color_idx = 0
            
        elif method == "number":
            if i != nb_iter-1 :
                sr_of_interest = subreg_sorted[count_cut*i:count_cut*(i+1)]
                color_idx = int((count_cut*i+count_cut*(i+1))/2)
            else : 
                sr_of_interest = subreg_sorted[count_cut*i:]
                color_idx = int((count_cut*i+len(subreg_sorted))/2)
        
        halos_found = np.sum([sr.num_halos for sr in sr_of_interest])
        nb_reg = len(sr_of_interest)
        if nb_reg != 0:
            radius_sr = sr_of_interest[0].radius
            # ax[0].axvline(R_to_M(radius_sr*h), alpha=0.3,  color="gray", linestyle='dashed')
            
            delta_NL_values = np.array([sr.delta_NL for sr in sr_of_interest if sr.cmf_th_cST is not None])
            min_index = np.argmin(delta_NL_values)
            max_index = np.argmax(delta_NL_values)
            mean_dNL = np.mean([sr.delta_NL for sr in sr_of_interest])
            mean_dL = np.mean([sr.delta_L for sr in sr_of_interest])
            
            ###### avg of cmf ######
            ###### theory
            ### cmf(avg(delta_L))
            # avg_cmf_from_dL = [mycmf.cmfcalc(M_list[ii], kh, pk, (sr_of_interest[0].radius*h)*nb_reg**(1/3), mean_dL, model="cST") for ii in range(len(M_list))]
            # ax[0].loglog(M_list, avg_cmf_from_dL, '-', color=color[color_idx], alpha=0.8)
            
            ### avg(cmf(delta_L))
            all_cmf_th = [sr.cmf_th_cST for sr in sr_of_interest if sr.cmf_th_cST is not None]
            avg_cmf_filtered = np.mean(all_cmf_th, axis=0)
            ax[0].loglog(M_list, avg_cmf_filtered, '--', color=color[color_idx], alpha=0.8) #, label = f"$\delta_L={mean_dL_directly:.2f}$"
            
            ###### empirical
            ## avg of emp cmf
            avg_cmf_emp = np.mean([sr.cmf_emp for sr in sr_of_interest if sr.cmf_emp is not None], axis=0)
            avg_bcen_emp = np.mean([sr.bcen_emp for sr in sr_of_interest if sr.bcen_emp is not None], axis=0)

            ## errors
            err_sr = np.array([sr.err_emp for sr in sr_of_interest if sr.err_emp is not None])
            
            upper_errors = err_sr[:, 0, :]
            lower_errors = err_sr[:, 1, :]
            
            mean_upper_error = np.sqrt(np.sum(upper_errors ** 2, axis=0)) / len(err_sr)
            mean_lower_error = np.sqrt(np.sum(lower_errors ** 2, axis=0)) / len(err_sr)
            
            if method == "range":
                ax[0].errorbar(avg_bcen_emp, avg_cmf_emp,
                                yerr=[mean_lower_error, mean_upper_error], 
                                fmt='o-', markersize=5, alpha=0.7, color=color[color_idx],
                                label=f"Nb_subreg={len(sr_of_interest)} |  {halos_found} halos | $\delta_{{NL}} \in [{lower_bound+i*indiv_range},{lower_bound+(i+1)*indiv_range}]$ | $\left\langle \delta_{{NL}} \\right\\rangle={mean_dNL:.2f}$")
            elif method == "number":
                ax[0].errorbar(avg_bcen_emp, avg_cmf_emp,
                                yerr=[mean_lower_error, mean_upper_error], 
                                fmt='o-', markersize=5, alpha=0.7, color=color[color_idx],
                                label=f"Nb_subreg={len(sr_of_interest)} |  {halos_found} halos | $\delta_{{NL}} \in [{delta_NL_values[min_index]:.2f}, {delta_NL_values[max_index]:.2f}]$ | $\left\langle \delta_{{NL}} \\right\\rangle={mean_dNL:.2f}$")
            
            ### avg with halos
            halos_idx_list = np.array([sr.halo_indices for sr in sr_of_interest if len(sr.halo_indices) > 0])
            halos_idx = np.concatenate(halos_idx_list)
            mh_in_all_sr = mh[halos_idx]
            current_bins=np.logspace(11, 16,num=nbins)

            current_bcen=0.5*(current_bins[1:]+current_bins[:-1])
            current_db=current_bins[1:]-current_bins[:-1]
            current_cmf, current_bin_edges = np.histogram(mh_in_all_sr*h,bins=current_bins) 
            
            if shape == "sphere":
                V_all_sr = nb_reg * 4/3*np.pi*(sr_of_interest[0].radius*h)**3
            elif shape == "cube":
                V_all_sr = nb_reg * (sr_of_interest[0].radius*h)**3
            
            current_dndm=current_cmf/current_db/V_all_sr
            
            #error bars
            current_dndm_upper_err=current_dndm+3*np.sqrt(current_cmf)/current_db/V_all_sr #3 sigma dispersion
            current_dndm_lower_err=current_dndm-3*np.sqrt(current_cmf)/current_db/V_all_sr #3 sigma dispersion

    ax[0].loglog([],[], '--', color="k", label=f"Avg($CMF_{{th}}(\delta_{{NL}})$)")
    ax[0].scatter([], [], marker='o', color="k", label=f"Avg($CMF_{{emp}}$)")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax[0])
    cbar.set_label(f'Overdensity $\delta_{{NL}}$')

    ax[0].set_xlabel(r"Mass $[h^{-1}M_\odot]$")
    ax[0].set_ylabel(r"dn/dM $[h^{4}{\rm Mpc}^{-3}M_\odot^{-1}]$")
    ax[0].set_xlim([2e12,5e14])
    ax[0].set_ylim([1e-20,1e-15])
    ax[0].legend(bbox_to_anchor=(1.2, 1), loc='upper left')
    # ax[0].legend()

    # second plot
    bin_edges = np.arange(-1, 3, 0.2)
    count, bins, _ = ax[1].hist(delta_NL_paved, bins=bin_edges, density=True, alpha=0.6, color='dimgrey')

    ax[1].axvline(-1, color=color[0], linestyle='dashed')
    
    for i in range(nb_iter):
        if method == "range":
            w = np.where((delta_NL_paved >= dNL_range[i]) & (delta_NL_paved < dNL_range[i+1]))[0]
            if len(w)>0:
                color_idx = int((w[0]+w[-1])/2)
            else :
                color_idx = 0
            ax[1].axvline(lower_bound+(i+1)*indiv_range, color=color[color_idx], linestyle='dashed')
        elif method == "number":
            dNL_range_left = delta_NL_paved[count_cut*(i)-1]
            ax[1].axvline(dNL_range_left, color=color[count_cut*(i)-1], linestyle='dashed')
    ax[1].plot([],[], color='black', linestyle='dashed', alpha=0.7, label="Ranges of $\delta_{NL}$")

    ax[1].set_xlim(-1.1, 3)
    
    ax[1].set_xlabel(r"$\delta_{NL}$")
    ax[1].set_ylabel("Density probability")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
    
def plot_every_sub_cmf(subreg_sorted,  arg, delta_NL_paved, Ncut, mh, nbins, M_list, shape = "sphere", method="range", showfig=True, z=None, savename=None):
    '''
    Plots the average CMF and the range of overdensities for each subregion in multiple subplots
    
    Inputs :
    subreg_sorted : Sorted list of subregion objects
    arg : Argument for the method, either a range of overdensities or a count of
    subregions to consider
    delta_NL_paved : Array of non-linear overdensities for each subregion
    Ncut : Number of cuts in each direction (there will be Ncut^3 subregions)
    mh : Masses of the halos in Msun (size N)
    nbins : Number of bins to use for the empirical CMF calculation
    M_list : List of masses in Msun/h for which to compute the CMF
    shape : Shape of the subregions, either "sphere" or "cube"
    method : Method to use for the analysis, either "range" for a range of overd
    '''
    
    if method == "range":
        dNL_range = arg
        nb_iter = len(dNL_range) - 1
        lower_bound = dNL_range[0]
        upper_bound = dNL_range[-1]
        indiv_range = dNL_range[1] - dNL_range[0]
    elif method == "number":
        count_cut = arg
        nb_iter = (len(subreg_sorted) // count_cut + 1 )-1
        
    cmap = cm.viridis
    norm = plt.Normalize(vmin=-1, vmax=2.5)
    color = cmap(norm(delta_NL_paved))
    fig, ax = plt.subplots(nb_iter, 1, figsize=(8, 5 * nb_iter))  
        
    
    if nb_iter == 1:
        ax = [ax]

    for i in range(nb_iter):
        if method == "range":
            # if i != nb_iter-1 :
            #     w = np.where((delta_NL_paved >= dNL_range[i]) & (delta_NL_paved < dNL_range[i+1]))[0]
            #     sr_of_interest = subreg_sorted[w]
            #     color_idx = int((w[0]+w[-1])/2)
            # else :
            #     w = np.where(delta_NL_paved >= dNL_range[i])[0]
            #     sr_of_interest = subreg_sorted[w]
            #     color_idx = int((w[0]+len(subreg_sorted))/2)
            w = np.where((delta_NL_paved >= dNL_range[i]) & (delta_NL_paved < dNL_range[i+1]))[0]
            sr_of_interest = subreg_sorted[w]
            if len(w)>0:
                color_idx = int((w[0]+w[-1])/2)
            else :
                color_idx = 0
        elif method == "number":
            if i != nb_iter-1 :
                sr_of_interest = subreg_sorted[count_cut*i:count_cut*(i+1)]
                color_idx = int((count_cut*i+count_cut*(i+1))/2)

            else : 
                sr_of_interest = subreg_sorted[count_cut*i:]
                color_idx = int((count_cut*i+len(subreg_sorted))/2)
                
        halos_found = np.sum([sr.num_halos for sr in sr_of_interest])
        nb_reg = len(sr_of_interest)
        
        if len(sr_of_interest) != 0:
            delta_NL_values = np.array([sr.delta_NL for sr in sr_of_interest if sr.cmf_th_cST is not None])
            cmf_values = np.array([sr.cmf_th_cST for sr in sr_of_interest if sr.cmf_th_cST is not None])
            halos_idx_list = np.array([sr.halo_indices for sr in sr_of_interest if len(sr.halo_indices) > 0])
            if len(halos_idx_list) > 0:
                radius_sr = sr_of_interest[0].radius
                # ax[i].axvline(R_to_M(radius_sr*h), color="k", linestyle='dashed', alpha=0.3)
                
                dNL_sr_of_interest = np.array([sr.delta_NL for sr in sr_of_interest])
                mean_dNL = np.mean(dNL_sr_of_interest)
                mean_dL = np.mean([sr.delta_L for sr in sr_of_interest])

                if method == "range":
                    title_text = f"$\delta_{{NL}} \in [{lower_bound+i*indiv_range},{lower_bound+(i+1)*indiv_range}]$ | {halos_found} halos in {len(sr_of_interest)} regions"
                elif method == "number":
                    title_text = f"$\delta_{{NL}} \in [{delta_NL_values[min_index]:.2f}, {delta_NL_values[max_index]:.2f}]$ | {halos_found} halos in {len(sr_of_interest)} regions"
                ax[i].plot([], [], ' ', label=title_text)
                
                # avg_cmf_from_dL = [mycmf.cmfcalc(M_list[ii], kh, pk, (sr_of_interest[0].radius*h)*nb_reg**(1/3), mean_dL, model="cST") for ii in range(len(M_list))]
                # ax[i].loglog(M_list, avg_cmf_from_dL, '-', color=color[color_idx], alpha=0.8)
                all_cmf_th = [sr.cmf_th_cST for sr in sr_of_interest if sr.cmf_th_cST is not None]
                all_cmf_emp = [sr.cmf_emp for sr in sr_of_interest if sr.cmf_emp is not None]
                all_bcen_emp = [sr.bcen_emp for sr in sr_of_interest if sr.bcen_emp is not None]
                
                for idx,cmf_th in enumerate(all_cmf_th):
                    if method=="range":
                        ax[i].loglog(M_list, cmf_th, '-', color=color[w[0]+idx], alpha=0.065)
                    elif method=="number":
                        ax[i].loglog(M_list, cmf_th, '-', color=color[count_cut*i+idx], alpha=0.065)
                
                # for idx, cmf_emp in enumerate(all_cmf_emp):
                #     if method == "range":
                #         ax[i].loglog(all_bcen_emp[idx], cmf_emp, '-o', markersize=5, alpha=0.5, color=color[w[0]+idx])
                #     else :
                #         ax[i].loglog(all_bcen_emp[idx], cmf_emp, '-o', markersize=5, alpha=0.5, color=color[count_cut*i+idx])

                avg_cmf_filtered = np.mean(all_cmf_th, axis=0)
                ax[i].loglog(M_list, avg_cmf_filtered, '--', color="k", alpha=0.8) #, label = f"$\delta_L={mean_dL_directly:.2f}$"
                
                # from simulation
                avg_cmf_emp = np.average([sr.cmf_emp for sr in sr_of_interest if sr.cmf_emp is not None], axis=0)
                avg_bcen_emp = np.average([sr.bcen_emp for sr in sr_of_interest if sr.bcen_emp is not None], axis=0)

                med_cmf_emp = np.median([sr.cmf_emp for sr in sr_of_interest if sr.cmf_emp is not None], axis=0)
                med_bcen_emp = np.median([sr.bcen_emp for sr in sr_of_interest if sr.bcen_emp is not None], axis=0)
                ##errors
                err_sr = np.array([sr.err_emp for sr in sr_of_interest if sr.err_emp is not None])
                
                upper_errors = err_sr[:, 0, :]
                lower_errors = err_sr[:, 1, :]
            
                mean_upper_error = np.sqrt(np.sum(upper_errors ** 2, axis=0)) / len(err_sr)
                mean_lower_error = np.sqrt(np.sum(lower_errors ** 2, axis=0)) / len(err_sr)
                
                if i == 0:
                    ax[i].errorbar(avg_bcen_emp, avg_cmf_emp,
                            yerr=[mean_lower_error, mean_upper_error], 
                            fmt='o-', markersize=5, alpha=0.7, color="k",#color=color[color_idx],
                            label=f"Avg($CMF_{{emp}}$)")
                    ax[i].loglog([],[], '--', color="k", label=f"Avg($CMF_{{th}}(\delta_{{NL}})$)")
                else :
                    ax[i].errorbar(avg_bcen_emp, avg_cmf_emp,
                            yerr=[mean_lower_error, mean_upper_error], 
                            fmt='o-', markersize=5, alpha=0.7, color="k")
                # ax[i].scatter(med_bcen_emp, med_cmf_emp, s=20, color="red", alpha=0.9, label=f"Median($CMF_{{emp}}$)")
                # ax[i].plot(med_bcen_emp, med_cmf_emp, '-o', markersize=5, color="red", alpha=0.5, label=f"Median($CMF_{{emp}}$)")
                
                ### avg with halos
                
                halos_idx = np.concatenate(halos_idx_list)
                mh_in_all_sr = mh[halos_idx]
                current_bins=np.logspace(11, 16,num=nbins)

                current_bcen=0.5*(current_bins[1:]+current_bins[:-1])
                current_db=current_bins[1:]-current_bins[:-1]
                current_cmf,current_bin_edges = np.histogram(mh_in_all_sr*h,bins=current_bins) 
                
                if shape == "sphere":
                    V_all_sr = nb_reg * 4/3*np.pi*(sr_of_interest[0].radius*h)**3
                elif shape == "cube":
                    V_all_sr = nb_reg * (sr_of_interest[0].radius*h)**3
                    
                current_dndm=current_cmf/current_db/V_all_sr
                
                #error bars
                current_dndm_upper_err=current_dndm+3*np.sqrt(current_cmf)/current_db/V_all_sr #3 sigma dispersion
                current_dndm_lower_err=current_dndm-3*np.sqrt(current_cmf)/current_db/V_all_sr #3 sigma dispersion
                dndm_err = [current_dndm_upper_err, current_dndm_lower_err]
                
                # ax[i].errorbar(current_bcen, current_dndm,
                #                 yerr=[current_dndm - dndm_err[1], dndm_err[0] - current_dndm], 
                #                 fmt='*-', markersize=7, alpha=0.7, color=color[color_idx],
                #                 label=f"Sum halos to $CMF_{{emp}}$ | $\delta_{{NL}}={mean_dNL:.2f}$ | {halos_found} halos")
                
                # ## min and max
                max_index = np.argmax(delta_NL_values)
                min_index = np.argmin(delta_NL_values)
                
                max_cmf = sr_of_interest[max_index].cmf_th_cST
                min_cmf = sr_of_interest[min_index].cmf_th_cST
                
                ax[i].loglog(M_list, min_cmf, '-', color=color[int(color_idx/2)], alpha=0.3, label=f"Min $\delta_{{NL}}$={delta_NL_values[min_index]:.2f}")
                ax[i].loglog(M_list, max_cmf, '-', color=color[int((color_idx+len(color))/2)], alpha=0.3, label=f"Max $\delta_{{NL}}$={delta_NL_values[max_index]:.2f}")

                if i == nb_iter - 1:
                    ax[i].set_xlabel(r"Mass $[h^{-1}M_\odot]$")
                else:
                    ax[i].set_xlabel("")
                    ax[i].set_xticklabels([])
                
                ax[i].set_ylabel(r"dn/dM $[h^{4}{\rm Mpc}^{-3}M_\odot^{-1}]$")
                
                ax[i].set_xlim([2e12,5e14])
                ax[i].set_yscale('log')
                ax[i].set_ylim([1e-20,1e-15])
                ticks = [1e-19, 1e-18, 1e-17, 1e-16, 1e-15]
                if i == nb_iter - 1:
                    ticks = [1e-20] + ticks
                    ax[i].set_yticks(ticks)
                else : 
                    ax[i].set_yticks(ticks)
                ax[i].legend(loc="upper right", alignment='left')
                ax[i].grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.3)


    plt.tight_layout()
    # fig.suptitle(f'CMF - z={z:.2f} ($\simeq$ {compute_age_Gyr(z):.2f} Gyr) - {Ncut}$^3$ {shape}s of radius {sr_of_interest[0].radius:.2f} Mpc', fontsize=16,  x=0.55)
    fig.subplots_adjust(top=0.97)
    fig.subplots_adjust(hspace=0)
    if savename is not None:
        plt.savefig(savename, dpi=300)
    if showfig == True:
        plt.show()

def compute_age_Gyr(z, H0=H0, om=om):
    """
    Computes the age of the Universe for a given redshift z
    
    Inputs :
    z : redshift
    H0 : Hubble constant in km/s/Mpc
    om : matter density parameter
    
    Outputs :
    age : age of the Universe in Gyr
    """
    H0_astropy = H0 * u.km / u.s / u.Mpc
    cosmo = FlatLambdaCDM(H0=H0_astropy, Om0=om)
    age = cosmo.age(z).to(u.Gyr).value
    return age

def plot_every_sub_cmf2(subreg_sorted,  arg, delta_NL_paved, Ncut, mh, nbins, M_list, shape = "sphere", method="range", showfig=True, z=None, savename=None):
    '''
    Plots the average CMF and the range of overdensities for each subregion in multiple subplots
    
    Inputs :
    subreg_sorted : Sorted list of subregion objects
    arg : Argument for the method, either a range of overdensities or a count of
    subregions to consider
    delta_NL_paved : Array of non-linear overdensities for each subregion
    Ncut : Number of cuts in each direction (there will be Ncut^3 subregions)
    mh : Masses of the halos in Msun (size N)
    nbins : Number of bins to use for the empirical CMF calculation
    M_list : List of masses in Msun/h for which to compute the CMF
    shape : Shape of the subregions, either "sphere" or "cube"
    method : Method to use for the analysis, either "range" for a range of overdensity
    '''

    if method == "range":
        dNL_range = arg
        nb_iter = len(dNL_range) - 1
        lower_bound = dNL_range[0]
        upper_bound = dNL_range[-1]
        indiv_range = dNL_range[1] - dNL_range[0]
    elif method == "number":
        count_cut = arg
        nb_iter = (len(subreg_sorted) // count_cut + 1 )-1
        
    cmap = cm.viridis
    norm = plt.Normalize(vmin=-1, vmax=2.5)
    color = cmap(norm(delta_NL_paved))
    
    nb_plots = nb_iter + 1  # +1 for the histogram plot
    ncols = 3
    nrows = (nb_plots + ncols - 1) // ncols  # pure numpy-compatible ceiling
    fig, ax = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows), )#sharex=True, sharey=True
    ax = np.array(ax).reshape(-1)
    
    for j in range(nb_iter+1, len(ax)):
        ax[j].set_visible(False)
    
    if nb_iter == 1:
        ax = [ax]

    for i in range(nb_iter):
        if method == "range":
            # if i != nb_iter-1 :
            #     w = np.where((delta_NL_paved >= dNL_range[i]) & (delta_NL_paved < dNL_range[i+1]))[0]
            #     sr_of_interest = subreg_sorted[w]
            #     color_idx = int((w[0]+w[-1])/2)
            # else :
            #     w = np.where(delta_NL_paved >= dNL_range[i])[0]
            #     sr_of_interest = subreg_sorted[w]
            #     color_idx = int((w[0]+len(subreg_sorted))/2)
            w = np.where((delta_NL_paved >= dNL_range[i]) & (delta_NL_paved < dNL_range[i+1]))[0]
            sr_of_interest = subreg_sorted[w]
            if len(w)>0:
                color_idx = int((w[0]+w[-1])/2)
            else :
                color_idx = 0
        elif method == "number":
            if i != nb_iter-1 :
                sr_of_interest = subreg_sorted[count_cut*i:count_cut*(i+1)]
                color_idx = int((count_cut*i+count_cut*(i+1))/2)

            else : 
                sr_of_interest = subreg_sorted[count_cut*i:]
                color_idx = int((count_cut*i+len(subreg_sorted))/2)
                
        halos_found = np.sum([sr.num_halos for sr in sr_of_interest])
        nb_reg = len(sr_of_interest)
        
        if len(sr_of_interest) != 0:
            delta_NL_values = np.array([sr.delta_NL for sr in sr_of_interest if sr.cmf_th_cST is not None])
            cmf_values = np.array([sr.cmf_th_cST for sr in sr_of_interest if sr.cmf_th_cST is not None])
            halos_idx_list = np.array([sr.halo_indices for sr in sr_of_interest if len(sr.halo_indices) > 0])
            if len(halos_idx_list) > 0:
                radius_sr = sr_of_interest[0].radius
                # ax[i].axvline(R_to_M(radius_sr*h), color="k", linestyle='dashed', alpha=0.3)
                
                dNL_sr_of_interest = np.array([sr.delta_NL for sr in sr_of_interest])
                mean_dNL = np.mean(dNL_sr_of_interest)
                mean_dL = np.mean([sr.delta_L for sr in sr_of_interest])

                if method == "range":
                    title_text = f"$\delta_{{NL}} \in [{lower_bound+i*indiv_range},{lower_bound+(i+1)*indiv_range}]$\n{halos_found} halos in {len(sr_of_interest)} regions"
                elif method == "number":
                    title_text = f"$\delta_{{NL}} \in [{delta_NL_values[min_index]:.2f}, {delta_NL_values[max_index]:.2f}]$\n{halos_found} halos in {len(sr_of_interest)} regions"
                ax[i].plot([], [], ' ', label=title_text)
                
                # avg_cmf_from_dL = [mycmf.cmfcalc(M_list[ii], kh, pk, (sr_of_interest[0].radius*h)*nb_reg**(1/3), mean_dL, model="cST") for ii in range(len(M_list))]
                # ax[i].loglog(M_list, avg_cmf_from_dL, '-', color=color[color_idx], alpha=0.8)
                all_cmf_th = [sr.cmf_th_cST for sr in sr_of_interest if sr.cmf_th_cST is not None]
                all_cmf_emp = [sr.cmf_emp for sr in sr_of_interest if sr.cmf_emp is not None]
                all_bcen_emp = [sr.bcen_emp for sr in sr_of_interest if sr.bcen_emp is not None]
                
                for idx,cmf_th in enumerate(all_cmf_th):
                    if method=="range":
                        ax[i].loglog(M_list, cmf_th, '-', color=color[w[0]+idx], alpha=0.065)
                    elif method=="number":
                        ax[i].loglog(M_list, cmf_th, '-', color=color[count_cut*i+idx], alpha=0.065)
                
                # for idx, cmf_emp in enumerate(all_cmf_emp):
                #     if method == "range":
                #         ax[i].loglog(all_bcen_emp[idx], cmf_emp, '-o', markersize=5, alpha=0.5, color=color[w[0]+idx])
                #     else :
                #         ax[i].loglog(all_bcen_emp[idx], cmf_emp, '-o', markersize=5, alpha=0.5, color=color[count_cut*i+idx])

                avg_cmf_filtered = np.mean(all_cmf_th, axis=0)
                ax[i].loglog(M_list, avg_cmf_filtered, '--', color="k", alpha=0.8) #, label = f"$\delta_L={mean_dL_directly:.2f}$"
                
                # from simulation
                avg_cmf_emp = np.average([sr.cmf_emp for sr in sr_of_interest if sr.cmf_emp is not None], axis=0)
                avg_bcen_emp = np.average([sr.bcen_emp for sr in sr_of_interest if sr.bcen_emp is not None], axis=0)

                med_cmf_emp = np.median([sr.cmf_emp for sr in sr_of_interest if sr.cmf_emp is not None], axis=0)
                med_bcen_emp = np.median([sr.bcen_emp for sr in sr_of_interest if sr.bcen_emp is not None], axis=0)
                ##errors
                err_sr = np.array([sr.err_emp for sr in sr_of_interest if sr.err_emp is not None])
                
                upper_errors = err_sr[:, 0, :]
                lower_errors = err_sr[:, 1, :]
            
                mean_upper_error = np.sqrt(np.sum(upper_errors ** 2, axis=0)) / len(err_sr)
                mean_lower_error = np.sqrt(np.sum(lower_errors ** 2, axis=0)) / len(err_sr)
                
                if i == 0:
                    ax[i].errorbar(avg_bcen_emp, avg_cmf_emp,
                            yerr=[mean_lower_error, mean_upper_error], 
                            fmt='o-', markersize=5, alpha=0.7, color="k",#color=color[color_idx],
                            # label=f"Avg($CMF_{{emp}}$)"
                            )
                    ax[i].loglog([],[], '--', color="k", 
                                #  label=f"Avg($CMF_{{th}}(\delta_{{NL}})$)"
                                 )
                else :
                    ax[i].errorbar(avg_bcen_emp, avg_cmf_emp,
                            yerr=[mean_lower_error, mean_upper_error], 
                            fmt='o-', markersize=5, alpha=0.7, color="k")
                # ax[i].scatter(med_bcen_emp, med_cmf_emp, s=20, color="red", alpha=0.9, label=f"Median($CMF_{{emp}}$)")
                # ax[i].plot(med_bcen_emp, med_cmf_emp, '-o', markersize=5, color="red", alpha=0.5, label=f"Median($CMF_{{emp}}$)")
                
                ### avg with halos
                
                halos_idx = np.concatenate(halos_idx_list)
                mh_in_all_sr = mh[halos_idx]
                current_bins=np.logspace(11, 16,num=nbins)

                current_bcen=0.5*(current_bins[1:]+current_bins[:-1])
                current_db=current_bins[1:]-current_bins[:-1]
                current_cmf,current_bin_edges = np.histogram(mh_in_all_sr*h,bins=current_bins) 
                
                if shape == "sphere":
                    V_all_sr = nb_reg * 4/3*np.pi*(sr_of_interest[0].radius*h)**3
                elif shape == "cube":
                    V_all_sr = nb_reg * (sr_of_interest[0].radius*h)**3
                    
                current_dndm=current_cmf/current_db/V_all_sr
                
                #error bars
                current_dndm_upper_err=current_dndm+3*np.sqrt(current_cmf)/current_db/V_all_sr #3 sigma dispersion
                current_dndm_lower_err=current_dndm-3*np.sqrt(current_cmf)/current_db/V_all_sr #3 sigma dispersion
                dndm_err = [current_dndm_upper_err, current_dndm_lower_err]
                
                # ax[i].errorbar(current_bcen, current_dndm,
                #                 yerr=[current_dndm - dndm_err[1], dndm_err[0] - current_dndm], 
                #                 fmt='*-', markersize=7, alpha=0.7, color=color[color_idx],
                #                 label=f"Sum halos to $CMF_{{emp}}$ | $\delta_{{NL}}={mean_dNL:.2f}$ | {halos_found} halos")
                
                # ## min and max
                max_index = np.argmax(delta_NL_values)
                min_index = np.argmin(delta_NL_values)
                
                max_cmf = sr_of_interest[max_index].cmf_th_cST
                min_cmf = sr_of_interest[min_index].cmf_th_cST
                
                ax[i].loglog(M_list, min_cmf, '-', color=color[int(color_idx/2)], alpha=0.3, label=f"Min $\delta_{{NL}}$={delta_NL_values[min_index]:.2f}")
                ax[i].loglog(M_list, max_cmf, '-', color=color[int((color_idx+len(color))/2)], alpha=0.3, label=f"Max $\delta_{{NL}}$={delta_NL_values[max_index]:.2f}")
                # ax[i].fill_between(M_list,min_cmf,max_cmf, color=color[color_idx], alpha=0.15)
                
                # ax[i].loglog([],[], '-', color=color[color_idx], label=f"$n_{{cST}}$(Avg($\delta_{{NL}}$))")
                
                # if z is None :
                #     if method=="range":
                #         ax[i].set_title(f"CMF $\delta_{{NL}} \in [{lower_bound+i*indiv_range},{lower_bound+(i+1)*indiv_range}]$ finds {len(sr_of_interest)} subreg with {halos_found} halos and $\left\langle \delta_{{NL}} \\right\\rangle={mean_dNL:.2f}$  ")
                #     elif method == "number":
                #         ax[i].set_title(f"CMF for {len(sr_of_interest)} subreg with {halos_found} halos and $\delta_{{NL}} \in [{delta_NL_values[min_index]:.2f}, {delta_NL_values[max_index]:.2f}]$ with $\left\langle \delta_{{NL}} \\right\\rangle={mean_dNL:.2f}$.")
                # else : 
                #     if method=="range":
                #         ax[i].set_title(f"$\delta_{{NL}} \in [{lower_bound+i*indiv_range},{lower_bound+(i+1)*indiv_range}]$ | $\left\langle \delta_{{NL}} \\right\\rangle={mean_dNL:.2f}$ | {len(sr_of_interest)} regions with {halos_found} halos")
                #     elif method == "number":
                #         ax[i].set_title(f"{len(sr_of_interest)} regions with {halos_found} halos and $\delta_{{NL}} \in [{delta_NL_values[min_index]:.2f}, {delta_NL_values[max_index]:.2f}]$ with $\left\langle \delta_{{NL}} \\right\\rangle={mean_dNL:.2f}$.")
                        
                # if i == nb_iter - 1:
                #     ax[i].set_xlabel(r"Mass $[h^{-1}M_\odot]$")
                # else:
                #     ax[i].set_xlabel(r"Mass $[h^{-1}M_\odot]$")
                #     # ax[i].set_xlabel("")
                #     ax[i].set_xticklabels([])
                ax[i].set_xlabel(r"Mass $[h^{-1}M_\odot]$")
                
                if i % ncols == 0:
                    ax[i].set_ylabel(r"dn/dM $[h^{4}{\rm Mpc}^{-3}M_\odot^{-1}]$")
                else:
                    ax[i].set_ylabel("")
                # ax[i].set_ylabel(r"dn/dM $[h^{4}{\rm Mpc}^{-3}M_\odot^{-1}]$")
                
                ax[i].set_xlim([2e12,5e14])
                ax[i].set_yscale('log')
                ax[i].set_ylim([1e-20,1e-15])
                ticks = [1e-19, 1e-18, 1e-17, 1e-16, 1e-15]
                # row_idx = i // ncols
                # if row_idx == nrows :
                #     ax[i].set_yticks(ticks)
                # else : 
                #     ax[i].set_yticks(ticks)
                last_row_start = ncols * (nrows - 1)
                if i >= last_row_start:
                    ticks = [1e-20] + ticks
                ax[i].set_yticks(ticks)
                ax[i].legend(loc="upper right", alignment='left')
                ax[i].grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.3)
                
                
                # leg = ax[i].legend(loc="upper right")  # or wherever your legend is
                # shift = max([t.get_window_extent().width for t in leg.get_texts()])
                # for t in leg.get_texts():
                #     t.set_ha('right') # ha is alias for horizontalalignment
                    # t.set_position((shift,0))
                ################## HISTOGRAM ################
                
    hist_ax = ax[nb_iter]
    bin_edges = np.arange(-1, 3, 0.2)
    count, bins, _ = hist_ax.hist(delta_NL_paved, bins=bin_edges, density=True, alpha=0.6, color='dimgrey')

    hist_ax.axvline(-1, color=color[0], linestyle='dashed')

    for i in range(nb_iter):
        if method == "range":
            w = np.where((delta_NL_paved >= dNL_range[i]) & (delta_NL_paved < dNL_range[i+1]))[0]
            if len(w) > 0:
                color_idx = int((w[0] + w[-1]) / 2)
            else:
                color_idx = 0
            hist_ax.axvline(lower_bound + (i + 1) * indiv_range, color=color[color_idx], linestyle='dashed')
        elif method == "number":
            if i > 0:
                dNL_range_left = delta_NL_paved[count_cut * i - 1]
                hist_ax.axvline(dNL_range_left, color=color[count_cut * i - 1], linestyle='dashed')
    hist_ax.plot([], [], color='black', linestyle='dashed', alpha=0.7, label="Ranges of $\delta_{NL}$")
    # hist_ax.set_xlim(-1.1, 3)
    hist_ax.set_xlabel(r"$\delta_{NL}$")
    hist_ax.set_ylabel("Density probability")
    hist_ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    # fig.suptitle(f'CMF - z={z:.2f} ($\simeq$ {compute_age_Gyr(z):.2f} Gyr) - {Ncut}$^3$ {shape}s of radius {sr_of_interest[0].radius:.2f} Mpc', fontsize=16, y=1.005,  x=0.55)
    fig.subplots_adjust(top=0.97)
    fig.subplots_adjust(hspace=0)
    if savename is not None:
        plt.savefig(savename, dpi=300, bbox_inches="tight")
    if showfig == True:
        plt.show()

def Mmin_finder(bcen_hmf, hmf_emp_err_up, hmf_emp_err_low, n_ST_over_bins):
    """
    Computes the minimal mass Mmin for which the empirical HMF is within the error bars of the theoretical HMF
    
    Inputs :
    bcen_hmf : array of bin centers for the HMF
    hmf_emp_err_up : upper error bar for the empirical HMF
    hmf_emp_err_low : lower error bar for the empirical HMF
    n_ST_over_bins : empirical HMF values over the bins
    
    Outputs :
    Mmin : minimal mass for which the empirical HMF is within the error bars of the theoretical HMF
    """
    w = np.where((n_ST_over_bins < hmf_emp_err_up) & (n_ST_over_bins > hmf_emp_err_low))[0]
    return bcen_hmf[w][0]


def compute_fcoll_emp(subreg_sorted, Mmin, mh, mpart):
    '''
    Compute the fraction of collapsed mass in the subregions emprically
    
    Inputs :
    subreg_sorted : list of SubRegion objects sorted by delta_NL
    Mmin : minimum mass to consider
    mh : array of halo masses
    mpart : mass of a particle in kg
    
    Outputs :
    fcoll_emp : array of fraction of collapsed mass in each subregion
    delta_NL_filter : array of delta_NL values for the subregions considered
    '''
    mass_collapsed = []
    mass_tot_in_sr = []
    delta_NL_filter = []
    for sr in subreg_sorted:
        if len(sr.halo_indices) > 0:
            mh_sr = mh[sr.halo_indices]
            mh_above_Mmin = mh_sr[mh_sr > Mmin/h]
            mcoll = np.sum(mh_above_Mmin)*h
            mtot = len((sr.part_indices).tolist())*mpart/2e30*h
            if mcoll > 0 and mtot > 0 and mtot > mcoll : 
                mass_collapsed.append(mcoll)
                mass_tot_in_sr.append(mtot)
                delta_NL_filter.append(sr.delta_NL)

    mass_collapsed = np.array(mass_collapsed)
    mass_tot_in_sr = np.array(mass_tot_in_sr)

    fcoll_emp = mass_collapsed / mass_tot_in_sr
    
    fcoll_emp = np.array(fcoll_emp)
    delta_NL_filter = np.array(delta_NL_filter)
    return fcoll_emp, delta_NL_filter


def compute_fcoll_th(subreg_sorted, Mmin, kh, pk, mh, mpart, model = "QcST"):
    '''
    Compute the fraction of collapsed mass in the subregions theoretically
    
    Inputs :
    subreg_sorted : list of SubRegion objects sorted by delta_NL
    Mmin : minimum mass to consider
    mh : Masses of the halos in Msun (size N)
    kh : Array of k/h values in h/Mpc
    pk : Array of power spectrum values in (Mpc/h)^3
    mpart : mass of a particle in kg
    
    Outputs :
    fcoll_emp : array of fraction of collapsed mass in each subregion
    delta_NL_filter : array of delta_NL values for the subregions considered
    '''
    fcoll_th = []
    delta_NL_filter = []
    radius_sr_h = subreg_sorted[0].radius * h
    for idx, sr in enumerate(subreg_sorted):
        if len(sr.halo_indices) > 0 :
            mh_sr = mh[sr.halo_indices]
            mh_above_Mmin = mh_sr[mh_sr > Mmin/h]
            mcoll = np.sum(mh_above_Mmin)*h
            if mcoll > 0 :
                dL = sr.delta_L
                dNL = sr.delta_NL
                Mreg = len((sr.part_indices).tolist())*mpart/2e30 * h
                fcoll_val = fcoll(Mmin, Mreg, kh, pk, radius_sr_h, dL, dNL, model)
                if fcoll_val > 0 and fcoll_val < 1 :
                    fcoll_th.append(fcoll_val)
                    delta_NL_filter.append(sr.delta_NL)
    
    fcoll_th = np.array(fcoll_th)
    delta_NL_filter = np.array(delta_NL_filter)
    
    return fcoll_th, delta_NL_filter


def halo_finder(nb_iter, save_dir="./saved_results/"):
    """
    Launches a halo finder on the particles data from a specific iteration.
    
    Inputs:
    nb_iter : iteration number to load the particles data from
    save_dir : directory to save the halo data
    
    Outputs:
    None but saves the halo data in the specified directory.
    """
    
    filename = f"cosmo_particles_particles_iter{nb_iter}.h5"
    fpart = h5py.File(f'datastageM2/{filename}', 'r')
    
    positions = np.array(fpart['coordinates'])

    x=positions[:,0]
    y=positions[:,1]
    z=positions[:,2]
    Npart=np.size(x)
    
    mpart=rho_0*(L/h*3.086e22)**3/Npart #kg #either rhoc * om or rho0
    
    masses=np.ones(Npart) # dummy particle masses array as a set of unit masses
    idx=np.arange(Npart) # monotonic indexes for particles

    data = dict(
        particle_position_x=x,
        particle_position_y=y,
        particle_position_z=z,
        particle_velocity_x=x,#dummy with no vel
        particle_velocity_y=y,#dummy with no vel
        particle_velocity_z=z,#dummy with no vel
        particle_mass=masses,
        particle_index=idx
    )
    ds = yt.load_particles(data,length_unit=L*3.086e24,periodicity=(True,True,True),mass_unit=mpart*1e3)
    
    hc = HaloCatalog(data_ds=ds, finder_method="hop",output_dir=save_dir)
    
    hc.create() # launch in background
    
    os.rename(f'{save_dir}/ParticleData/ParticleData.0.h5', f'{save_dir}/ParticleData/ParticleData{nb_iter}.h5')
    
    # fhalo = h5py.File('saved_results/ParticleData/ParticleData.0.h5', 'r') # upload Hop results