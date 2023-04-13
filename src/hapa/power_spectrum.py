import numpy as np

from . import cosmology as cosmo
from . import params
from .units import Quantity


def mass_vs_lagrangian_radius(radius: Quantity, volume_factor = 4*np.pi**2, cosmo = cosmo.PLANCK18):
    """
    # Mass vs Lagrangian radius
    """
    rho_average = cosmo.rho_matter_average_0
    return Quantity(rho_average.value * volume_factor * radius.value**3, rho_average.unit * radius.unit**3)


def lagrangian_radius_vs_mass(mass_MSUN: float, volume_factor = 4*np.pi**2, cosmo = cosmo.PLANCK18):
    """
    # Lagrangian radius vs mass in MSUN / Result in MPC
    """
    return (mass_MSUN/cosmo.rho_matter_average_0/volume_factor)**(1./3.)



#def growth_factor_D1_Carroll(z, cosmo): 
# Carroll+ 1992  // Mo and White p. 172
# corresponds to D1(a=1) with the definition of Dodelson 2003 -> To be checked

    # Abundances in a Universe with no radiation
    #double E2_h2_bis = _Omega_m_h2*pow(1+z,3) + _Omega_l_h2;
    #double Om = _Omega_m_h2*pow(1+z,3)/E2_h2_bis;
    #double Ol = _Omega_l_h2/E2_h2_bis;

    #double res = 2.5*Om/(pow(Om,4./7)-Ol+(1+0.5*Om)*(1+1./70.*Ol))/(1+z);
    #return res;


class Cosmology(cosmo.BackgroundCosmology):

    def __init__(self, curvature_ps, window_type):
        self._curvature_ps = curvature_ps
        self._window_type  = window_type

    def matter_power_spectrum(self, k, z):
        pass
        
        #return  (4. / 25.) * pow(D1_z * k * k * tf * c_over_H0_Mpc * c_over_H0_Mpc / Omega_m_0, 2) * curvature_power_spectrum(k)


    def _compute_sigmaR2(self, R):
        
        def __integrand(X):
            pass
    