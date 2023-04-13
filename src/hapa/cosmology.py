import numpy as np
from . import units
from .units import Quantity

class BackgroundCosmology:

    def __init__(self, omega_cdm_h2_0, omega_baryons_h2_0, hubble_parameter, T0_CMB_K = 2.72548):

        self._omega_cdm_h2_0     = omega_cdm_h2_0
        self._omega_baryons_h2_0 = omega_baryons_h2_0
        self._hubble_parameter   = hubble_parameter

        self._H_0_GEV           = ((1.e+4*self._hubble_parameter)/(units.KPC_TO_CM))*(units.H_BAR/units.GEV_TO_J)
        self._rhoc_0_GEV4       = (3.0/(8.0*np.pi))*pow(self._H_0_GEV,2)*pow(units.PLANCK_MASS_GEV,2)
        self._omega_photons_0   = (2*((np.pi*np.pi)/30.)*pow(T0_CMB_K * units.K_TO_GEV,4)/self._rhoc_0_GEV4)
        self._omega_neutrinos_0 = self._omega_photons_0 * units.N_EFF * (7./8.) * pow(4./11.,4./3.);

        self._omega_matter_h2_0    = self._omega_cdm_h2_0 + self._omega_baryons_h2_0
        self._omega_radiation_h2_0 = (self._omega_photons_0 + self._omega_neutrinos_0)*self._hubble_parameter**2
        self._omega_lambda_h2_0    = self._hubble_parameter**2 - self._omega_matter_h2_0 - self._omega_radiation_h2_0;

        self._omega_matter_0    = self._omega_matter_h2_0 / (self._hubble_parameter**2)
        self._omega_cdm_0       = self._omega_cdm_h2_0 / (self._hubble_parameter**2)
        self._omega_baryons_0   = self._omega_baryons_h2_0 / (self._hubble_parameter**2)
        self._omega_radiation_0 = self._omega_photons_0 + self._omega_neutrinos_0
        self._omega_lambda_0    = self._omega_lambda_h2_0 / (self._hubble_parameter**2)

        self._hubble2_to_critical_density_MSUN_MPC3 = 3. / (8*np.pi*units.G_NEWTON) * 1e+9 * units.KG_TO_MSUN * units.KPC_TO_M


    def omega_matter(self, z):
        return self._omega_matter_0 * (1+z)**3 
    
    def omega_radiation(self, z):
        return self._omega_radiation_0 * (1+z)**4

    def omega_cdm(self, z):
        return self._omega_cdm_0 * (1+z)**3

    def omega_baryons(self, z):
        return self._omega_baryons_0 * (1+z)**3
    
    def omega_neurtinos(self, z):
        return self._omega_neutrinos_0 * (1+z)**4
    
    def omega_lambda(self, z):
        return self._omega_lambda_0
    
    @property
    def omega_matter_0(self):
        return self._omega_matter_0 

    @property
    def omega_cdm_0(self):
        return self._omega_cdm_0
    
    @property
    def omega_radiation_0(self):
        return self._omega_radiation_0
    
    @property
    def omega_lambda_0(self):
        return self._omega_lambda_0
    
    @property
    def omega_baryons_0(self):
        return self._omega_baryons_0
    
    @property
    def hubble_constant(self):
        return Quantity(100*self._hubble_parameter, units.km / units.s / units.Mpc)
    
    @property
    def hubble_parameter(self):
        return self._hubble_parameter
    
    @property
    def critical_density_0(self):
        return Quantity(self.hubble_constant.value * self._hubble2_to_critical_density_MSUN_MPC3, units.Msun/units.Mpc**3)

    @property
    def rho_matter_average_0(self):
        return Quantity(self.omega_matter_0 * self.critical_density_0.value,  self.critical_density_0.unit)


    def E(self, z):
        return np.sqrt((self._Omh2_0*pow(1+z,3) + self._Orh2_0*pow(1+z,4) + self._Olh2_0)/(self._h**2))
    
    def hubble_rate(self, z: float) -> Quantity:
        _E2h2 = self._Omh2_0*pow(1+z,3) + self._Orh2_0*pow(1+z,4) + self._Olh2_0
        return Quantity(100.*np.sqrt(_E2h2), units.km / units.s / units.Mpc) 
    
    def critical_density_MSUN_MPC3(self, z):
        return (self.hubble_rate(z).value**2 * self._hubble2_to_critical_density_MSUN_MPC3,  units.Msun/units.Mpc**3)




PLANCK18 = BackgroundCosmology(0.1200, 0.02237, 0.9649)