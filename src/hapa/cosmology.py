import numpy as np
from . import units


class BackgroundCosmology:

    def __init__(self, Och2_0, Obh2_0, h, T0_CMB_K = 2.72548):

        self._Och2_0        = Och2_0
        self._Obh2_0        = Obh2_0
        self._h             = h

        self._H_0_GEV     = ((1e+4*self._h)/(units.KPC_TO_CM))*(units.H_BAR/units.GEV_TO_J)
        self._rhoc_0_GEV4 = (3.0/(8.0*np.pi))*pow(self._H_0_GEV,2)*pow(units.PLANCK_MASS_GEV,2)
        self._Og          = (2*((np.pi*np.pi)/30.)*pow(T0_CMB_K * units.K_TO_GEV,4)/self._rhoc_0_GEV4)
        self._On          = self._Og * units.N_EFF * (7./8.) * pow(4./11.,4./3.);

        self._Omh2_0   = self._Och2_0 + self._Obh2_0
        self._Omh2_0 = self._Och2_0 + self._Obh2_0
        self._Orh2_0 = (self._Og + self._On)*self._h**2
        self._Olh2_0   = self._h**2 - self._Omh2_0 - self._Orh2_0;

        self._Om_0 = self._Omh2_0 / (self._h**2)
        self._Oc_0 = self._Och2_0 / (self._h**2)
        self._Ob_0 = self._Obh2_0 / (self._h**2)
        self._Or_0 = self._Orh2_0 / (self._h**2)
        self._Ol_0 = self._Olh2_0 / (self._h**2)

        self._hubble2_to_critical_density_MSUN_MPC3 = 3. / (8*np.pi*units.G_NEWTON) * 1e+9 * units.KG_TO_MSUN * units.KPC_TO_M


    def omega_matter(self, z):
        return self._Om_0 * (1+z)**3 
    
    def omega_radiation(self, z):
        return self._Or_0 * (1+z)**4

    def omega_cdm(self, z):
        return self._Oc_0 * (1+z)**3

    def omega_baryons(self, z):
        return self._Ob_0 * (1+z)**3
    
    def omega_neurtinos(self, z):
        return self._On_0 * (1+z)**4
    
    def omega_lambda(self, z):
        return self._Ol_0
    
    @property
    def omega_matter_0(self):
        return self._Om_0 

    @property
    def omega_cdm_0(self):
        return self._Oc_0
    
    @property
    def omega_radiation_0(self):
        return self._Or_0
    
    @property
    def omega_lambda_0(self):
        return self._Ol_0
    
    @property
    def omega_baryons_0(self):
        return self._Ob_0
    
    @property
    def hubble_constant(self):
        """
        # Hubble constant in [km/s/Mpc]
        """
        return 100*self._h
    
    @property
    def h(self):
        return self._h


    def E(self, z):
        return np.sqrt((self._Omh2_0*pow(1+z,3) + self._Orh2_0*pow(1+z,4) + self._Olh2_0)/(self._h**2))
    
    def hubble_rate(self, z):
        """
        # Hubble constant in [km/s/Mpc]
        """
        _E2h2 = self._Omh2_0*pow(1+z,3) + self._Orh2_0*pow(1+z,4) + self._Olh2_0
        return 100.*np.sqrt(_E2h2)
    
    @property
    def critical_density_0_MSUN_MPC3(self):
        return  self.hubble_constant * self._hubble2_to_critical_density_MSUN_MPC3

    @property
    def rho_matter_average_0_MSUN_MPC3(self):
        return self.omega_matter_0 * self.critical_density_0_MSUN_MPC3

    def critical_density_MSUN_MPC3(self, z):
        """
        # Critical density in [MSUN/MPC^3]
        """
        return self.hubble_rate(z)**2 * self._hubble2_to_critical_density_MSUN_MPC3




PLANCK18 = BackgroundCosmology(0.1200, 0.02237, 0.9649)