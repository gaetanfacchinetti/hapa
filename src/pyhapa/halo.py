import numpy as np
from scipy import integrate
from scipy import interpolate
from scipy import optimize

import astropy.cosmology
import astropy.constants as const
import astropy.units as units

from functools import partial


from abc import abstractmethod

xmin_int = 1e-20
xmax_int = 1e+4


class HaloProfile() :

    """
    Dimensionless halo profile and derived quantities
    """

    def __init__(self, density_profile = None) :
        self._density_profile       = density_profile
        self.mass_profile_computed  = False

        self.luminosity_profile = np.vectorize(self._luminosity_profile, excluded=['l_angular'], doc="TODO")

    @property
    def density_profile(self):
        return self._density_profile

    
    @abstractmethod
    def mass_profile(self, x: float):
        ...

    
    def _compute_mass_profile(self):
        """
        Function that pre-compute the mass profile and stores it in an interpolated function if needed again
        """

        # We interpolate on a grid with 20 points per decades (at least)
        n_points = np.max([20*int(np.log10(xmax_int/xmin_int)), 150])
        _x_arr = np.logspace(np.log10(xmin_int), np.log10(xmax_int), n_points)

        _mass : float = 0
        _mass_arr = np.array([0])
        for ix, x in enumerate(_x_arr[:-1]) :
            _mass = _mass + integrate.quad(lambda y : self.density_profile(np.exp(y)) * (np.exp(y)**3), np.log(_x_arr[ix]), np.log(_x_arr[ix+1]), epsrel=1e-5)[0]
            _mass_arr = np.append(_mass_arr, _mass)
        
        _mass_arr[0] = 1e-10*_mass_arr[1]
        self._interp_mass_profile = interpolate.interp1d(np.log10(_x_arr), np.log10(_mass_arr))

        self.mass_profile_computed = True


    
    def velocity_dispersion_profile(self, x: float, x_max: float):
        """
        ## Velocity dispersion (dimensionless)

        Params:
        ----------
        x: float
            value where to evaluate the potential
        x_max: float
            maximal extenstion of the halo

        """

        def __integrand(Z, Y):
            """ Integrand of the integral we want to evaluate """

            _z = np.exp(Z)
            _y = np.exp(Y)

            return self.density_profile(_y) / _y * self.density_profile(_z) * (_z**3)

        def __boundary_min(Y):
            return np.log(xmin_int)

        def __boundary_max(Y):
            return Y #np.log(np.exp(Y))

        return integrate.dblquad(__integrand, np.log(x), np.log(x_max), __boundary_min, __boundary_max, epsrel=1e-3)[0]/self.density_profile(x)


    def gravitational_potential_profile(self, x: float, x_max: float):
        """
        ## Gravitational potential (dimensionless)

        Params:
        -------
        x: float
            value where to evaluate the potential
        x_max: float
            maximal extenstion of the halo

        Returns:
        --------
        gravitational potential: float
        """

        def __integrand(Z, Y):

            _z = np.exp(Z)
            _y = np.exp(Y)

            return 1. / _y *  self.density_profile(_z) * (_z**3)

        def __boundary_min(Y):
            return np.log(self.xmin_int)

        def __boundary_max(Y):
            return Y #np.log(np.exp(Y))

        return integrate.dblquad(__integrand, np.log(x), np.log(x_max), __boundary_min, __boundary_max, epsrel=1e-3)[0]
    
    
    def _luminosity_profile(self, x_delta, l_angular = 0):
    

        if l_angular > 1 :
            ValueError("Cannot set values of l_angular larger than 0")

        def __integrand(X) :
            _x = np.exp(X)
            if l_angular == 0 :
                return (self.density_profile(_x)**2)*(_x**2)*_x
            if l_angular == 1:
                return 6*self.density_profile(_x)*(self.mass_profile(_x)**2)/(_x**2)*_x
        
        return integrate.quad(__integrand, np.log(xmin_int), np.log(x_delta), epsrel=1e-3)[0]






## Generic functions in the module to convert every quantities together

def c_delta_from_rhos(rhos, delta, rho_ref, mass_profile):
    return optimize.bisect(_solve_for_concentration, 1.1*xmin_int, 0.9*xmax_int, args = (rhos, delta, rho_ref, mass_profile))

def _solve_for_concentration(c, rhos, delta, rho_ref, mass_profile):
    return c**3/mass_profile(c) - 3 * rhos.to('M_sun/kpc^3') / delta / rho_ref.to('M_sun/kpc^3')
    
def m_delta_from_rhos_and_rs(rhos, rs, delta, rho_ref, mass_profile):
    c_delta = c_delta_from_rhos(rhos, delta, rho_ref, mass_profile)
    return 4 * np.pi * rhos.to('M_sun/kpc^3') * rs.to('kpc')**3 * mass_profile(c_delta)
    
def rhos_from_c_delta(c_delta, delta, rho_ref, mass_profile) :
    return delta * rho_ref.to('M_sun/kpc^3')  / 3 * c_delta**3 / mass_profile(c_delta)

def rs_from_c_delta_and_m_delta(c_delta, m_delta, delta, rho_ref):
    return (3*m_delta.to('M_sun')/(4*np.pi * delta * rho_ref.to('M_sun/kpc^3')))**(1./3.) / c_delta

def m_delta_from_rs_and_c_delta(rs, c_delta, delta, rho_ref, mass_profile):
    rhos = rhos_from_c_delta(c_delta, delta, rho_ref, mass_profile)
    return 4 * np.pi * rhos.to('M_sun/kpc^3') * rs.to('kpc')**3 * mass_profile(c_delta)

def rs_from_rhos_and_m_delta(rhos, m_delta, delta, rho_ref, mass_profile):
    c_delta = c_delta_from_rhos(rhos, delta, rho_ref, mass_profile)
    return (3*m_delta.to('M_sun')/(4*np.pi * delta * rho_ref.to('M_sun/kpc^3')))**(1./3.) / c_delta

def rho_ref(cosmo, z, is_critical):
    return cosmo.critical_density(z).to('M_sun/kpc^3') * (1. if is_critical is True else cosmo.Om(z))

def r_delta_from_rhos_and_rs(rhos, rs, delta, rho_ref, mass_profile):
    m_delta = m_delta_from_rhos_and_rs(rhos, rs, delta, rho_ref, mass_profile)
    return (3*m_delta.to('M_sun')/(4*np.pi * delta * rho_ref.to('M_sun/kpc^3')))**(1./3.)


DEFAULT_COSMO = astropy.cosmology.Planck18


class ABGHaloProfile(HaloProfile):
    
    def __init__(self, alpha, beta, gamma):
        
        self._alpha = alpha
        self._beta  = beta
        self._gamma = gamma
        
        self.params = [self._alpha, self._beta, self._gamma]

        super().__init__(density_profile = self.abg_density_profile)


    def abg_density_profile(self, x):
        return x**(-self._gamma)*(1+x**self._alpha)**(-(self._beta-self._gamma)/self._alpha)
        

    def mass_profile(self, x:float):

        if self.params == [1, 3, 1]:
            return np.log(1+x) - x/(1+x) if x > 1e-3 else (x**2/2. - 2.*x**3/3. + 3.*x**4/4. - 4.*x**5/5.)
        if self.params == [1, 3, 1.5]:
            return -((2*x)/np.sqrt(x * (1 + x))) + 2 * np.arcsinh(np.sqrt(x)) if x > 1e-2 else np.sqrt(x)*(2.*x/3. - 3.*x**2/5. + 15.*x**(3.5)/28. - 35.*x**4/72.)
        # Add here new analytical formulas
        
        else :
            if self.mass_profile_computed is False:
                self._compute_mass_profile()
        
            return 10**self._interp_mass_profile(np.log10(x))



class TruncatedHaloProfile(HaloProfile):
    
    def __init__(self, init_density_profile, x_turn):
        
        self._init_density_profile  = init_density_profile
        self._x_turn                = x_turn
        self._rho_turn              = self._init_density_profile(self._x_turn)

        super().__init__(density_profile = self.density_profile)


    def density_profile(self, x):
        
        _rho : float = self._init_density_profile(x)
        
        if x > self._x_turn :
            return _rho
        else :
            return self._rho_turn
        

    def mass_profile(self, x:float):

        if self.mass_profile_computed is False:
            self._compute_mass_profile()
        
        return 10**self._interp_mass_profile(np.log10(x))





class Halo():
    
    
    """
    # Class to define generalised NFW halos

    Params:
    -------
    rhos: Astropy Quantity (mass / length^3)
        scale density of the halo
    rs: Astropy Quantity (length)
        scale radius of the halo
    c_delta: float
        virial concentration of the halo
    m_delta: Astropy Quantity (mass)
        virial mass of the halo
    delta: float, optional
        overdensity defining the virial quantities
        by default, delta = 200
    z: float, optional
        redshift at which the halo is defined
        by default, z = 0
    is_critical: bool, optional
        if true relates the virial parameters to the scale parameters via the critical density
        if false relates the virial parameters to the scale parameters via the matter density
        default is true
    cosmo: astropy.cosmology module, optional
        cosmological model
        by default astropy.cosmology.Planck18
    """
    
    def __init__(self, halo_profile, rhos : float = None, rs : float = None, c_delta : float = None, m_delta : float = None, **kwargs):
       
        if rhos is None and c_delta is None:
            ValueError("rhos and rs cannot be simultaneously none")
        if rhos is not None and c_delta is not None:
            ValueError("rhos and c_delta cannot be both set to a value")

        self._halo_profile  = halo_profile
        self._rhos          = rhos
        self._rs            = rs

        self._z           = None
        self._delta       = None
        self._is_critical = None
    

        # In case we need thes values we load them
        if rhos is None or rs is None:
            _cosmo            = kwargs.get('cosmo', DEFAULT_COSMO)
            self._z           = kwargs.get('z', 0)
            self._is_critical = kwargs.get('is_critical', True)
            self._delta       = kwargs.get('delta', 200)
            _rho_ref          = rho_ref(_cosmo, self._z, self._is_critical)


        if rhos is None :
            self._rhos = rhos_from_c_delta(c_delta, self._delta, _rho_ref, halo_profile.mass_profile)
        if rs is None and c_delta is not None and m_delta is not None:
            self._rs   = rs_from_c_delta_and_m_delta(c_delta, m_delta, self._delta, _rho_ref)
        if rs is None and rhos is not None and m_delta is not None:
            self._rs   = rs_from_rhos_and_m_delta(rhos, m_delta, self._delta, _rho_ref, halo_profile.mass_profile)


    
    @property
    def rhos(self) -> float:
        return self._rhos

    @property
    def rs(self) -> float:
        return self._rs

    @property
    def halo_profile(self):
        return self._halo_profile


    def c_delta(self, z= 0, delta=200, is_critical = True, cosmo = DEFAULT_COSMO) :
        if (self._z is not None and self._z != z) \
            or (self._delta is not None and self._delta != delta) \
            or (self._is_critical is not None and self._is_critical != is_critical):
            print("WARNING: evaluating c_delta for z, delta or is_critical at which it was not originally defined")
        _rho_ref = rho_ref(cosmo, z, is_critical)
        return c_delta_from_rhos(self.rhos, delta, _rho_ref, self.halo_profile.mass_profile)

    def m_delta(self, z=0, delta=200, is_critical = True, cosmo = DEFAULT_COSMO) :
        if (self._z is not None and self._z != z) \
            or (self._delta is not None and self._delta != delta) \
            or (self._is_critical is not None and self._is_critical != is_critical):
            print("WARNING: evaluating m_delta for z, delta or is_critical at which it was not originally defined")
        _rho_ref = rho_ref(cosmo, z, is_critical)
        return m_delta_from_rhos_and_rs(self.rhos, self.rs, delta, _rho_ref, self.halo_profile.mass_profile)

    def r_delta(self, z=0, delta=200, is_critical = True, cosmo = DEFAULT_COSMO) :
        if (self._z is not None and self._z != z) \
            or (self._delta is not None and self._delta != delta) \
            or (self._is_critical is not None and self._is_critical != is_critical):
            print("WARNING: evaluating r_delta for z, delta or is_critical at which it was not originally defined")
        _rho_ref = rho_ref(cosmo, z, is_critical)
        return r_delta_from_rhos_and_rs(self.rhos, self.rs, delta, _rho_ref, self.halo_profile.mass_profile)

    def __str__(self) -> str:
        return "Halo of parameters :" + \
                "\n| rhos    = {:.2e}".format(self.rhos) + \
                "\n| rs      = {:.2e}".format(self.rs)


    ## Shortcut
    def mass_profile(self, x:float) -> float:
        return self.halo_profile.mass_profile(x)

    def density_profile(self, x:float) -> float:
        return self.halo_profile.density_profile(x)

    def enclosed_mass(self, r:float) -> float:
        return 4*np.pi * self.rhos *self.rs **3 * self.mass_profile(r/self.rs)

    def one_halo_boost(self, l_angular : int = 0, z: float = 0, delta: float = 200, is_critical: bool = True, cosmo = DEFAULT_COSMO) -> float:
        """
        Boost factor due to one halo (dimensionless)
        """
        
        _c_delta = self.c_delta(z, delta, is_critical, cosmo)
        _rho_m   = cosmo.critical_density(z).to('M_sun/kpc^3') * cosmo.Om(z)
        _beta    = self.halo_profile.luminosity_profile(_c_delta, l_angular)

        return self.rhos/_rho_m/self.halo_profile.mass_profile(_c_delta) * (4*np.pi*const.G.to('m^2*kpc/(M_sun*s^2)') * self.rhos * self.rs**2/const.c**2 )**l_angular * _beta


class TruncatedHalo(Halo):


    def __init__(self, halo_profile, rhos: float = None, rs: float = None, c_delta: float = None, m_delta: float = None, **kwargs):

        ## Call the parent constructor

        self._init_halo_profile = halo_profile

        _cosmo     : float = kwargs.get('cosmo', DEFAULT_COSMO)
        _sigv      : float = kwargs.get('sigv', 3e-26 * units.cm**3/units.s)
        _mchi      : float = kwargs.get('mchi', 1 * units.TeV / const.c**2)
        _zf        : float = kwargs.get('zf', 1000)
        _z         : float = kwargs.get('z', 0)
        _tz                = _cosmo.lookback_time(_z)
        _tzf               = _cosmo.lookback_time(_zf)
        self._rho_turn     = self._rho_turn_func(_mchi, _sigv, _tzf - _tz)
      
        # In case we need thes values
        _is_critical : float = kwargs.get('is_critical', True)
        _delta       : float = kwargs.get('delta', 200)
        _rho_ref     : float = rho_ref(_cosmo, _z, _is_critical)

        if rhos is None and c_delta is None:
            ValueError("rhos and rs cannot be simultaneously none")
        if rhos is not None and c_delta is not None:
            ValueError("rhos and c_delta cannot be both set to a value")
        
        if c_delta is not None:
            self._x_turn = self._x_turn_func(self._rho_turn, c_delta = c_delta, delta = _delta, rho_ref=_rho_ref)
        if rhos is not None:
            self._x_turn = self._x_turn_func(self._rho_turn, rhos=rhos)
            

        halo_profile = TruncatedHaloProfile(init_density_profile=self._init_halo_profile.density_profile, x_turn = self._x_turn)
        super().__init__(halo_profile = halo_profile, rhos = rhos, rs = rs, c_delta = c_delta, m_delta = m_delta, delta = _delta, z = _z, cosmo = _cosmo)
        

    @property
    def rhos(self) -> float:
        return self._rhos

    @property
    def rs(self) -> float:
        return self._rs

    @property
    def x_turn(self) -> float:
        return self._x_turn

    @property
    def rho_turn(self) -> float:
        return self._rho_turn

    def truncated_profile(self, x : float) -> float :
        
        _rho : float = self._init_halo_profile.density_profile(x)
        
        if x > self._x_turn :
            return _rho
        else :
            return self._init_halo_profile.density_profile(self._x_turn)


    def _rho_turn_func(self, mchi, sigv, delta_t)-> float:
        return mchi.to('M_sun') / sigv.to('kpc^3/s') / delta_t.to('s')


    def _x_turn_func(self, rho_turn: float, rhos = None, c_delta = None, delta: float = None, rho_ref: float = None) -> float :
        return optimize.bisect(self.__solve_for_x_turn, 1.1*xmin_int, 0.9*xmax_int, args=(rho_turn, rhos, c_delta, delta, rho_ref))


    def __solve_for_x_turn(self, x : float, rho_turn: float, rhos: float = None, c_delta: float = None, delta: float = None, rho_ref: float = None) -> float:
    
        rho_x = self._init_halo_profile.density_profile(x)
        
        if rhos is not None:
            res = np.log10(rhos *  rho_x / rho_turn)
       
        elif c_delta is not None:
            
            integ = 0
            
            if c_delta > x :
                integ = integrate.quad(lambda y : self._init_halo_profile.density_profile(np.exp(y)) * (np.exp(y)**3), np.log(x), np.log(c_delta), epsrel=1e-5)[0]
            
            mass_term = rho_x/3. * x**3 + integ
            res = np.log10(rho_turn / (delta * rho_ref.to('M_sun/kpc**3')/3. * c_delta**3 * rho_x / mass_term))
        
        return res.value
