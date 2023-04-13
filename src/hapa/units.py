import numpy as np

# fundamental constants (in the SI units)
G_NEWTON: float    = 6.67408e-11; # [m^3.kg^-1.s^-2]
LIGHT_SPEED: float = 299792458;   # [m/s]
K_BOLTZMANN: float = 1.3806488e-23    # Boltzmann constant [J.K^-1]
E_CHARGE: float    = 1.602176565e-19  # [C = A.s]
H_BAR: float       = 1.054571726e-34; # reduced Planck constant [J.s]

N_EFF : float = 3.046

# mass conversions
EV_TO_KG: float   = 1.782661845e-36
KG_TO_EV: float   = 1./EV_TO_KG
MSUN_TO_KG: float = 1.32712442099e+20/G_NEWTON
KG_TO_MSUN: float = 1./MSUN_TO_KG

PLANCK_MASS_GEV = np.sqrt(H_BAR*LIGHT_SPEED/G_NEWTON)*KG_TO_EV*1e-9 # Planck mass [GeV]

# length conversions
KPC_TO_M: float  = 3.08567758149e+19
KPC_TO_CM: float = 3.08567758149e+21

# time conversions
M_EV: float = 1e+9/(197.3269718)
EV_TO_SM1: float = M_EV*LIGHT_SPEED
GEV_TO_SM1: float = 1e-9*M_EV*LIGHT_SPEED
YR_TO_S = 365.25*24*3600

# energy conversions
EV_TO_J : float  = E_CHARGE
GEV_TO_J: float = 1.e+9*EV_TO_J
J_TO_GEV: float = 1/GEV_TO_J

# temperature conversions
EV_TO_K  : float = EV_TO_J/K_BOLTZMANN
GEV_TO_K : float = 1.e+9*EV_TO_K
K_TO_GEV : float = 1./GEV_TO_K




_LENGTHS_KEYS: tuple = ('nm', 'mm', 'cm', 'm', 'km', 'pc', 'kpc', 'Mpc')
_LENGTHS_KEY_TO_POS: dict = dict(nm=0, mm=1, cm=2, m=3, km=4, pc=5, kpc=6, Mpc=7)
_CONVERT_LENGTHS: tuple = (1e-9, 1e-3, 1e-2, 1, 1e+3, 1e-3*KPC_TO_M, KPC_TO_M, 1e+3*KPC_TO_M)

_TIMES_KEYS: tuple   = ('s', 'yr', 'Myr', 'Gyr')
_TIMES_KEY_TO_POS: dict   = dict(s=0, yr=1, Myr=2, Gyr=3)
_CONVERT_TIMES: tuple = (1, YR_TO_S, 1e+6*YR_TO_S, 1e+9*YR_TO_S)

_MASSES_KEYS: tuple = ('g', 'kg', 'Msun')
_MASSES_KEY_TO_POS: dict = dict(g=0, kg=1, Msun=2)
_CONVERT_MASSES: tuple = (1e-3, 1, MSUN_TO_KG) 

_TEMPERATURES_KEYS: tuple = ('K')
_TEMPERATURES_KEY_TO_POS: dict = dict(K=0)
_CONVERT_TEMPERATURES: tuple = (1)

_ENERGIES_KEYS: tuple = ('eV', 'keV', 'MeV', 'GeV', 'TeV', 'J', 'kJ')
_ENERGIES_KEY_TO_POS: dict = dict(eV=0, keV=1, MeV=2, GeV=3, TeV=4, J=5, kJ=6)
_CONVERT_ENERGIES: tuple = (1e-9, 1e-6, 1e-3, 1, 1e+3, J_TO_GEV, 1e+3*J_TO_GEV)



class Unit:

    def __init__(self, **kwargs):  

        # Define the different units one can have trhough arrays
        self._lengths        = np.zeros(len(_LENGTHS_KEYS), dtype=np.float64)
        self._times          = np.zeros(len(_TIMES_KEYS),   dtype=np.float64)
        self._masses         = np.zeros(len(_MASSES_KEYS),  dtype=np.float64)
        self._temperatures   = np.zeros(len(_TEMPERATURES_KEYS),  dtype=np.float64)
        self._energies       = np.zeros(len(_ENERGIES_KEYS),      dtype=np.float64)

        self._name = ''

        for key in kwargs.keys():

            if key in _LENGTHS_KEYS:
                _ikey  = _LENGTHS_KEY_TO_POS[key]
                self._lengths[_ikey] = kwargs.get(key)
            elif key in _TIMES_KEYS:
                _ikey  = _TIMES_KEY_TO_POS[key]
                self._times[_ikey] = kwargs.get(key)
            elif key in _MASSES_KEYS:
                _ikey  = _MASSES_KEY_TO_POS[key]
                self._masses[_ikey] = kwargs.get(key)
            elif key in _TEMPERATURES_KEYS:
                _ikey  = _TEMPERATURES_KEY_TO_POS[key]
                self._temperatures[_ikey] = kwargs.get(key)
            elif key in _ENERGIES_KEYS:
                _ikey  = _ENERGIES_KEY_TO_POS[key]
                self._energies[_ikey] = kwargs.get(key)
            else:
                raise ValueError('Unit not recognised')

            _value = kwargs.get(key, 0)

            if _value == 1:
                self._name = self._name + key + '.'

            if _value !=1:
                _str_value = str(int(_value)) if _value.is_integer else _value
            if _value < 0: 
                self._name  = self._name + key + '^(' +  _str_value + ').'
            if _value > 1: 
                self._name  = self._name + key + '^' +  _str_value + '.'
    
        self._name = self._name[:-1]
        
    @property
    def name(self):
        return self._name
    
    def __str__(self):
        return self._name

    def __repr__(self):
        return self._name

        
    def _set_args(cls, lengths: np.ndarray, times: np.ndarray, masses: np.ndarray, temperatures: np.ndarray, energies: np.ndarray) -> None:

        args_lengths: dict       = {_LENGTHS_KEYS[i]: lengths[i] for i in np.where(lengths != 0)[0]}
        args_times: dict         = {_TIMES_KEYS[i]: times[i] for i in np.where(times != 0)[0]}
        args_masses: dict        = {_MASSES_KEYS[i]: masses[i] for i in np.where(masses != 0)[0]}
        args_temperatures: dict  = {_MASSES_KEYS[i]: temperatures[i] for i in np.where(temperatures != 0)[0]}
        args_energies: dict      = {_ENERGIES_KEYS[i]: energies[i] for i in np.where(energies != 0)[0]}

        return {**args_lengths, **args_times, **args_masses, **args_temperatures, **args_energies}

    def __mul__(self, other):

        new_lengths       = self._lengths + other._lengths
        new_times         = self._times   + other._times 
        new_masses        = self._masses  + other._masses
        new_temperatures  = self._temperatures  + other._temperatures
        new_energies      = self._energies  + other._energies

        return Unit(**self._set_args(new_lengths, new_times, new_masses, new_temperatures, new_energies))
    
    def __truediv__(self, other): 

        new_lengths       = self._lengths - other._lengths
        new_times         = self._times   - other._times 
        new_masses        = self._masses  - other._masses
        new_temperatures  = self._temperatures  - other._temperatures
        new_energies      = self._energies  - other._energies

        return Unit(**self._set_args(new_lengths, new_times, new_masses, new_temperatures, new_energies))

    def __rtruediv__(self, value):

        if value == 1:
            new_lengths = -self._lengths
            new_times   = -self._times
            new_masses  = -self._masses
            new_temperatures = -self._temperatures
            new_energies      = -self._energies

            return Unit(**self._set_args(new_lengths, new_times, new_masses, new_temperatures, new_energies))
        else:
            raise SyntaxError('No disivion between integer different than 1 and units')

    def __pow__(self, value):

        if isinstance(value, (float, int)) : 
            new_lengths = value * self._lengths
            new_times   = value * self._times
            new_masses  = value * self._masses
            new_temperatures = value * self._temperatures
            new_energies     = value * self._energies

            return Unit(**self._set_args(new_lengths, new_times, new_masses, new_temperatures, new_energies)) 
        else:
            raise ValueError('Can only raise an unit to a scalar power')

    def __eq__(self, other): 
        
        ok_lengths       = np.all(self._lengths == other._lengths)
        ok_times         = np.all(self._times   == other._times)
        ok_masses        = np.all(self._masses == other._masses)
        ok_temperatures  = np.all(self._temperatures   == other._temperatures)
        ok_energies      = np.all(self._energies   == other._energies)

        return (ok_lengths and ok_times and ok_masses and ok_temperatures and ok_energies)

    @property   
    def powers_SI(self) -> tuple:

        _n_length = np.sum(self._lengths) + 2*np.sum(self._energies)
        _n_mass   = np.sum(self._masses) + np.sum(self._energies)
        _n_times  = np.sum(self._times) - 2*np.sum(self._energies)
        _n_temp   = np.sum(self._temperatures)
    
        return (_n_length, _n_mass, _n_times, _n_temp)


# Define the basics building bloc units
s    = Unit(s=1)
nm   = Unit(nm=1)
cm   = Unit(cm=1)
m    = Unit(m=1)
km   = Unit(km=1)
kpc  = Unit(kpc=1)
Mpc  = Unit(Mpc=1)
Msun = Unit(Msun=1)
g    = Unit(g=1)
kg   = Unit(kg=1)
GeV  = Unit(GeV=1)


_NO_UNIT = Unit()

class Quantity:

    def __init__(self, value, unit):
        self._value = value
        self._unit  = unit


    @property
    def reduced_unit(self):
        _unit = self._unit
        return Unit(m =  np.sum(_unit._lengths), kg = np.sum(_unit._masses), s =  np.sum(_unit._times), K = np.sum(_unit._temperatures), GeV= np.sum(_unit._energies))

    @property
    def reduced_value(self):
        
        _unit = self._unit

        _conversion_lenghts        = np.prod([_CONVERT_LENGTHS[i]**_unit._lengths[i] for i in np.where(_unit._lengths != 0)[0]])
        _conversion_times          = np.prod([_CONVERT_TIMES[i]**_unit._times[i] for i in np.where(_unit._times != 0)[0]])
        _conversion_masses         = np.prod([_CONVERT_MASSES[i]**_unit._masses[i] for i in np.where(_unit._masses != 0)[0]])
        _conversion_temperatures   = np.prod([_CONVERT_TEMPERATURES[i]**_unit._temperatures[i] for i in np.where(_unit._temperatures != 0)[0]])
        _conversion_energies       = np.prod([_CONVERT_ENERGIES[i]**_unit._energies[i] for i in np.where(_unit._energies != 0)[0]])

        return self._value * _conversion_lenghts * _conversion_times * _conversion_masses * _conversion_temperatures * _conversion_energies
        
    @property
    def value(self):
        return self._value
    
    @property
    def unit(self):
        return self._unit
    
    def convert(self, new_unit):
        _diff_unit = self._unit / new_unit

        if self._unit.powers_SI != new_unit.powers_SI:
            raise ArithmeticError('Cannot convert ' + str(self._unit) + ' to ' + str(new_unit))

        _conversion_lenghts        = np.prod([_CONVERT_LENGTHS[i]**_diff_unit._lengths[i] for i in np.where(_diff_unit._lengths != 0)[0]])
        _conversion_times          = np.prod([_CONVERT_TIMES[i]**_diff_unit._times[i] for i in np.where(_diff_unit._times != 0)[0]])
        _conversion_masses         = np.prod([_CONVERT_MASSES[i]**_diff_unit._masses[i] for i in np.where(_diff_unit._masses != 0)[0]])
        _conversion_temperatures   = np.prod([_CONVERT_TEMPERATURES[i]**_diff_unit._temperatures[i] for i in np.where(_diff_unit._temperatures != 0)[0]])
        _conversion_energies       = np.prod([_CONVERT_ENERGIES[i]**_diff_unit._energies[i] for i in np.where(_diff_unit._energies != 0)[0]])

        self._value = self._value * _conversion_lenghts * _conversion_times * _conversion_masses * _conversion_temperatures * _conversion_energies
        
        self._unit  = new_unit 
    

    def __str__(self):
        return str(self._value) + ' ' + self._unit.name

    def __repr__(self):
        return str(self._value) + ' ' + self._unit.name

    def __mul__(self, other): 
        if isinstance(other, Quantity):
            if self.reduced_unit * other.reduced_unit != _NO_UNIT:
                new_value = self.value * other.value
                return Quantity(new_value, self.unit * other.unit)
            else:
                new_value = self.value * other.value
                return new_value
        else:
            new_value = self.value * other
            return Quantity(new_value, self.unit)
    
    def __rmul__(self, other):
        if isinstance(other, Quantity):
            if self.reduced_unit * other.reduced_unit != _NO_UNIT:
                return Quantity(self.value * other.value, self.unit * other.unit)
            else:
                return self.reduced_value * other.reduced_value
        else:
            new_value = self.value * other
            return Quantity(new_value, self.unit)

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            if self.reduced_unit / other.reduced_unit != _NO_UNIT:
                return Quantity(self.value / other.value, self.unit / other.unit)
            else:
                return self.reduced_value / other.reduced_value
        else:
            return Quantity(self.value / other, self.unit)
        
    def __rtruediv__(self, other):
        if isinstance(other, Quantity):
            if  other.reduced_unit / self.reduced_unit != _NO_UNIT:
                return Quantity(other.value / self.value, other.unit / self.unit)
            else:
                return other.reduced_value / self.reduced_value 
        else:
            return Quantity(other / self.value , 1/self.unit)
    
    def __pow__(self, value):
        return Quantity(self.value ** value, self.unit ** value)
    
    def __add__(self, other):
        assert (self.unit == other.unit)
        return Quantity(self.value + other.value, self.unit)

    def __sub__(self, other):
        assert (self.unit == other.unit)
        return Quantity(self.value - other.value, self.unit)

  