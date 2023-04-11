import numpy as np

# fundamental constants (in the SI units)
G_NEWTON    : float = 6.67408e-11; # [m^3.kg^-1.s^-2]
LIGHT_SPEED : float = 299792458;   # [m/s]
K_BOLTZMANN : float = 1.3806488e-23    # Boltzmann constant [J.K^-1]
E_CHARGE    : float = 1.602176565e-19  # [C = A.s]
H_BAR       : float = 1.054571726e-34; # reduced Planck constant [J.s]

N_EFF : float = 3.046

# mass conversions
EV_TO_KG = 1.782661845e-36
KG_TO_EV = 1./EV_TO_KG
MSUN_TO_KG = 1.32712442099e+20/G_NEWTON
KG_TO_MSUN = 1./MSUN_TO_KG

PLANCK_MASS_GEV = np.sqrt(H_BAR*LIGHT_SPEED/G_NEWTON)*KG_TO_EV*1e-9 # Planck mass [GeV]

# length conversions
KPC_TO_M : float  = 3.08567758149e+19
KPC_TO_CM : float = 3.08567758149e+21


# energy conversions
EV_TO_J  : float  = E_CHARGE
GEV_TO_J : float = 1.e+9*EV_TO_J

# temperature conversions
EV_TO_K  : float = EV_TO_J/K_BOLTZMANN
GEV_TO_K : float = 1.e+9*EV_TO_K
K_TO_GEV : float = 1./GEV_TO_K
