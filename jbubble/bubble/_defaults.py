"""
Centralized default parameter values for bubble models.

This module replaces hardcoded defaults scattered throughout bubble model
files, ensuring a single source of truth and making it easy to adjust
physical parameters globally.
"""

# Lipid-coated microbubble shell
GAMMA_LIPID = 1.07
CHI_LIPID = 0.38  # N/m
KAPPA_S_LIPID = 2.4e-9  # N·s/m

# Uncoated / bare gas bubble (air)
GAMMA_AIR = 1.4

# Water / liquid properties
MU_WATER = 0.00089  # Pa·s
RHO_WATER = 1000.0  # kg/m³
C_WATER = 1498.0  # m/s
P_ATM = 101.3e3  # Pa
SIGMA_WATER = 72e-3  # N/m

# Shell geometry scaling ratios
R_BUCKLE_RATIO = 0.99
R_BREAK_RATIO = 1.1

# Van der Waals divisor (Marmottant model)
VDW_DIVISOR = 5.61
