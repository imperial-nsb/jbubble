import jax.numpy as jnp

# ---------------- Threshold Functions ----------------

def expansion_threshold(R, R0, threshold):
    R_max = jnp.max(R)
    ratio = R_max / R0
    exceeded = jnp.where(ratio > threshold, 1, 0)
    return ratio, exceeded


def KE_threshold(R, vel, R0, rho_L, c_L, threshold):
    max_energy = 2 * jnp.pi * rho_L * jnp.max((R**3 * vel**2))
    ratio = max_energy / (R0**3 * c_L**2)
    exceeded = jnp.where(ratio > threshold, 1, 0)
    return ratio, exceeded


# ---------------- Threshold Dictionaries ----------------

expansion_thresholds = {
    "expansion_2R0": lambda R, R0: expansion_threshold(R, R0, 2.0),
    "expansion_1.75R0": lambda R, R0: expansion_threshold(R, R0, 1.75),
    "expansion_2.3R0": lambda R, R0: expansion_threshold(R, R0, 2.3),
    "expansion_10R0": lambda R, R0: expansion_threshold(R, R0, 10.0),
}

KE_thresholds = {
    "KE_2": lambda R, vel, R0, rho_L, c_L:
        KE_threshold(R, vel, R0, rho_L, c_L, 2.0),
    "KE_1.5": lambda R, vel, R0, rho_L, c_L:
        KE_threshold(R, vel, R0, rho_L, c_L, 1.5),
    "KE_1": lambda R, vel, R0, rho_L, c_L:
        KE_threshold(R, vel, R0, rho_L, c_L, 1),
}