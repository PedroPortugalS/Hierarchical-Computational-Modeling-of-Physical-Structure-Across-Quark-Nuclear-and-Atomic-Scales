import numpy as np
from scipy.special import genlaguerre

# ------------------------------
#  Particle-level metrics
# ------------------------------

def quark_positions():
    """
    Returns example quark positions in a nucleon.
    """
    quarks = [
        {"type": "up", "position": np.array([0.0, 0.1, 0.0])},
        {"type": "down", "position": np.array([0.1, -0.1, 0.0])},
        {"type": "up", "position": np.array([-0.1, 0.0, 0.0])}
    ]
    return quarks

def flux_tube_lengths(quarks):
    """
    Calculate distances between quarks (effective flux-tube lengths).
    """
    lengths = []
    n = len(quarks)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(quarks[i]["position"] - quarks[j]["position"])
            lengths.append(dist)
    return lengths

# ------------------------------
#  Nuclear-level metrics
# ------------------------------

def nuclear_binding_energy(A, Z):
    """
    Semi-empirical mass formula (SEMF) for binding energy in MeV
    """
    # Constants
    av = 15.8      # MeV
    as_ = 18.3     # MeV
    ac = 0.714     # MeV
    aa = 23.2      # MeV

    # Pairing term δ
    if A % 2 != 0:
        delta = 0
    elif Z % 2 == 0:
        delta = 12.0 / np.sqrt(A)
    else:
        delta = -12.0 / np.sqrt(A)

    B = av*A - as_*A**(2/3) - ac*Z*(Z-1)/A**(1/3) - aa*((A-2*Z)**2)/A + delta
    return B

def nuclear_radius(A):
    """
    Empirical nuclear radius in fm
    """
    r0 = 1.20  # fm
    return r0 * A**(1/3)

# ------------------------------
#  Atomic-level metrics
# ------------------------------

def atomic_orbital_radius(n, Z_eff):
    """
    Hydrogenic Bohr radius scaled by effective nuclear charge
    Returns radius in meters
    """
    a0 = 5.29177210903e-11  # Bohr radius in meters
    return n**2 * a0 / Z_eff

def hydrogenic_wavefunction(n, l, m, Z_eff, r):
    """
    Compute hydrogenic radial wavefunction at distance r (m)
    """
    a0 = 5.29177210903e-11  # m
    rho = 2 * Z_eff * r / (n * a0)
    # Radial part (normalized simplified version)
    R = (rho**l) * np.exp(-rho/2) * genlaguerre(n-l-1, 2*l+1)(rho)
    return R

# ------------------------------
#  Main: Run all metrics
# ------------------------------

if __name__ == "__main__":
    # --- Particle-level ---
    print("=== Particle-level Metrics ===")
    quarks = quark_positions()
    for q in quarks:
        print(f"Quark: {q['type']}, Position: {q['position']}")
    lengths = flux_tube_lengths(quarks)
    print(f"Flux-tube lengths between quarks: {lengths}\n")

    # --- Nuclear-level ---
    print("=== Nuclear-level Metrics ===")
    examples = [(56,26), (12,6), (208,82)]  # A,Z examples: Fe-56, C-12, Pb-208
    for A,Z in examples:
        B = nuclear_binding_energy(A,Z)
        R = nuclear_radius(A)
        print(f"Nucleus A={A}, Z={Z} -> Binding Energy = {B:.2f} MeV, Radius = {R:.2f} fm")
    print()

    # --- Atomic-level ---
    print("=== Atomic-level Metrics ===")
    atomic_examples = [(1,1), (2,1), (3,2)]  # (n,Z_eff): H1s, H2s, He3s
    for n, Z_eff in atomic_examples:
        r = atomic_orbital_radius(n, Z_eff)
        print(f"n={n}, Z_eff={Z_eff} -> Orbital radius = {r:.3e} m")
