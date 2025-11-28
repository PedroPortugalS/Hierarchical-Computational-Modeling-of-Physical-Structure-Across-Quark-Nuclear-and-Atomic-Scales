import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Nuclear Metrics
# -----------------------------
# Sample data for nuclei: A = mass number, Z = protons
nuclei = [
    {'A': 12, 'Z': 6},
    {'A': 56, 'Z': 26},
    {'A': 208, 'Z': 82}
]

# SEMF parameters (simplified for visualization)
av = 15.8  # MeV
as_ = 18.3
ac = 0.714
aa = 23.2
delta = lambda A, Z: 12.0 / np.sqrt(A) if (Z%2==0 and (A-Z)%2==0) else (-12.0/np.sqrt(A) if (Z%2==1 and (A-Z)%2==1) else 0)

def binding_energy(A, Z):
    return av*A - as_*A**(2/3) - ac*Z*(Z-1)/A**(1/3) - aa*(A-2*Z)**2/A + delta(A,Z)

def nuclear_radius(A):
    r0 = 1.2  # fm
    return r0 * A**(1/3)

# Compute binding energies per nucleon and radii for plotting
A_range = np.arange(1, 210)
BE_per_nucleon = [binding_energy(A, A//2)/A for A in A_range]
Radii = [nuclear_radius(A) for A in A_range]

# -----------------------------
# Atomic Metrics
# -----------------------------
# Hydrogenic orbital radii: r_n = n^2 a0 / Z_eff
a0 = 5.292e-11  # Bohr radius in meters
orbitals = [
    {'n': 1, 'Z_eff': 1},
    {'n': 2, 'Z_eff': 1},
    {'n': 3, 'Z_eff': 2},
]

orbital_radii = [orb['n']**2 * a0 / orb['Z_eff'] for orb in orbitals]
orbital_labels = [f"n={orb['n']}, Z_eff={orb['Z_eff']}" for orb in orbitals]

# -----------------------------
# Figure 5: Nuclear Binding Energy per Nucleon
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(A_range, BE_per_nucleon, color='blue', lw=2)
plt.scatter([n['A'] for n in nuclei], [binding_energy(n['A'], n['Z'])/n['A'] for n in nuclei], color='red', zorder=5)
plt.title("Nuclear Binding Energy per Nucleon")
plt.xlabel("Mass Number A")
plt.ylabel("Binding Energy per Nucleon (MeV)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure5_BindingEnergy.png")
plt.show()

# -----------------------------
# Figure 6: Nuclear Radii vs Mass Number
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(A_range, Radii, color='green', lw=2)
plt.scatter([n['A'] for n in nuclei], [nuclear_radius(n['A']) for n in nuclei], color='red', zorder=5)
plt.title("Nuclear Radii vs Mass Number")
plt.xlabel("Mass Number A")
plt.ylabel("Radius (fm)")
plt.grid(True)
plt.tight_layout()
plt.savefig("Figure6_NuclearRadii.png")
plt.show()

# -----------------------------
# Figure 7: Hydrogenic Orbital Radii
# -----------------------------
plt.figure(figsize=(8,5))
plt.bar(orbital_labels, orbital_radii, color='purple')
plt.title("Hydrogenic Orbital Radii")
plt.ylabel("Radius (m)")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("Figure7_OrbitalRadii.png")
plt.show()
