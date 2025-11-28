# plot_binding.py  (save as UTF-8)
import os, importlib.util, numpy as np, matplotlib.pyplot as plt
gui_path = r"C:\Users\pedro\Downloads\Nucleo Atomic Vis.py"
spec = importlib.util.spec_from_file_location('vis', gui_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# NuclearPhysics class is in the file you showed
NP = mod.NuclearPhysics

# choose Z range or A range; we'll plot B/A vs A for stable-ish Z=A/2 case
A_vals = np.arange(4,160)  # 4..159
Z_vals = (A_vals // 2).astype(int)
B_over_A = np.array([NP.binding_energy_per_nucleon(int(A), int(Z)) for A,Z in zip(A_vals, Z_vals)])

plt.figure(figsize=(8,4))
plt.plot(A_vals, B_over_A, '-', marker='o', markersize=3)
plt.xlabel('Mass number A')
plt.ylabel('Binding energy per nucleon (MeV)')
plt.title('Binding energy per nucleon (SEMF) — symmetric Z≈A/2')
plt.grid(True)
out = os.path.join(os.path.dirname(gui_path),'fig2_binding_curve.png')
plt.tight_layout()
plt.savefig(out, dpi=200)
print("Saved binding curve:", out)
plt.show()
