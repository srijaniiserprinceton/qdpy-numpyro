import numpy as np
import matplotlib.pyplot as plt

from qdpy_jax import globalvars as gvar_jax

gvar= gvar_jax.GlobalVars()
outdir_dpy = f"{gvar.scratch_dir}/dpy_jax"
outdir_qdpy = f"{gvar.scratch_dir}/qdpy_jax"

# loading the respective multiplets
dpy_mults = np.load(f'{outdir_dpy}/qdpy_multiplets.npy')
qdpy_mults = np.load(f'{outdir_qdpy}/qdpy_multiplets.npy')

# loading the respective frequencies in mHz
omega_dpy = np.load(f'{outdir_dpy}/omega_qdpy_multiplets.npy') * 1e-3
omega_qdpy = np.load(f'{outdir_qdpy}/omega_qdpy_multiplets.npy') * 1e-3

plt.figure(figsize=(10,8))

plt.plot(dpy_mults[:,1], omega_dpy, 'ok', label='DPT multiplets', markersize=2)
plt.plot(qdpy_mults[:,1], omega_qdpy, 'or', label='QDPT multiplets', markersize=2)

plt.xlabel('$\ell$', fontsize=16)
plt.ylabel('$\\nu$ in mHz', fontsize=16)

plt.legend()
plt.tight_layout()

plt.savefig('inversion_mults.pdf')
