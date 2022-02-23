import numpy as np
import argparse
import sys
#-------------- local imports --------------------------------
from qdpy import globalvars as gvar_jax
#------------------------- argument parser -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--nmin", help="min radial order", type=int)
parser.add_argument("--nmax", help="max radial order", type=int)
parser.add_argument("--lmin", help="min angular degree", type=int)
parser.add_argument("--lmax", help="max angular degree", type=int)
parser.add_argument("--exclude_qdpy", help="choose modes not in qdpy",
                    type=int, default=0)
parser.add_argument("--instrument", help="mdi or hmi",
                    type=str, default='hmi')
parser.add_argument("--outdir", help="dpy or qdpy",
                    type=str, default='dpy_jax')
ARGS = parser.parse_args()
#-----------------------------------------------------------------------
# {{{ def get_exclude_mask(exclude_qdpy=False):
def get_exclude_mask(exclude_qdpy=False):
    mask = np.ones_like(obsdata[:, 0], dtype=np.bool)
    if exclude_qdpy:
        qdpy_mults = np.load(f'{GVARS.scratch_dir}/qdpy_jax/qdpy_multiplets.npy')
        qdpy_ell = qdpy_mults[:, 1]
        qdpy_enn = qdpy_mults[:, 0]
        for i in range(len(qdpy_ell)):
            midx, midx_efs = modedata_exists(obsdata, qdpy_ell[i],
                                             qdpy_enn[i], 0)
            if midx != None:
                mask[midx] = False
    return mask
# }}} get_exclude_mask(exclude_qdpy=False)

# {{{ def modedata_exists(data, l, n, m):
def modedata_exists(data, l, n, m):
    '''Checks if mode-data for a chosen mode exists'''
    try:
        modeindex = np.where((data[:, 0] == l)*
                             (data[:, 1] == n))[0][0]
        modeindex_efs = GVARS.nl_all.index((int(n), int(l)))
    except:
        print("MODE NOT FOUND : l = %3s, n = %2s" %(l, n))
        return None, None
    return modeindex, modeindex_efs
# }}} modedata_exists(data, l, n, m)


# {{{ def get_multiplet_list(exclude_qdpy=False):
def get_multiplet_list(exclude_qdpy=False):
    # initializing (first value discarded later)
    nl_arr = np.array([-1, -1], dtype=int)
    omega_arr = np.array([-1.])

    for n in range(nmin, nmax+1):
        for l in range(lmin, lmax+1):
            a, b = modedata_exists(obsdata, l, n, 0)
            if (a != None):
                nl_arr = np.vstack((nl_arr, np.array([n, l])))
                omega_arr = np.append(omega_arr, a)
    # rejecting the first dummy entry
    return nl_arr[1:], omega_arr[1:]
# }}} get_multiplet_list(exclude_qdpy=False)


# {{{ def print_multiplet_list(nl_arr):
def print_multiplet_list(nl_arr):
    print(f"Multiplet list:")
    num_multiplets = len(nl_arr)
    maxlen = 10
    numprints = num_multiplets//maxlen + 1
    for i in range(numprints):
        sidx = i*numprints
        eidx = (i+1)*numprints
        print(f"{nl_arr[sidx:eidx]}")
    return None
# }}} print_multiplet_list(nl_arr)
#-----------------------------------------------------------------------
nmin, nmax = ARGS.nmin, ARGS.nmax
lmin, lmax = ARGS.lmin, ARGS.lmax
GVARS = gvar_jax.GlobalVars()
outdir = f"{GVARS.scratch_dir}/{ARGS.outdir}"
obsdata = GVARS.hmidata_in # only use of GVARS
mask_qdpy = get_exclude_mask(ARGS.exclude_qdpy)

# masking the observed data to remove qdpy
obsdata = obsdata[mask_qdpy, :]
obs_ell = obsdata[:, 0]
obs_enn = obsdata[:, 1]

nl_arr, omega_arr = get_multiplet_list(ARGS.exclude_qdpy)

#---------------------- printing and saving ---------------------------
print_multiplet_list(tuple(map(tuple, nl_arr)))
print(f'Total multiplets: {len(nl_arr)}')
np.save(f'{outdir}/qdpy_multiplets.npy', nl_arr)
np.save(f'{outdir}/omega_qdpy_multiplets.npy', omega_arr)