import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.interpolate import interp1d
import numpy as np
import argparse
import os

# ------------------------ INPUT ARGUMENTS ------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument("--rth", help="radius threshold",
                    type=float, default=0.5)
parser.add_argument("--knot_num", help="number of knots",
                    type=int, default=20)
PARGS = parser.parse_args()

#------------------------ DIRECTORY HANDLING ----------------------------------#
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
ipdir = f"{scratch_dir}/input_files"
#---------------------------------------------------------------------------
def interpolate2global(var, r_local):
    varint = interp1d(r_local, var, fill_value="extrapolate")
    var_global = varint(r_global)
    return var_global


def get_traveltime(cg, rg, idx):
    rl = rg[:idx+1]
    cl = cg[:idx+1]
    cinv = 1./cl

    traveltime = simps(cinv, x=rl)
    return abs(traveltime)


def get_grid(cg, rg, rth, N):
    rth_idx = np.argmin(abs(rg - rth))
    traveltime = get_traveltime(cg, rg, rth_idx)

    rl = rg[:rth_idx+1]
    cl = cg[:rth_idx+1]
    cinv = 1./cl
    
    dt = traveltime/N
    grids = [0]

    tt = 0
    for i in range(rth_idx):
        dr = abs(rl[i+1] - rl[i])
        tt += cinv[i] * dr
        if tt < dt:
            continue
        else:
            grids.append(i)
            tt = 0
    return grids, rth_idx


r_global = np.loadtxt(f'{ipdir}/r.dat')[::-1]
model_s = np.loadtxt(f'{ipdir}/modelS.dat')
r = model_s[:, 0]
c = model_s[:, 1]

if __name__ == "__main__":
    c_global = interpolate2global(c, r)

    grids, rth_idx = get_grid(c_global, r_global, PARGS.rth, PARGS.knot_num)
    r_new = r_global[grids]
    dr_new = np.diff(r_global[grids])
    print(r_new)

    print("----------------------------------------------")
    num_points = rth_idx
    skip_len = num_points//PARGS.knot_num
    r_old = r_global[:rth_idx][::skip_len]
    c_old = c_global[:rth_idx][::skip_len]
    dr_old = np.diff(r_old)
    print(r_old)

    plt.figure()
    plt.plot(r, c, 'k')
    for i in range(len(grids)):
        plt.plot(r_global[grids[i]], c_global[grids[i]], '*r')
        plt.plot(r_old, c_old, '+b')
    plt.show()

    plt.figure()
    plt.plot(r_old[1:], abs(dr_old), '*k', label='Old method')
    plt.plot(r_new[1:], abs(dr_new), '+r', label='Based on traveltime')
    plt.xlabel('r')
    plt.ylabel('$\\delta r$')
    plt.title('Knot spacing')
    plt.legend()
    plt.show()
