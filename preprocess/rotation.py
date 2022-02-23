import os
import numpy as np
from astropy.io import fits
from math import sqrt, pi
import matplotlib.pyplot as plt
from scipy.integrate import simps
import scipy.interpolate as interp
from scipy.special import legendre as scipy_legendre
import argparse

NAX = np.newaxis
OM = 2.096367060263e-05
unit_conv = 1e-9/OM

#-----------------------------------------------------------------------#
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--instrument", help="hmi or mdi",
                    type=str, default="hmi")
PARSER.add_argument("--tslen", help="72d or 360d",
                    type=str, default="72d")
ARGS = PARSER.parse_args()
del PARSER
#-----------------------------------------------------------------------#
INSTR = ARGS.instrument
#------------------------ directory structure --------------------------#
current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
with open(f"{package_dir}/.config", "r") as f:
    dirnames = f.read().splitlines()
scratch_dir = dirnames[1]
ipdir = f"{scratch_dir}/input_files"
dldir = f"{ipdir}/{INSTR}"
#----------------------------------------------------------------------#

def writefitsfile(a, fname, overwrite=False):
    from astropy.io import fits
    hdu = fits.PrimaryHDU()
    hdu.data = a
    hdu.writeto(fname, overwrite=overwrite)
    print(f"Writing {fname}")


def get_both_hemispheres(w2d_upper):
    """Mirrors the rotation data from top hemisphere to the bottom hemisphere

    Inputs:
    -------
    w2d_upper - np.ndarray(ndim=2, dtype=float)
        Rotation profile in the top hemisphere

    Returns:
    --------
    w2d - np.ndarray(ndim=2, dtype=float)
        Rotation profile on the full sphere
    """
    lenr = w2d_upper.shape[0]
    lent = w2d_upper.shape[1]

    w2d_lower = w2d_upper[:, ::-1][:, 1:]
    w2d = np.zeros((lenr, w2d_upper.shape[1]*2-1))

    w2d[:, :lent] = w2d_upper
    w2d[:, lent:] = w2d_lower
    return w2d


def get_omegak(rot_profile, smax):
    costh = np.cos(theta)
    klist = np.arange(smax+2)[::2]
    lplist = []
    omegak = []
    for k in klist:
        lplist.append(scipy_legendre(k)(costh))
        omegak.append(np.zeros(lenr))

    for ik, k in enumerate(klist):
        for ir in range(lenr):
            omegak[ik][ir] = simps(rot_profile[ir, :]*lplist[ik], costh)*(2*k+1.)/2.
    return omegak


def get_ws(rot_profile, smax):
    omegak = get_omegak(rot_profile, smax)
    klist = np.arange(smax+2)[::2]
    slist = np.arange(smax)[::2] + 1
    ws = []
    for iess, s in enumerate(slist):
        prefac = -2*sqrt(pi/(2*s+1))
        omfac1 = 2*klist[iess] + 1
        omfac2 = omfac1 + 4
        ws.append(prefac*rmesh*(omegak[iess]/omfac1 - omegak[iess+1]/omfac2))
        # writefitsfile(ws[iess], f'{output_dir}/w{s}{fprefix}-{INSTR}.fits',
        #               overwrite=True)
    return ws


def get_interpolated_ws(rot_profile, smax):
    ws_hmi_list = get_ws(rot_profile, smax)
    ws_list = []
    for ws in ws_hmi_list:
        ws_int = interp.interp1d(rmesh, ws, fill_value="extrapolate")
        ws_global = ws_int(r_global)
        ws_list.append(ws_global*unit_conv)
    return np.vstack(ws_list)


def load_data(fname_re):
    """Reads hemispherical rotation data and returns full rotation profiles"""

    # Reading radial-mesh, rotation profile and error
    rmesh = np.loadtxt(f'{dldir}/rmesh.{INSTR}')[::4]
    rot2d = np.loadtxt(f'{fname_re[0]}')
    err2d = np.loadtxt(f'{fname_re[1]}')
    lenr = len(rmesh)

    # converting hemispherical theta-mesh to full spherical mesh
    tmesh = np.arange(rot2d.shape[1])
    theta = 90 - tmesh*15./8
    theta = np.append(-theta, theta[::-1][1:]) + 90.
    theta = theta*pi/180.

    tharr = np.arange(0, 90, 15)
    err_list = []
    for th in tharr:
        absdiff = abs(th*np.pi/180 - theta)
        thidx = np.argmin(abs(th*np.pi/180 - theta))
        if absdiff[thidx] == 0:
            err_list.append(err2d[:, thidx])
        else:
            err_plus = err2d[:, thidx+1]
            dth_plus = abs(theta[thidx+1] - th*np.pi/180)
            err_minus = err2d[:, thidx-1]
            dth_minus = abs(theta[thidx-1] - th*np.pi/180)
            dth = abs(theta[thidx+1] - theta[thidx-1])
            err_list.append((err_plus*dth_plus +
                             err_minus*dth_minus)/dth)

    rot2d = get_both_hemispheres(rot2d)
    err2d = get_both_hemispheres(err2d)
    err2d = err2d**2 #the linear operator acts on variance
    # writefitsfile(rmesh, f'{output_dir}/rad-{INSTR}.fits', overwrite=True)
    # writefitsfile(rot2d, f'{output_dir}/rot2dfull-{INSTR}.fits', overwrite=True)
    # np.save(f"{output_dir}/err1d-{INSTR}.npy", np.array(err_list))
    return (rmesh, theta), (rot2d, err2d)

def get_fnames(suffix="rot"):
    os.system(f"ls {dldir}/hmi* | grep {suffix} > {dldir}/fnames_{suffix}.txt")
    with open(f"{dldir}/fnames_{suffix}.txt", "r") as f:
        fnames = f.read().splitlines()
    return fnames


if __name__=="__main__":
    r_global = np.loadtxt(f"{ipdir}/r_jesper.dat")
    fnames_rot2d = get_fnames("rot")
    fnames_err2d = get_fnames("err")
    smax = 5

    for i in range(len(fnames_rot2d)):
        fname_re = [fnames_rot2d[i],
                    fnames_err2d[i]]
        print(fname_re[0])
        (rmesh, theta), (rot2d, err2d) = load_data(fname_re)
        lenr = len(rmesh)
        ws = get_interpolated_ws(rot2d, smax)
        es = get_interpolated_ws(err2d, smax)
        wsig = []

        fname_splits = fnames_rot2d[i].split('.')
        mdi_day = fname_splits[2]
        numsplits = fname_splits[3]

        np.savetxt(f'{dldir}/wsr.{INSTR}.{ARGS.tslen}.{mdi_day}.{numsplits}', ws)
        np.savetxt(f'{dldir}/err.{INSTR}.{ARGS.tslen}.{mdi_day}.{numsplits}', es)

