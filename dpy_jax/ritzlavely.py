B"""Class to handle Ritzwoller-Lavely polynomials"""
import logging
import numpy as np
import scipy.special as special
import numpy.polynomial.legendre as npleg

from dpy_jax import globalvars as gvar_jax

NAX = np.newaxis

class ritzLavelyPoly():
    """Computes the Ritzwoller-Lavely polynomials for a given 
    ell and jmax.
    """

    __all__ = ["get_Pjl",
               "normalize_Pjl",
               "get_coeffs",
               "polyval"]

    def __init__(self, GVARS):
        self.jmax = GVARS.smax + 1
        self.ell0_arr = GVARS.ell0_arr
        
        self.rlp_ell_map = None
        self.leg_poly = None
        self.RL_poly = None

        # generating the legendre polynomials
        self.precompute_legpoly()

        # generating the Pjl needed for get a-coeffs
        self.gen_RL_poly()

    def get_Pjl(self, ell, ell_i):

        assert ell > 0, "Ritzwoller-Lavely polynomials don't exist for ell=0"
        assert self.jmax + 1 <= 2*ell, "Max degree (jmax) should be smaller than 2*ell"
        
        m  = np.arange(-ell, ell+1)
        L = np.sqrt(ell * (ell+1))
        ell_ind = self.rlp_ell_map[ell_i] 
        m_by_L = m/L
        
        Pjl = np.zeros((self.jmax, len(m)), dtype=np.float64)
        Pjl[0, :] += ell
        Pjl[1, :] += m
        
        for j in range(2, self.jmax):
            coeffs = np.zeros(j+1)
            coeffs[-1] = 1.0
            P2j = L * self.leg_poly[ell_ind, j]
            print(Pjl.shape, P2j.shape)
            cj = Pjl[:j, :] @ P2j / (Pjl[:j, :]**2).sum(axis=1)
            P1j = P2j - (cj[:, NAX] * Pjl[:j, :]).sum(axis=0)
            Pjl[j, :] += ell * P1j/P1j[-1]
            
        return Pjl

    def get_coeffs(self, arrm):
        if not self.Pjl_exists:
            self.get_Pjl()
        assert len(arrm) == len(self.m), "Length of input array =/= 2*ell+1"
        aj = (self.Pjl @ arrm) / np.diag(self.Pjl @ self.Pjl.T)
        return aj

    def polyval(self, acoeffs):
        assert len(acoeffs) == self.jmax, (f"Number of coeffs ({len(acoeffs)} " +
                                           f"=/= jmax ({self.jmax})")
        return (acoeffs[:, NAX] * self.Pjl).sum(axis=0)

    def precompute_legpoly(self):
        ellmax = np.max(self.ell0_arr)
        
        # ell index for rlp (to avoid repeated computation and storage
        self.rlp_ell_map = np.zeros_like(self.ell0_arr)
        
        # containing only the unique number of ells
        ell0_unique = np.sort(np.unique(self.ell0_arr))  # sorting for ease of search
        ell_count_arr = np.zeros((2, len(ell0_unique)))
        
        # filling first dimension wtih value of unique ell
        ell_count_arr[0, :] = ell0_unique
        # the second dimension is a count for how many times it
        # has been encountered. If > 1, not computed again.
        
        # jmax - 2 since we don't need to store j=0,1
        self.leg_poly = np.zeros((len(ell0_unique), self.jmax, 2*ellmax+1))
        
        for i, ell in enumerate(self.ell0_arr):
            m = np.arange(-ell, ell+1) * 1.0
            L = np.sqrt(ell*(ell+1))
            m_by_L = m/L
            
            ell_ind = np.searchsorted(ell0_unique, ell)
            
            # not storing if already exists
            if(ell_count_arr[1, ell_ind] > 0):
                self.rlp_ell_map[i] = ell_ind
                
            # storing if doesn't exist already
            else:
                for j in range(2, self.jmax):
                    coeffs = np.zeros(j+1)
                    coeffs[-1] = 1.0
                    self.leg_poly[ell_ind, j, :2*ell+1] = npleg.legval(m_by_L, coeffs)  
                    
            # counting ell the present ell
            ell_count_arr[1, ell_ind] += 1


    def gen_RL_poly(self):
        ellmax = np.max(self.ell0_arr)
        
        self.RL_poly = np.zeros((len(self.ell0_arr),
                                 self.jmax, 2*ellmax+1),
                                dtype=np.float64)
            
        for ell_i, ell in enumerate(self.ell0_arr):
            self.RL_poly[ell_i, :, :2*ell+1] = self.get_Pjl(ell, ell_i)

    
if __name__ == '__main__':
    GVARS = gvar_jax.GlobalVars()
    make_rl_poly = ritzLavelyPoly(GVARS)
