"""Class to handle Ritzwoller-Lavely polynomials"""
import logging
import numpy as np
import scipy.special as special
import numpy.polynomial.legendre as npleg

NAX = np.newaxis

class ritzLavelyPoly():
    """Computes the Ritzwoller-Lavely polynomials for a given 
    ell and jmax.
    """

    __all__ = ["get_Pjl",
               "normalize_Pjl",
               "get_coeffs",
               "polyval"]

    def __init__(self, ell, jmax):
        assert ell > 0, "Ritzwoller-Lavely polynomials don't exist for ell=0"
        assert jmax + 1 <= 2*ell, "Max degree (jmax) should be smaller than 2*ell"
        self.ell = ell
        self.jmax = jmax + 1
        self.m = np.arange(-ell, ell+1) * 1.0
        self.L = np.sqrt(ell*(ell+1))
        self.m_by_L = self.m/self.L
        self.Pjl = np.zeros((self.jmax, len(self.m)), dtype=np.float64)
        self.Pjl_exists = False

    def get_Pjl(self):
        if self.Pjl_exists:
            print('Ritzwoller-Lavely polynomials already computed')
            return self.Pjl
        else:
            self.Pjl[0, :] += self.ell
            self.Pjl[1, :] += self.m
            for j in range(2, self.jmax):
                coeffs = np.zeros(j+1)
                coeffs[-1] = 1.0
                P2j = self.L * npleg.legval(self.m_by_L, coeffs)
                cj = self.Pjl[:j, :] @ P2j / (self.Pjl[:j, :]**2).sum(axis=1)
                P1j = P2j - (cj[:, NAX] * self.Pjl[:j, :]).sum(axis=0)
                self.Pjl[j, :] += self.ell * P1j/P1j[-1]
            self.Pjl_exists = True
            # self.normalize_Pjl()
            return self.Pjl

    def normalize_Pjl(self):
        norms = np.zeros(self.Pjl.shape[0])
        for i in range(len(norms)):
            norms[i] = np.sqrt(self.Pjl[i, :] @ self.Pjl[i, :])
        self.Pjl = self.Pjl / norms[:, NAX]

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


def precompute_RLpoly(GVARS):
    ellmax = np.max(GVARS.ell0_arr)
    jmax = GVARS.smax

    # ell index for rlp (to avoid repeated computation and storage
    rlp_ell_map = np.zeros_like(GVARS.ell0_arr)
    
    # containing only the unique number of ells
    ell0_unique = np.sort(np.unique(ell0_arr))  # sorting for ease of search
    ell_count_arr = np.zeros((2, len(ell0_unique)))
    
    # filling first dimension wtih value of unique ell
    ell_count_arr[0, :] = ell0_unique
    # the second dimension is a count for how many times it
    # has been encountered. If > 1, not computed again.
    
    # jmax - 2 since we don't need to store j=0,1
    leg_poly = np.zeros((len(ell0_unique), jmax-2, 2*ellmax+1))

    for i, ell in enumerate(GVARS.ell0_arr):
        m = np.arange(-ell, ell+1) * 1.0
        L = np.sqrt(ell*(ell+1))
        m_by_L = m/L

        ell_ind = np.search_sorted(ell0_unique, ell)
        
        # not storing if already exists
        if(ell_count_arr[1, ell_ind] > 0):
            rlp_ell_map[i] = ell_ind
            
        # storing if doesn't exist already
        else:
            for j in range(2, jmax):
            coeffs = np.zeros(j+1)
            coeffs[-1] = 1.0
            
            leg_poly[ell_ind, j, :2*ell+1] = npleg.legval(m_by_L, coeffs)  
        
        # counting ell the present ell
        ell_count_arr[1, ell_ind] += 1
        
    return leg_poly

    
