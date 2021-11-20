import globalvars as gvar_jax
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nmin", help="radial order", type=int)
parser.add_argument("--nmax", help="radial order", type=int)
parser.add_argument("--lmin", help="radial order", type=int)
parser.add_argument("--lmax", help="radial order", type=int)

args = parser.parse_args()

nmin, nmax = args.nmin, args.nmax
lmin, lmax = args.lmin, args.lmax


gvar= gvar_jax.GlobalVars()

# that's all we use gvar for
data = gvar.hmidata

# starting with the desired shape. Will reject this value later
nl_arr = np.array([-1,-1], dtype=int)

# {{{ def findfreq(data, l, n, m):                                                                                                                                                                                
def findfreq(data, l, n, m):
    '''                                      `                                                                                                                                                                
    Find the eigenfrequency for a given (l, n, m)                                                                                                                                                             
    using the splitting coefficients                                                                                                                                                                          
    
    Inputs: (data, l, n, m)                                                                                                                                                                                   
        data - array (hmi.6328.36)                                                                                                                                                                        
        l - harmonic degree                                                                                                                                                                               
        n - radial order                                                                                                                                                                                  
        m - azimuthal order                                                                                                                                                                               
    
    Outputs: (nu_{nlm}, fwhm_{nl}, amp_{nl})                                                                                                                                                                  
        nu_{nlm}        - eigenfrequency in microHz                                                                                                                                                       
        fwhm_{nl} - FWHM of the mode in microHz                                                                                                                                                           
        amp_{nl}        - Mode amplitude (A_{nl})                                                                                                                                                         
    '''
    
    L = np.sqrt(l*(l+1))
    try:
        modeindex = np.where((data[:, 0]==l) * (data[:,1]==n))[0][0]
    except:
        print( "MODE NOT FOUND : l = %3s, n = %2s" %( l, n ) )
        return None, None, None
        
    (nu, amp, fwhm) = data[modeindex, 2:5]
    if m==0:
        return nu, fwhm, amp
    else:
        splits = np.append([0.0], data[modeindex, 12:48])
        splits[1] -= 31.7
        totsplit = legval(1.0*m/L, splits)*L*0.001
        return nu + totsplit, fwhm, amp
# }}} findfreq(data, l, n, m) 

for n in range(nmin, nmax+1):
    for l in range(lmin, lmax+1):
        a, b, c = findfreq(data, l, n, 0)
        if (a != None):
            nl_arr = np.vstack((nl_arr, np.array([n,l])))

# rejecting the first dummy entry
nl_arr = nl_arr[1:]

print(nl_arr)

print(f'Total multiplets: {len(nl_arr)}')


# saving the nl_arr
np.save('qdpy_multiplets.npy', nl_arr)
