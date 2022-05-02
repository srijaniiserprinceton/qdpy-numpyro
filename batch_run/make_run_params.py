from collections import namedtuple

#------------------------DEFAULT VALUES------------------------------#
NMIN_DEFAULT=0
NMAX_DEFAULT=30
LMIN_DEFAULT=5
LMAX_DEFAULT=295
SMIN_DEFAULT=1
SMAX_DEFAULT=7
SMAX_GLOBAL_DEFAULT=7
KNOTNUM_DEFAULT=15
RTH_DEFAULT=0.9
EXCLUDE_QDPY_MODES_DEFAULT=0
INSTR_DEFAULT="hmi"
TSLEN_DEFAULT=72
DAYNUM_DEFAULT=6328
NUMSPLIT_DEFAULT=36
#--------------------------------------------------------------------#

def make_run_params(nmin=NMIN_DEFAULT, nmax=NMAX_DEFAULT, lmin=LMIN_DEFAULT,
                    lmax=LMAX_DEFAULT, smin=SMIN_DEFAULT, smax=SMAX_DEFAULT,
                    smax_global=SMAX_GLOBAL_DEFAULT,
                    knotnum=KNOTNUM_DEFAULT, rth=RTH_DEFAULT,
                    instr=INSTR_DEFAULT, tslen=TSLEN_DEFAULT,
                    daynum=DAYNUM_DEFAULT, numsplit=NUMSPLIT_DEFAULT,
                    exclude_qdpy=EXCLUDE_QDPY_MODES_DEFAULT):
    RUN_PARAMS_ = namedtuple('run_params', ['nmin', 'nmax', 'lmin',
                                            'lmax', 'smin', 'smax',
                                            'knotnum', 'rth', 'instr',
                                            'tslen', 'daynum', 'numsplit',
                                            'exclude_qdpy', 'smax_global'])
    RUN_PARAMS = RUN_PARAMS_(nmin, nmax, lmin, lmax, smin, smax, knotnum,
                             rth, instr, tslen, daynum, numsplit, exclude_qdpy,
                             smax_global)
    
    return RUN_PARAMS
