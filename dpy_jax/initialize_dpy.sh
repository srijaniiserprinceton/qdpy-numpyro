#!/bin/sh
NMIN_DEFAULT=0
NMAX_DEFAULT=30
LMIN_DEFAULT=5
LMAX_DEFAULT=295
SMIN_DEFAULT=1
SMAX_DEFAULT=7
KNOTNUM_DEFAULT=15
RTH_DEFAULT=0.9
EXCLUDE_QDPY_MODES_DEFAULT=0
INSTR_DEFAULT="hmi"
TSLEN_DEFAULT=72
DAYNUM_DEFAULT=6328
NUMSPLIT_DEFAULT=18

echo "----------------------------------------"
echo "         Model attributes"
echo "----------------------------------------"
read -p "nmin (default=$NMIN_DEFAULT)= " NMIN
read -p "nmax (default=$NMAX_DEFAULT)= " NMAX
read -p "lmin (default=$LMIN_DEFAULT)= " LMIN
read -p "lmax (default=$LMAX_DEFAULT)= " LMAX
read -p "smin (default=$SMIN_DEFAULT)= " SMIN
read -p "smax (default=$SMAX_DEFAULT)= " SMAX
read -p "knot_num (default=$KNOTNUM_DEFAULT)= " KNOTNUM
read -p "rth (default=$RTH_DEFAULT)= " RTH
read -p "exclude_qdpy (default=$EXCLUDE_QDPY_MODES_DEFAULT)= " EXCLUDE_QDPY_MODES
echo "----------------------------------------"
echo "        Data attributes"
echo "----------------------------------------"
read -p "instrument (default=$INSTR_DEFAULT)= " INSTR
read -p "tslen (default=$TSLEN_DEFAULT)= " TSLEN
read -p "daynum (default=$DAYNUM_DEFAULT)= " DAYNUM
read -p "numsplit (default=$NUMSPLIT_DEFAULT)= " NUMSPLIT

# Setting default values if empty
NMIN="${NMIN:-$NMIN_DEFAULT}"
NMAX="${NMAX:-$NMAX_DEFAULT}"
LMIN="${LMIN:-$LMIN_DEFAULT}"
LMAX="${LMAX:-$LMAX_DEFAULT}"
SMIN="${SMIN:-$SMIN_DEFAULT}"
SMAX="${SMAX:-$SMAX_DEFAULT}"
KNOTNUM="${KNOTNUM:-$KNOTNUM_DEFAULT}"
RTH="${RTH:-$RTH_DEFAULT}"
EXCLUDE_QDPY_MODES="${EXCLUDE_QDPY_MODES:-$EXCLUDE_QDPY_MODES_DEFAULT}"
INSTR="${INSTR:-$INSTR_DEFAULT}"
TSLEN="${TSLEN:-$TSLEN_DEFAULT}"
DAYNUM="${DAYNUM:-$DAYNUM_DEFAULT}"
NUMSPLIT="${NUMSPLIT:-$NUMSPLIT_DEFAULT}"

echo "---model parameters ---"
echo "nmin      = $NMIN"
echo "nmax      = $NMAX"
echo "lmin      = $LMIN"
echo "lmax      = $LMAX"
echo "smin      = $SMIN"
echo "smax      = $SMAX"
echo "knot_num  = $KNOTNUM"
echo "rth       = $RTH"
echo "excl_qdpy = $EXCLUDE_QDPY_MODES"
echo "---data parameters ---"
echo "instrument = $INSTR"
echo "tslen      = $TSLEN"
echo "daynum     = $DAYNUM"
echo "numsplit   = $NUMSPLIT"
echo "-------------------------"

echo "[ 1. ] Creating list of modes ..."
if [ $EXCLUDE_QDPY_MODES == '1' ]; then
	echo "       -- Using only DPY modes"
	python ../qdpy/mode_lister.py --nmin $NMIN --nmax $NMAX --lmin $LMIN --lmax $LMAX \
		   --instrument $INSTR --tslen $TSLEN --daynum $DAYNUM --numsplits $NUMSPLIT \
		   --outdir "dpy_jax" --exclude_qdpy 1 >.mlist.out 2>.mlist.err
else
	echo "       -- Using QDPY+DPY modes"
	python ../qdpy/mode_lister.py --nmin $NMIN --nmax $NMAX --lmin $LMIN --lmax $LMAX \
		   --instrument $INSTR --tslen $TSLEN --daynum $DAYNUM --numsplits $NUMSPLIT \
		   --outdir "dpy_jax" >.mlist.out 2>.mlist.err
fi
echo "       -- `tail -1 .mlist.out`"
echo "[ 2. ] Generating synthetic eigenvalues ..."
python generate_synthetic_eigvals.py --lmin $LMIN --lmax $LMAX \
	   --load_mults 1 --knot_num $KNOTNUM --rth $RTH \
	   --instrument $INSTR --tslen $TSLEN --daynum $DAYNUM --numsplits $NUMSPLIT
echo "       -- DONE"

echo "[ 3. ] Saving reduced problem ..."
python save_reduced_problem.py --instrument $INSTR --smin $SMIN --smax $SMAX
echo "       -- DONE"

echo "[ 4. ] Creating Ritzwoller Lavely polynomials ..."
python ../qdpy/precompute_ritzlavely.py --outdir "dpy_jax" \
	   --instrument $INSTR >.rl.out 2>.rl.err
echo "       -- `tail -1 .rl.out`"
echo "----- INITIALIZATION COMPLETE --------------------"
