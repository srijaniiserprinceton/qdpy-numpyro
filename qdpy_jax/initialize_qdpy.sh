#!/bin/sh
NMIN_DEFAULT=0
NMAX_DEFAULT=30
LMIN_DEFAULT=5
LMAX_DEFAULT=295
SMAX_DEFAULT=7
SMAX_GLOBAL_DEFAULT=7
KNOTNUM_DEFAULT=15
RTH_DEFAULT=0.9
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
read -p "smax (default=$SMAX_DEFAULT) = " SMAX
read -p "smax_global (default=$SMAX_GLOBAL_DEFAULT) = " SMAX_GLOBAL
read -p "knot_num (default=$KNOTNUM_DEFAULT)= " KNOTNUM
read -p "rth (default=$RTH_DEFAULT)= " RTH
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
SMAX="${SMAX:-$SMAX_DEFAULT}"
SMAX_GLOBAL="${SMAX_GLOBAL:-$SMAX_GLOBAL_DEFAULT}"
KNOTNUM="${KNOTNUM:-$KNOTNUM_DEFAULT}"
RTH="${RTH:-$RTH_DEFAULT}"
INSTR="${INSTR:-$INSTR_DEFAULT}"
TSLEN="${TSLEN:-$TSLEN_DEFAULT}"
DAYNUM="${DAYNUM:-$DAYNUM_DEFAULT}"
NUMSPLIT="${NUMSPLIT:-$NUMSPLIT_DEFAULT}"

echo "---model parameters ---"
echo "nmin      = $NMIN"
echo "nmax      = $NMAX"
echo "lmin      = $LMIN"
echo "lmax      = $LMAX"
echo "smax        = $SMAX"
echo "smax_global = $SMAX_GLOBAL"
echo "knot_num  = $KNOTNUM"
echo "rth       = $RTH"
echo "---data parameters ---"
echo "instrument = $INSTR"
echo "tslen      = $TSLEN"
echo "daynum     = $DAYNUM"
echo "numsplit   = $NUMSPLIT"
echo "-------------------------"

echo "[ 1. ] Creating list of modes ..."
python ../qdpy/mode_lister.py --nmin $NMIN --nmax $NMAX --lmin $LMIN --lmax $LMAX \
		--instrument $INSTR --tslen $TSLEN --daynum $DAYNUM --numsplits $NUMSPLIT \
		--outdir "qdpy_jax" --smax_global $SMAX_GLOBAL >.mlist.out 2>.mlist.err
echo "       -- `tail -1 .mlist.out`"

echo "[ 2. ] Saving reduced problem ..."
python save_reduced_problem.py --load_mults 1 --knot_num $KNOTNUM --rth $RTH \
	   --instrument $INSTR --tslen $TSLEN --daynum $DAYNUM --numsplits $NUMSPLIT \
           --smax $SMAX --smax_global $SMAX_GLOBAL
echo "       -- DONE"

echo "[ 3. ] Creating Ritzwoller Lavely polynomials ..."
python ../qdpy/precompute_ritzlavely.py --outdir "qdpy_jax" \
	   --instrument $INSTR >.rl.out 2>.rl.err
echo "       -- `tail -1 .rl.out`"

echo "----- INITIALIZATION COMPLETE --------------------"
