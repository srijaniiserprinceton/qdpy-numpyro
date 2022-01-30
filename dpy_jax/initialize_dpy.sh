#!/bin/sh
NMIN_DEFAULT=0
NMAX_DEFAULT=30
LMIN_DEFAULT=5
LMAX_DEFAULT=295
KNOTNUM_DEFAULT=15
RTH_DEFAULT=0.8
EXCLUDE_QDPY_MODES_DEFAULT=0

read -p "nmin (default=$NMIN_DEFAULT)= " NMIN
read -p "nmax (default=$NMAX_DEFAULT)= " NMAX
read -p "lmin (default=$LMIN_DEFAULT)= " LMIN
read -p "lmax (default=$LMAX_DEFAULT)= " LMAX
read -p "knot_num (default=$KNOTNUM_DEFAULT)= " KNOTNUM
read -p "rth (default=$RTH_DEFAULT)= " RTH
read -p "exclude_qdpy (default=$EXCLUDE_QDPY_MODES_DEFAULT)= " EXCLUDE_QDPY_MODES

# Setting default values if empty
NMIN="${NMIN:-$NMIN_DEFAULT}"
NMAX="${NMAX:-$NMAX_DEFAULT}"
LMIN="${LMIN:-$LMIN_DEFAULT}"
LMAX="${LMAX:-$LMAX_DEFAULT}"
KNOTNUM="${KNOTNUM:-$KNOTNUM_DEFAULT}"
RTH="${RTH:-$RTH_DEFAULT}"
EXCLUDE_QDPY_MODES="${EXCLUDE_QDPY_MODES:-$EXCLUDE_QDPY_MODES_DEFAULT}"

echo "---problem parameters ---"
echo "nmin      = $NMIN"
echo "nmax      = $NMAX"
echo "lmin      = $LMIN"
echo "lmax      = $LMAX"
echo "knot_num  = $KNOTNUM"
echo "rth       = $RTH"
echo "excl_qdpy = $EXCLUDE_QDPY_MODES"
echo "-------------------------"

echo "[ 1. ] Creating list of modes ..."
if [ $EXCLUDE_QDPY_MODES == '1' ]; then
	echo "       -- Using only DPY modes"
	python mode_lister.py --nmin $NMIN --nmax $NMAX --lmin $LMIN --lmax $LMAX \
		   --exclude_qdpy 1 >.mlist.out 2>.mlist.err
else
	echo "       -- Using QDPY+DPY modes"
	python mode_lister.py --nmin $NMIN --nmax $NMAX --lmin $LMIN --lmax $LMAX \
		   >.mlist.out 2>.mlist.err
fi
echo "       -- `tail -1 .mlist.out`"

echo "[ 2. ] Generating synthetic eigenvalues ..."
python generate_synthetic_eigvals.py --lmin $LMIN --lmax $LMAX \
	   --load_mults 1 --knot_num $KNOTNUM --rth $RTH 
echo "       -- DONE"

echo "[ 3. ] Creating Ritzwoller Lavely polynomials ..."
python precompute_ritzlavely.py >.rl.out 2>.rl.err
echo "       -- `tail -1 .rl.out`"

echo "[ 4. ] Saving reduced problem ..."
python save_reduced_problem.py 
echo "       -- DONE"
echo "----- INITIALIZATION COMPLETE --------------------"
