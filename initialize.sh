#!/bin/sh
python dpy_jax/mode_lister.py --nmin 0 --nmax 30 --lmin 5 --lmax 295
python dpy_jax/precompute_ritzlavely.py
python dpy_jax/generate_synthetic_eigvals.py --load_mults 1 --knot_num 45 --rth 0.4
python dpy_jax/save_reduced_problem.py
