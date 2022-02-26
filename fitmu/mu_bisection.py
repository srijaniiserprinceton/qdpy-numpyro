import os

mu_limits = [1e-8, 1e-3]
os.system(f"python ../dpy_jax/run_reduced_problem_newton.py --mu {muval}")
