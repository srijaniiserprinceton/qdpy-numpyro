import os
import time

current_dir = os.path.dirname(os.path.realpath(__file__))
package_dir = os.path.dirname(current_dir)
jax_dir = f"{package_dir}/qdpy_jax"
numpy_dir = f"{package_dir}/qdpy_numpy"

t1 = time.time()
os.system(f"python {jax_dir}/class_Cvec.py")
t2 = time.time()
print("===========================================")
print("===========================================")
print("===========================================")
t1n = time.time()
os.system(f"python {numpy_dir}/class_Cvec.py")
t2n = time.time()

print(f"Time taken [numpy] = {t2n-t1n:.3e} seconds")
print(f"Time taken [ jit ] = {t2-t1:.3e} seconds")
print(f"Speedup = {(t2n-t1n)/(t2-t1):.3f}x")
