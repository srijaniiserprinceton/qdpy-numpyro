from jax import jit
import numpy as np

def f(a_tuple):
    a_arr = np.asarray(a_tuple)
    a_arr += 1
    a_tuple_ret = tuple(map(tuple,a_arr))
    print(a_tuple_ret)
    return a_tuple_ret

f_ = jit(f, static_argnums = (0,))

# the tuple
a_tuple = ((1,2),(3,4))

a_tuple_final = f_(a_tuple)

print(a_tuple_final)
