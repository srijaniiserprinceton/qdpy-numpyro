import jax
import jax.numpy as np
import numpy as onp
import numpy.random as npr
import timeit

from functools import partial


def mesh(*control):
    ''' Generate a control point mesh.
    Parameters:
    -----------
    - *control: A variable number of 1d array objects.  These are turned into a
                control mesh of that many dimensions.  So to create a 2D mesh,
                you could give it a sequence of length J and a sequence of length
                K; it will return an ndarray that is J x K x 2. If you want to
                create a 3D mesh, you could give it three sequences of lengths
                J, K, and M, respectively, and you'd get back an ndarray of size
                J x K x M x 3.  The last dimension will always correspond to the
                number of sequences provided.
    Returns:
    --------
     Returns an ndarray object with a mesh of control points.
    Examples:
    ---------
    >> mesh(np.arange(3), np.arange(4))
        DeviceArray([[[0, 0],
                      [0, 1],
                      [0, 2],
                      [0, 3]],
                     [[1, 0],
                      [1, 1],
                      [1, 2],
                      [1, 3]],
                     [[2, 0],
                      [2, 1],
                      [2, 2],
                      [2, 3]]], dtype=int32)
    >> mesh(np.arange(3), np.arange(4), np.arange(5)).shape
        (3, 4, 5, 3)
    '''
    return np.stack(np.meshgrid(*control, indexing='ij'), axis=-1)


def divide00(num, denom):
    ''' Divide such that 0/0 = 0.
    The trick here is to do this in such a way that reverse-mode and forward-mode
    automatic differentation via JAX still work reasonably.
    '''

    force_zero = np.logical_and(num == 0, denom == 0)
    return np.where(force_zero, np.float32(0.0), num) \
        / np.where(force_zero, np.float32(1.0), denom)


def default_knots(degree, num_ctrl):
    return np.hstack([np.zeros(degree),
                      np.linspace(0, 1, num_ctrl - degree + 1),
                      np.ones(degree)])


def bspline1d_basis(u, knots, degree):
    ''' Computes b-spline basis functions in one dimension.
    Parameters:
    -----------
     - u: The locations at which to evaluate the basis functions, generally
          assumed to be in the interval [0,1) although they really just need to
          be consistent with the knot ranges. Can be an ndarray of any size.
     - knots: The knot vector. Should be non-decreasing and consistent with the
              specified degree.  A one-dimensional ndarray.
     - degree: The degree of the piecewise polynomials. Integer.
    Returns:
    --------
     Returns an ndarray whose first dimensions are the same as u, but with an
     additional dimension determined by the number of basis functions, i.e.,
     one less than the number of knots minus the degree.
    '''
    u1d = np.atleast_1d(u)

    # Set things up for broadcasting.
    # Append a singleton dimension onto the u points.
    # Prepend the correct number of singleton dimensions onto knots.
    # The vars are named 2d but they could be bigger.
    u2d = np.expand_dims(u1d, -1)
    k2d = np.expand_dims(knots, tuple(range(len(u1d.shape))))

    # Handle degree=0 case first.
    # Modify knots so that when u=1.0 we get 1.0 rather than 0.0.
    k2d = np.where(k2d == knots[-1], knots[-1]+np.finfo(u2d.dtype).eps, k2d)

    # The degree zero case is just the indicator function on the
    # half-open interval specified by the knots.
    B = (k2d[..., :-1] <= u2d) * (u2d < k2d[..., 1:]) + 0.0

    # We expect degree to be small, so unrolling is tolerable.
    for deg in range(1, degree+1):

        # There are two halves.  The indexing is a little tricky.
        # Also we're using the np.divide 'where' argument to deal
        # with the fact that we want 0/0 to equal zero.
        # This all computes much more than we strictly need, because
        # so much of this is just multiplying zero by things.
        # However, I think the vectorized implementation is worth
        # it for using things like JAX and GPUs.
        # FIXME: Pretty sure I could do the denominator with one subtract.
        v0_num = B[..., :-1] * (u2d - k2d[..., :-deg-1])
        v0_denom = k2d[..., deg:-1] - k2d[..., :-deg-1]
        v0 = divide00(v0_num, v0_denom)

        v1_num = B[..., 1:] * (k2d[..., deg+1:] - u2d)
        v1_denom = k2d[..., deg+1:] - k2d[..., 1:-deg]
        v1 = divide00(v1_num, v1_denom)

        B = v0 + v1

    return B

@partial(jax.jit, static_argnums=(3,))
def bspline1d(u, control, knots, degree):
    ''' Evaluate a one-dimensional bspline function.
     - u: The locations at which to evaluate the basis functions, generally
          assumed to be in the interval [0,1) although they really just need to
          be consistent with the knot ranges. Can be an ndarray of any size.
     - control: The control points.  The first dimension should have the same
                size as the number of unique knots.
     - knots: The knot vector. Should be non-decreasing and consistent with the
              specified degree.  A one-dimensional ndarray.
     - degree: The degree of the piecewise polynomials. Integer.
    Returns:
    --------
     Returns an ndarray whose first dimension are the same as the first dimension
     of u, but with the second dimension being the same as the second dimension
     of the control point ndarray.
    '''

    return bspline1d_basis(u, knots, degree) @ control
