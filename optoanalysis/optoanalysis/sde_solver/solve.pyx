cimport numpy as np
cimport cython

@cython.boundscheck(False) # Turns off IndexError type warnings - e.g. a = [1, 2, 3]; a[5]
@cython.wraparound(False) # Turns off Python a[-1] indexing - will segfault.
@cython.overflowcheck(False) # Check that no integer overflows occur with arithmetic operations
@cython.initializedcheck(False) # Checks that memory is initialised when passed in
cpdef solve(np.ndarray[double, ndim=1] q,
            np.ndarray[double, ndim=1] v,
            double dt,
            np.ndarray[double, ndim=1] dwArray,
            double Gamma0,
            double Omega0,
            double eta,
            double b_v,
            int N ):
    """
    Solves the SDE from timeTuple[0] to timeTuple[1]
    
    Returns
    -------
    self.q : ndarray
        array of positions with time
    self.v : ndarray
        array of velocities with time
    """
    cdef int n
    for n in range(N): # had enumerate here - it took ~3.5 seconds!! now ~110ms
        v[n+1] = v[n] + (-(Gamma0 - Omega0*eta*q[n]**2)*v[n] - Omega0**2*q[n])*dt + b_v*dwArray[n]
        q[n+1] = q[n] + v[n]*dt
    return q, v


