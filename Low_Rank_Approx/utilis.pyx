import numpy as np
import tensorly as tl
cimport cython

from cython.parallel cimport prange  # Parallel range
from libc.math cimport sin

cdef extern from "math.h" nogil:
    double sqrt(double m)


@cython.cdivision(True)  # Modulo is checking for 0 div, no need
@cython.boundscheck(False)
cdef double c_tsi(double x, int N) nogil:
    cdef double val = x / (N + 1)
    return(val)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:, :, :] get_B_one(int N):
    cdef int i, j, z
    cdef double[:, :, :] B1 = np.zeros((N, N, N), dtype=np.float)
    for i in prange(N, nogil=True):
        for j in range(N):
            for z in range(N):
                B1[i, j, z] = sin(
                    c_tsi(i, N) + c_tsi(j, N) + c_tsi(z, N))
    return B1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double[:, :, :] get_B_two(int N):
    cdef int i, j, z
    cdef double[:, :, :] B2 = np.zeros((N, N, N), dtype=np.float)

    for i in prange(N, nogil=True):
        for j in range(N):
            for z in range(N):
                B2[i, j, z] = sqrt(
                    c_tsi(i, N)**2 + c_tsi(j, N)**2 + c_tsi(z, N)**2)
    return B2

cpdef double frobenius_norm(double[:, :, :] tensor):
    """Computed the frobenius norm of a tensor or matrix

    Parameters
    ----------
    tensor : tl.tensor or ndarray
    """
    cdef int i, j, z
    cdef double frob_norm = 0

    for i in range(N):  # noloop, as double is not atomic
        for j in range(N):
            for z in range(N):
                frob_norm = frob_norm + tensor[i, j, z] * tensor[i, j, z]
    frob_norm = sqrt(frob_norm)

    return(frob_norm)


def mode_n_multiplication(tensor, matrix, mode=0):
    """Computed the mode product between a matrix and a tensor

    Parameters
    ----------
    tensor : tl.tensor or ndarray with ndim=3
    matrix : ndarray
    mode : int
    """

    if matrix.shape[1] != tensor.shape[mode]:
        raise ValueError("Dimensions for mode multiplication were wrong! Tensor: {0}, Matrix: {1}".format(
            str(tensor.shape[mode]), str(matrix.shape[1])))
    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[0]
    out = np.dot(matrix, tl.unfold(tensor, mode))
    return tl.fold(out, mode, new_shape)


#######################
#    HOSVD Utilis     #
#######################


def compute_core(tensor, ranks=None, max_rel_error=1e-4:
    if len(tensor.shape) != 3:
        raise ValueError("Please enter a 3 - tensor")
    U, sig=calculate_SVD(tensor)
    if ranks is None:
        # Compute the ranks that result in a bounded error as required
        max_error = max_rel_error * frobenius_norm(tensor)
        max_error_square = max_error ** 2
        # Vectorized computation using some numpy functionality
        sng_vals = np.stack(Sigma_list)
        sng_vals_cumsum = np.cumsum(sng_vals[:, ::-1], axis=1)
        ranks = np.sum(
            (np.cumsum(sng_vals[:, ::-1], axis=1) > max_error_square / 3),
            axis=1)
        print(f'Resulting ranks: {ranks}')

    else:
        pass
    for i, rank in enumerate(ranks):
        U[i] = U[i][:, 0:int(rank)]
    Core = tensor
    for i in range(3):
        Core = mode_n_multiplication(Core, U[i].T, mode=i)
    return Core, U


def calculate_SVD(tensor):
    """Computed the core tensor of a given tensor and all orthogonal matrices s.t. A=Sx_1U_1.Tx_2U_2.Tx_3U_3.T
       If ranks are given, the tucker decomposition with the given ranks is calculated

    Parameters
    ----------
    tensor : tl.tensor or ndarray
    """
    # Calculate SVD for all mode matricitations and also save economic
    if len(tensor.shape) != 3:
        raise ValueError("Please enter a 3 - tensor")
    U = []
    SIGMA = []
    for (i, size) in enumerate(tensor.shape):
        U_tmp, SIGMA_tmo, V=np.linalg.svd(tl.unfold(
            tensor, mode=i), full_matrices=False)
        U.append(U_tmp)
        SIGMA.append(SIGMA_tmo)
    return(U, SIGMA)


def reconstruct_tensor(U, core, origin=None):
    """ Reconstruct tensor using the tucker decomposition.
    Print the absolute and relative error. Return the reconstructed tensor
    """

    # Calculate number of ranks needed to achieve demanded accuracy
    recon_tensor = core
    for i in range(len(core.shape)):
        recon_tensor = mode_n_multiplication(recon_tensor, U[i], mode=i)
    if not origin:
        delta = recon_tensor - origin
        error = frobenius_norm(delta)
        print(f"The absolute error between the reconstructed and the real tensor is {error}. The relative error ist: {error/frobenius_norm(origin)}")
    return recon_tensor
