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


@cython.boundscheck(False)
cpdef double frobenius_norm(double[:, :, :] tensor, int N):
    """Computed the frobenius norm of a tensor or matrix

    Parameters
    ----------
    tensor : tl.tensor or ndarray
    """
    cdef int i, j, z
    cdef double frob_norm = 0
    for i in prange(N, nogil=True):
        for j in range(N):
            for z in range(N):
                frob_norm += tensor[i, j, z] * \
                    tensor[i, j, z]  # uses openmp reduce +
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
@cython.boundscheck(False)
cpdef int find_ranks(double[:, :] sigma, double error):
    cdef double sq_err_per_rank = error * error
    # start with a relative error higher than target to make the loop start
    cdef double est_error = 0
    cdef int i = 0
    while (i == 0 or est_error < sq_err_per_rank):
        for j in range(3):
            est_error += sigma[j, i] * sigma[j, i]
        i += 1
    return i


def compute_core(tensor, ranks=None, max_rel_error=1e-4):
    """Computed the core tensor of a given tensor and all orthogonal matrices s.t. A=Sx_1U_1.Tx_2U_2.Tx_3U_3.T
       If ranks are given, the tucker decomposition with the given ranks is calculated.
       Otherwise the maximum relative error is taken as measurement

    Parameters
    ----------
    tensor : tl.tensor or ndarray
    """
    if len(tensor.shape) != 3:
        raise ValueError("Please enter a 3 - tensor")
    U, sig = calculate_SVD(tensor)
    if ranks is None:
        max_error = max_rel_error * frobenius_norm(tensor, tensor.shape[0])
        opt_rank = tensor.shape[0] - \
            find_ranks(np.stack(sig)[:, ::-1], max_error) + 1
        print opt_rank
        ranks = np.repeat(opt_rank, 3)
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
    """Computed the HOSVD of a tensor. For this the 3 tensor is unfolded to 3 matrices
    and the regular SVD of these is calculated.

    Parameters
    ----------
    tensor : tl.tensor or ndarray
    """
    if len(tensor.shape) != 3:
        raise ValueError("Please enter a 3 - tensor")
    U = []
    SIGMA = []
    for (i, size) in enumerate(tensor.shape):
        U_tmp, SIGMA_tmo, V = np.linalg.svd(tl.unfold(
            tensor, mode=i), full_matrices=False)
        U.append(U_tmp)
        SIGMA.append(SIGMA_tmo)
    return(U, SIGMA)


def reconstruct_tensor(U, core, origin):
    """ Reconstruct tensor using the tucker decomposition.
    Print the absolute and relative error. Return the reconstructed tensor
    """

    # Calculate number of ranks needed to achieve demanded accuracy
    recon_tensor = core
    for i in range(len(core.shape)):
        recon_tensor = mode_n_multiplication(recon_tensor, U[i], mode=i)
    delta = recon_tensor - origin
    error = frobenius_norm(delta, delta.shape[0])
    print(f"The absolute error between the reconstructed and the real tensor is {error}. The relative error is: {error/frobenius_norm(origin, origin.shape[0])}")
    return recon_tensor
