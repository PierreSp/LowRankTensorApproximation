# distutils: language = c++
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
cpdef double frobenius_norm_tensor(double[:, :, :] tensor):
    """Computed the frobenius norm of a tensor
    """
    cdef int i, j, z, I_dim, J_dim, Z_dim
    I_dim = tensor.shape[0]
    J_dim = tensor.shape[1]
    Z_dim = tensor.shape[2]
    cdef double frob_norm = 0
    for i in prange(I_dim, nogil=True):
        for j in range(J_dim):
            for z in range(Z_dim):
                frob_norm += tensor[i, j, z] * tensor[i, j, z]
    frob_norm = sqrt(frob_norm)
    return(frob_norm)


@cython.boundscheck(False)
cpdef double frobenius_norm_mat(double[:, :] o_matrix):
    """Computed the frobenius norm of a matrix
    """

    cdef int i, j, I_dim, J_dim
    I_dim = o_matrix.shape[0]
    J_dim = o_matrix.shape[1]
    cdef double frob_norm = 0
    for i in prange(I_dim, nogil=True):
        for j in range(J_dim):
            frob_norm += o_matrix[i, j] * o_matrix[i, j]
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
        max_error = max_rel_error * frobenius_norm_tensor(tensor)
        opt_rank = tensor.shape[0] - \
            find_ranks(np.stack(sig)[:, ::-1], max_error) + 1
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
    error = frobenius_norm_tensor(delta)
    rel_error = error / np.max([10e-16, frobenius_norm_tensor(origin)])
    print(f"The absolute error between the reconstructed and the real tensor is {error}. The relative error is: {rel_error}")
    return recon_tensor

#######################
#      ACA Utilis     #
#######################
from libc.math cimport fabs


@cython.cdivision(True)
@cython.boundscheck(False)
cpdef aca_full_pivoting(o_matrix, double max_error):
    """ACA with full pivoting
    Computes CUR decomposition from matrix and my error
    """
    cdef int i, j
    cdef double delta
    cdef double[:] u, v
    Rk = o_matrix.copy()  # set R_0
    i_set = []
    j_set = []

    while frobenius_norm_mat(Rk) > max_error * frobenius_norm_mat(o_matrix):
        i, j = np.unravel_index(
            np.argmax(np.abs(np.asarray(Rk).ravel())), Rk.shape)
        i_set.append(i)
        j_set.append(j)
        delta = Rk[i, j]
        u = Rk[:, j]
        v = np.divide(Rk[i, :].T, delta)
        Rk = Rk - np.outer(u, v)
    R = o_matrix[i_set, :]
    U = np.linalg.inv(o_matrix[i_set, :][:, j_set])
    C = o_matrix[:, j_set]
    print(i_set)
    print(j_set)

    return C, U, R


# cdef class Function:
#     cpdef double evaluate(self, int i, int j, int z, int N) except *:
#         return 0

# Functions to calculate value of B1, B2 (for partial pivoting)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double b1(int i, int j, int z, int N):
    cdef double result
    result = sin(c_tsi(i, N) + c_tsi(j, N) + c_tsi(z, N))
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double b2(int i, int j, int z, int N):
    cdef double result
    result = sqrt(c_tsi(i, N)**2 + c_tsi(j, N)**2 + c_tsi(z, N)**2)
    return result

# cdef class b1(Function):
#     # These wrappers are needed to make sure, that the functions are evaluated
#     # in C

#     cpdef double evaluate(self, int i, int j, int z, int N):
#         cdef double result
#         result = sin(c_tsi(i, N) + c_tsi(j, N) + c_tsi(z, N))
#         return result


# cdef class b2(Function):
#     cpdef double evaluate(self, int i, int j, int z, int N):
#         cdef double result
#         result = sqrt(c_tsi(i, N)**2 + c_tsi(j, N)**2 + c_tsi(z, N)**2)
#         return result


def mode_m_matricization_fun(f,  d1,  d2,  d3,  m):
    def matricized_f(i, j, N):
        if m == 0:
            i1, i2, i3 = i, j % d2, j // d2
        elif m == 1:
            i1, i2, i3 = j % d1, i, j // d1
        elif m == 2:
            i1, i2, i3 = j % d2, j // d2, i
        return f(i1, i2, i3, N)
    return matricized_f


def closure_fk(fk,  u, v):
    # Closure for the function. Update f without knowing it
    def fk_kk(i, j, N):
        return fk(i, j, N) - u[i] * v[j]
    return fk_kk

cdef double mem_view_dot(double[:] vec, int N):
    # own c function for a dot product
    cdef double result = 0
    for i in range(N):
        result += vec[i] * vec[i]
    return result

cpdef aca_partial_pivoting(f, int m, int n, int N, double max_error):
    """ACA with partial pivoting
    Computes CUR decomposition from matrix and my error

    Parameters
    ----------
    f : functional matrizied tensor
    n : matrix shape[0]
    m : inmatrix shape[1]
    N : tensor dimenson
    max error : max "relative" error
    """

    cdef int i, j, k, elem_counter, size_i, size_j
    cdef double delta, est_norm_Rk
    cdef double[:] u, v
    # cdef set[int] i_used = set[int](n/10)
    # cdef set[int] j_used = set[int](m/10)
    i_used = []
    j_used = []
    i_not_used = [x for x in range(m)]
    j_not_used = [x for x in range(n)]
    size_i = len(i_not_used)
    size_j = len(j_not_used)
    u_list = []
    v_list = []

    elem_counter = 1
    i_counter = 0
    est_norm_Rk = 0
    fk = f
    i = max_value_row(fk, np.array(i_not_used, dtype=np.int), 0, N, size_i)

    while elem_counter == 1 or mem_view_dot(u, m) * mem_view_dot(v, n) > max_error * max_error * est_norm_Rk:
        j = max_value_col(fk, i, np.array(j_not_used, dtype=np.int), N, size_j)
        delta = fk(i, j, N)
        if np.isclose(delta, 0):
            if i_counter == np.min((n, m)) - 1:
                break
        else:
            u = get_u(fk, m, j, N)
            v = get_v(fk, i, n, delta, N)
            v_list.append(v)
            u_list.append(u)
            fk = closure_fk(fk, u, v)
            elem_counter += 1
            est_norm_Rk += (mem_view_dot(u, m) * mem_view_dot(v, n) +
                            np.sum([np.asarray(u).T.dot(np.asarray(u_list[l])) * (np.asarray(v_list[l]).T).dot(np.asarray(v))
                                    for l in range(0, elem_counter - 1)]))
            i_used.append(i)
            j_used.append(j)
        i_counter += 1
        i_not_used.remove(i)
        size_i -= 1
        j_not_used.remove(j)
        size_j -= 1

        i = max_value_row(fk, np.array(i_not_used, dtype=np.int), j, N, size_i)
    print("done")
    print(i_used)
    print(j_used)
    R = np.array([[f(i, j, N) for j in range(n)] for i in i_used])
    U = np.linalg.inv(np.array([[f(i, j, N) for j in j_used] for i in i_used]))
    C = np.array([[f(i, j, N) for j in j_used] for i in range(m)])
    return C, U, R


cpdef double[:] get_u(f, int m, int j, int N):
    # returns u vector from b1 or b2
    cdef double[:] u = np.zeros(m)
    cdef int i
    for i in range(m):
        u[i] = f(i, j, N)
    return u

cpdef double[:] get_v(f, int i, int m, double delta, int N):
    # returns v vector from b1 or b2
    cdef double[:] v = np.zeros(m)
    cdef int j
    for j in range(m):
        v[j] = f(i, j, N) / delta
    return v


cpdef int max_value_col(f, int i, long[:] columns, int N, int size_col):
    # Calculate maximum value in a row from b1 or b2
    cdef int bestcol, j
    cdef double currval
    cdef double bestval = -1
    bestcol = -1
    for j in range(size_col):
        currval = fabs(f(i, columns[j], N))
        if bestval < currval:
            bestcol = j
    return bestcol


cpdef int max_value_row(f, long[:] rows, int j, int N, int size_row):
    # Calculate maximum value in a row from b1 or b2
    cdef int bestrow, i
    cdef double currval
    cdef double bestval = -1
    bestrow = -1
    for i in range(size_row):
        currval = fabs(f(rows[i], j, N))
        if bestval < currval:
            bestrow = i
    return bestrow
