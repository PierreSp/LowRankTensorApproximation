# distutils: language = c++
import numpy as np
cimport cython
from libc.math cimport fabs
from libc.math cimport sin
from cython.parallel cimport prange  # Parallel range
from utilis import *

#######################
#      ACA Utilis     #
#######################

cdef extern from "math.h" nogil:
    double sqrt(double m)


@cython.cdivision(True)  # Modulo is checking for 0 div, no need
@cython.boundscheck(False)
cdef double c_tsi(double x, int N) nogil:
    cdef double val = x / (N + 1)
    return(val)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double b1(int i, int j, int z, int N) nogil:
    cdef double result
    result = sin(c_tsi(i, N) + c_tsi(j, N) + c_tsi(z, N))
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double b2(int i, int j, int z, int N) nogil:
    cdef double result
    result = sqrt(c_tsi(i, N)**2 + c_tsi(j, N)**2 + c_tsi(z, N)**2)
    return result


def mode_m_matricization_fun(f,  d1,  d2,  d3):
    def matricized_f(i, j, N):
        return f(i, j % d2, j // d2, N)
    return matricized_f


def closure_fk(fk, u, v):
    # Closure for the function. Updates f
    def fk_kk(i, j, N):
        return fk(i, j, N) - u[i] * v[j]
    return fk_kk

cdef double mem_view_norm_sq(double[:] vec, int N):
    # own c function for a dot product
    cdef double result = 0
    for i in range(N):
        result += vec[i] * vec[i]
    return result


cdef double mem_view_scalar_prod(double[:] vec_one, double[:] vec_two, int N):
    # own c function for a dot product
    cdef double result = 0
    for i in range(N):
        result += vec_one[i] * vec_two[i]
    return result


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


cpdef int arg_max_col(f, int i, long[:] columns, int N, int size_col):
    # Calculate maximum value in a row from b1 or b2
    cdef int bestcol, j
    cdef double currval
    cdef double bestval = -1
    bestcol = -1
    for j in range(size_col):
        currval = fabs(f(i, columns[j], N))
        if bestval < currval:
            bestcol = columns[j]
            bestval = currval
    return bestcol


cpdef int arg_max_row(f, long[:] rows, int j, int N, int size_row):
    # Calculate maximum value in a row from b1 or b2
    cdef int bestrow, i
    cdef double currval
    cdef double bestval = -1
    bestrow = -1
    for i in range(size_row):
        currval = fabs(f(rows[i], j, N))
        if bestval < currval:
            bestrow = rows[i]
            bestval = currval
    return bestrow


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
        # print(f"Full rk error : {frobenius_norm_mat(Rk)}")
    C = o_matrix[:, j_set]
    U = np.linalg.inv(o_matrix[i_set, :][:, j_set])
    R = o_matrix[i_set, :]
    return C, U, R

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
    cdef double delta, est_norm_Rk, added_u_v
    cdef double[:] u, v
    # cdef set[int] i_used = set[int](n/10)
    # cdef set[int] j_used = set[int](m/10)
    i_used = []
    j_used = []
    i_not_used = [x for x in range(m)]
    j_not_used = [x for x in range(n)]
    size_i = len(i_not_used)
    size_j = len(j_not_used)
    all_u = []
    all_v = []

    elem_counter = 1
    est_norm_Rk = 0
    fk = f
    i = arg_max_row(fk, np.array(i_not_used, dtype=np.int), 0, N, size_i)

    while elem_counter == 1 or mem_view_norm_sq(u, m) * mem_view_norm_sq(v, n) > max_error * max_error * est_norm_Rk:
        j = arg_max_col(fk, i, np.array(j_not_used, dtype=np.int), N, size_j)
        # fk represents current matrix, after elem_counter rank 1 updates
        delta = fk(i, j, N)
        if np.isclose(delta, 0):
            if np.min([size_i, size_j]) == 1:
                print("break occured")
                break
        else:
            u = get_u(fk, m, j, N)  # gets row vector of fk
            v = get_v(fk, i, n, delta, N)  # gets column vector of fk
            v = np.asarray(v)
            u = np.asarray(u)
            all_v.append(v)
            all_u.append(u)
            fk = closure_fk(fk, u, v)  # do rank 1 update
            elem_counter += 1
            added_u_v = 0
            for p in range(elem_counter - 1):
                added_u_v += mem_view_scalar_prod(
                    u, all_u[p], m) * mem_view_scalar_prod(all_v[p], v, n)
            est_norm_Rk += mem_view_norm_sq(u, m) * \
                mem_view_norm_sq(v, n) + 2 * added_u_v
            i_used.append(i)
            j_used.append(j)
        try:
            i_not_used.remove(i)
        except Exception as ex:
            print(i)
        size_i -= 1
        try:
            j_not_used.remove(j)
        except Exception as ex:
            print(j)
        size_j -= 1
        i = arg_max_row(fk, np.array(i_not_used, dtype=np.int), j, N, size_i)

    R = np.array([[f(i, j, N) for j in range(n)] for i in i_used])
    U = np.linalg.inv(np.array([[f(i, j, N) for j in j_used] for i in i_used]))
    C = np.array([[f(i, j, N) for j in j_used] for i in range(m)])
    return C, U, R
