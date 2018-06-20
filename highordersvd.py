import tensorly as tl
import numpy as np
import pytest
import time

tensor = tl.tensor(np.arange(24).reshape((3, 4, 2)))
unfolded = tl.unfold(tensor, mode=0)
tl.fold(unfolded, mode=0, shape=tensor.shape)


class HighOrderSVD():

    def __init__(self, N=200, d=3):
        self.N = N
        self.d = d
        self.B1 = self._generator_B_one()

    def _mode_mul(self, tensor, matrix, mode=0):
        """Computed the mode product between a matrix and a tensor


        Parameters
        ----------
        tensor : tl.tensor or ndarray
        matrix : ndarray
        mode : int
        """

        if matrix.shape[1] != tensor.shape[mode]:
            raise ValueError("Dimensions for mode_mul were wrong! Tensor: {0}, Matrix: {1}".format(
                str(tensor.shape(mode)), str(matrix.shape(1))))
        new_shape = matrix.shape[0]
        out = np.matmul(matrix, tl.unfold(tensor, mode))
        return tl.fold(out, mode, new_shape)

    def _generator_B_one(self):
        # Generates a matrix of the form ()
        B1 = tl.tensor(np.ones((np.repeat(self.N, self.d))))
        # Very bad three loop style
        for i in range(self.N):
            for j in range(self.N):
                for z in range(self.N):
                    B1[i, j, z] = np.sin(
                        i / (self.N + 1) + j / (self.N + 1) + z / (self.N + 1))
        return B1

    def calculate_core(self, tensor):
        """Computed the core tensor of a given tensor and all orthogonal matrices s.t. A=Sx_1U_1x_2U_2x_3U_3


        Parameters
        ----------
        tensor : tl.tensor or ndarray
        """
        self.U = np.shape(len(tensor.shape), self.N, self.N)

        # Calculate SVD for all mode matricitations and save the U matrices
        for (i, size) in enumerate(tensor.shape):
            self.U[i, :, :], s0, vh0 = np.linalg.svd(tl.unfold(
                tensor, mode=i), full_matrices=False)

            self.S = self._mode_mul(self.u0,
                                    self._mode_mul(self.u1,
                                                   self._mode_mul(self.u2, self.B1, 2), 1), 0)
        # Calculate S = A x_1 U_1^T x_2...
        res = tensor
        for i in range(len(tensor.shape)):
            res = self._mode_mul(tensor, self.U[i], mode=i)
        self.S = res
        return res


def testing_suit(benchmark):
    HO = HighOrderSVD()
    benchmark(HO.calculate_core())
    assert(np.allclose(HO.B1, HO.checkA()))
