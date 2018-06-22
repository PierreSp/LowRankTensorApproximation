import tensorly as tl
import numpy as np
import numba as nb


class NotInitializedError(Exception):

    def __init__(self, message):
        self.message = message


class HighOrderSVD():

    def __init__(self, tensor):
        self.N = tensor.shape[0]
        self.tensor = tensor
        self.dims = self.tensor.shape
        self.ndims = len(self.dims)
        if self.ndims < 3:
            raise ValueError("Please enter a tensor not a matrix or vector")
        self.U = []
        self.U_truncated = []

#######################
#        Tools        #
#######################

    @nb.jit
    def _mode_mul(self, tensor, matrix, mode=0):
        """Computed the mode product between a matrix and a tensor

        Parameters
        ----------
        tensor : tl.tensor or ndarray
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

    def _recon_hosvd(self):
        """
        Reconstruct inital tensor from HOSVD and check whether it is the original tensor
        """
        recon = self.S
        for i in range(self.ndims):
            recon = self._mode_mul(recon, self.U[i], mode=i)
        return np.allclose(self.tensor, recon)

#######################
#      Functions      #
#######################

    @nb.jit
    def hosvd(self, ranks=None):
        """Computed the core tensor of a given tensor and all orthogonal matrices s.t. A=Sx_1U_1.Tx_2U_2.Tx_3U_3.T
           If ranks are given, the tucker decomposition with the given ranks is calculated

        Parameters
        ----------
        tensor : tl.tensor or ndarray
        """
        if ranks is not None:
            if ranks is None or len(ranks) != self.ndims:
                raise ValueError(
                    "Please set ranks according to the dimensions of the tensor")
            if self.U == []:
                self.hosvd()
            self.U_truncated = []
            for i, rank in enumerate(ranks):
                self.U_truncated.append(self.U[i][:, 0:int(rank)])
            Core = self.tensor
            for i in range(self.ndims):
                Core = self._mode_mul(Core, self.U_truncated[i].T, mode=i)
            self.TC = Core  # Save tucker core
            self.tucker = True
            return Core
        else:
            tensor = self.tensor
            self.U = []
            # Calculate SVD for all mode matricitations and also save economic
            # U matrices
            for (i, size) in enumerate(tensor.shape):
                U, s0, vh0 = np.linalg.svd(tl.unfold(
                    tensor, mode=i), full_matrices=False)
                self.U.append(U)
            # Calculate S = A x_1 U_1^T x_2...
            res = tensor
            for i in range(self.ndims):
                res = self._mode_mul(res, self.U[i].T, mode=i)
            self.S = res
            self.core = True
            return res

    def tucker_opt(self, lbound=1, rbound=None, acc=10e-4):
        """ Search for symmetric rank using recursive binary search"""
        if rbound is None:
            rbound = self.N
        if lbound + 1 == rbound:
            print("Achieved accuracy with ranks: " + str(np.ones(self.ndims) * rbound))
            return rbound
        middle = np.floor((lbound + rbound) / 2)
        ranks = middle * np.ones(self.ndims)
        self.hosvd(ranks=ranks)
        err = self.recon_tucker()
        print("Accuracy with ranks: " + str(ranks) + " is : " + str(err))
        if err < acc:
            new_rank = self.tucker_opt(lbound, middle)
            return new_rank
        elif err > acc:
            new_rank = self.tucker_opt(middle, rbound)
            return new_rank

    @nb.jit
    def recon_tucker(self):
        """ Reconstruct tensor using the tucker decomposition.
        Also returns the error wrt. the frobenius norm compared to the input.the
        """
        if self.U_truncated == []:
            raise NotInitializedError(
                "The tucker was not initialized, please run tucker_decomposition")
        res = self.TC
        for i in range(len(self.tensor.shape)):
            res = self._mode_mul(res, self.U_truncated[i], mode=i)
        delta = tl.unfold(res - self.tensor, 0)
        error = np.linalg.norm(delta, "fro")
        return error


def testing(benchmark):
    tensor = np.ones((100, 100, 100))
    HO = HighOrderSVD(tensor)
    benchmark(HO.hosvd)
    # tl.set_backend('tensorflow')
    # benchmark(HO.calculate_core, HO.tensor)
    # tl.set_backend('pytorch')
    # benchmark(HO.calculate_core)
    tl.set_backend('numpy')
    assert(HO._recon_hosvd())
    HO.hosvd(ranks=[100 - 1, 100 - 1, 100 - 1])
    assert(HO.recon_tucker() < 10e-1)
