import tensorly as tl
import numpy as np
import numba as nb


class NotInitializedError(Exception):
    """Exception raised when core not initialized.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


class HighOrderSVD():

    def __init__(self, N=200, d=3, tensor="B1"):
        self.N = N
        self.d = d
        if tensor == "B1":
            self.tensor = self._generator_B_one()
        elif tensor == "B2":
            self.tensor = self._generator_B_one()
        else:
            self.tensor = tensor
        self.dims = self.tensor.shape
        if len(self.dims) < 3:
            raise ValueError("Please enter a tensor not a matrix or vector")
        self.core = False
        self.tucker = False

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
            raise ValueError("Dimensions for mode_mul were wrong! Tensor: {0}, Matrix: {1}".format(
                str(tensor.shape[mode]), str(matrix.shape[1])))
        new_shape = list(tensor.shape)
        new_shape[mode] = matrix.shape[0]
        out = np.dot(matrix, tl.unfold(tensor, mode))
        return tl.fold(out, mode, new_shape)

    @nb.jit
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

    @nb.jit
    def _tensor_frob_norm(self, tensor):
        """Compute the frobenius norm of a tensor

        Parameters
        ----------
        tensor : tl.tensor or ndarray
        """
        pass

    @nb.jit
    def calculate_core(self, tucker=False):
        """Computed the core tensor of a given tensor and all orthogonal matrices s.t. A=Sx_1U_1.Tx_2U_2.Tx_3U_3.T

        Parameters
        ----------
        tensor : tl.tensor or ndarray
        """
        if tucker:
            if not self.core:
                raise NotInitializedError(
                    "The core was not initialized, please run calculate_core with tucker=False")
            Core = self.tensor
            for i in range(len(self.dims)):
                Core = self._mode_mul(Core, self.U_truncated[i].T, mode=i)
            self.C = Core
            self.tucker = True
            return Core
        else:
            tensor = self.tensor
            self.U = []
            # Calculate SVD for all mode matricitations and save the U matrices
            for (i, size) in enumerate(tensor.shape):
                U, s0, vh0 = np.linalg.svd(tl.unfold(
                    tensor, mode=i), full_matrices=False)
                self.U.append(U)
            # Calculate S = A x_1 U_1^T x_2...
            res = tensor
            for i in range(len(self.dims)):
                res = self._mode_mul(res, self.U[i].T, mode=i)
            self.S = res
            self.core = True
            return res

    def _checkHOSVD(self):
        """
        Computed t
        """
        recon = self.S
        for i in range(len(self.dims)):
            recon = self._mode_mul(recon, self.U[i], mode=i)
        return np.allclose(self.tensor, recon)

    def tucker_decomposition(self, ranks):
        if not self.core:
            raise NotInitializedError(
                "The core was not initialized, please run calculate_core with tucker=False")
        self.U_truncated = []
        for i, rank in enumerate(ranks):
            rank = int(rank)
            self.U_truncated.append(self.U[i][:, 0:rank])
        self.calculate_core(tucker=True)

    def tucker_opt(self, acc=10e-4):
        ranks = np.ones(len(self.dims))
        err = 1
        counter = 0
        while (err > acc):
            self.tucker_decomposition(ranks)
            err = self.frob_tucker()
            ranks += 1
            counter += 1
            if counter >= self.N:
                raise ValueError("Did not achieve wanted accuracy?")
        print("Achieved wanted accuracy with ranks: " + str(ranks))
        print("Accuracy: " + str(err))

    def frob_tucker(self):
        if not self.tucker:
            raise NotInitializedError(
                "The tucker was not initialized, please run tucker_decomposition")

        # Calculate truncated tensor using tucker core C
        res = self.C
        for i in range(len(self.tensor.shape)):
            res = self._mode_mul(res, self.U_truncated[i], mode=i)
        delta = tl.unfold(res - self.tensor, 0)
        error = np.linalg.norm(delta, "fro")
        print("The difference wrt to the frobenius norm is: " +
              str(error))
        return error


def testing_suit(benchmark):
    N = 200
    HO = HighOrderSVD(N)
    benchmark(HO.calculate_core)
    # tl.set_backend('tensorflow')
    # benchmark(HO.calculate_core, HO.tensor)
    # tl.set_backend('pytorch')
    # benchmark(HO.calculate_core)
    tl.set_backend('numpy')
    assert(HO._checkHOSVD())
    HO.tucker_decomposition([N - 1, N - 1, N - 1])
    assert(HO.frob_tucker() < 10e-1)


if __name__ == "__main__":
    HO = HighOrderSVD(100)
    # print("The core of the B1 tensor given is: ")
    HO.calculate_core()
    HO.tucker_decomposition([5, 5, 5])
    print(HO.frob_tucker())
    HO.tucker_opt()
