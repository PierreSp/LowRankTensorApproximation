import numpy as np
import numba as nb
from highordersvd import HighOrderSVD


def tsi(x, N):
    return(x / (N + 1))


@nb.jit
def gen_B_one(N):
    B1 = np.ones((np.repeat(N, 3)))
    for i in range(N):
        for j in range(N):
            for z in range(N):
                B1[i, j, z] = np.sin(
                    np.sum(tsi(np.array([i, j, z]), N)))
    return B1


@nb.jit
def gen_B_two(N):
    B2 = np.ones((np.repeat(N, 3)))
    for i in range(N):
        for j in range(N):
            for z in range(N):
                B2[i, j, z] = np.linalg.norm(
                    tsi(np.array([i, j, z]), N))
    return B2


if __name__ == "__main__":
    # Sinus rhs
    tensor = gen_B_two(100)
    HO = HighOrderSVD(tensor)
    HO.tucker_opt(1, HO.N)
    # L2 rhs
    tensor = gen_B_one(100)
    HO = HighOrderSVD(tensor)
    HO.tucker_opt(1, HO.N)
