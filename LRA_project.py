import numpy as np
import numba as nb
import argparse
from highordersvd import HighOrderSVD


parser = argparse.ArgumentParser(
    description='Calculate HOSVD and tucker decomposition for the LRA homework.')
parser.add_argument('dimension', type=int, default=200,
                    help='Dimension for the tensor')
parser.add_argument('--acc', type=float, default=10e-4,
                    help='Allowed error wrt. frobenius norm between tucker decomposition and original tensor')
args = parser.parse_args()


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
    N = args.dimension
    print("Number of dimension: " + str(N))
    print("Calculation rhs B with sin")
    tensor = gen_B_one(N)
    print("Calculated B1")
    HO = HighOrderSVD(tensor)
    HO.tucker_opt(1, HO.N, acc=args.acc)
    print("Calculation rhs B with L2")
    tensor = gen_B_two(N)
    print("Calculated B2")
    HO = HighOrderSVD(tensor)
    HO.tucker_opt(1, HO.N, acc=args.acc)
