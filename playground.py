import tensorly as tl
import numpy as np
import pytest
import time

tensor = tl.tensor(np.arange(24).reshape((3, 4, 2)))
unfolded = tl.unfold(tensor, mode=0)
tl.fold(unfolded, mode=0, shape=tensor.shape)


def generator_B_one(N, d=3):
    # Generates a matrix of the form ()
    B1 = tl.tensor(np.ones((np.repeat(N, d))))
    # Very bad three loop style
    for i in range(N):
        for j in range(N):
            for z in range(N):
                B1[i, j, z] = np.sin(i / (N + 1) + j / (N + 1) + z / (N + 1))
    return B1


B1 = generator_B_one(200)
# Do SVD for each node unfolded matrix:
u0, s0, vh0 = np.linalg.svd(unfolded=tl.unfold(
    B1, mode=0), full_matrices=False)
u1, s1, vh1 = np.linalg.svd(unfolded=tl.unfold(
    B1, mode=1), full_matrices=False)
u2, s2, vh2 = np.linalg.svd(unfolded=tl.unfold(
    B1, mode=2), full_matrices=False)


def test_generator(benchmark):
    # benchmark something
    benchmark(generator_B_one(200))
    assert(True)
