import pytest
import lra.utilis as utilis
import numpy as np
import tensorly as tl


def _tsi(x, N):
    return(x / (N + 1))


def py_gen_B1(N):
    # creating B1 with python for comparison
    B1 = tl.tensor(np.ones((np.repeat(N, 3))))
    for i in range(N):
        for j in range(N):
            for z in range(N):
                B1[i, j, z] = np.sin(
                    np.sum(_tsi(np.array([i, j, z]), N)))
    return B1


def py_gen_B2(N):
    # creating B2 with python for comparison
    B2 = tl.tensor(np.ones((np.repeat(N, 3))))
    for i in range(N):
        for j in range(N):
            for z in range(N):
                B2[i, j, z] = np.linalg.norm(
                    _tsi(np.array([i, j, z]), N))
    return B2


##########################
#         Project 1      #
##########################

def test_hosvd_B1():
    # Full rank repo should have low frob norm error
    N = 50
    tensor = np.asarray(utilis.get_B_one(N))
    Core, U = utilis.compute_core(tensor, ranks=[N, N, N])
    recon_tensor = utilis.reconstruct_tensor(U, Core, tensor)
    error = utilis.frobenius_norm(recon_tensor - tensor, N)
    assert(error < 10e-10)


def test_hosvd_B2():
    # Full rank repo should have low frob norm error
    N = 50
    tensor = np.asarray(utilis.get_B_two(N))
    Core, U = utilis.compute_core(tensor, ranks=[N, N, N])
    recon_tensor = utilis.reconstruct_tensor(U, Core, tensor)
    error = utilis.frobenius_norm(recon_tensor - tensor, N)
    assert(error < 10e-10)


##########################
#      Benchmarking      #
##########################

def test_speed_hosvd_B1(benchmark):
    # Benchmark speed for N=200 for decomposition without reconstruction
    N = 200
    tensor = np.asarray(utilis.get_B_one(N))
    benchmark(utilis.compute_core, tensor, ranks=[N, N, N])


def test_speed_hosvd_B2(benchmark):
    # Benchmark speed for N=200 for decomposition without reconstruction
    N = 200
    tensor = np.asarray(utilis.get_B_two(N))
    benchmark(utilis.compute_core, tensor, ranks=[N, N, N])


@pytest.mark.slow
def test_speed_gen_B1_py(benchmark):
    # Benchmark speed for N=200 for decomposition without reconstruction
    N = 200
    benchmark(py_gen_B1, N)


@pytest.mark.slow
def test_speed_gen_B2_py(benchmark):
    # Benchmark speed for N=200 for decomposition without reconstruction
    N = 200
    benchmark(py_gen_B2, N)


def test_speed_gen_B1(benchmark):
    # Benchmark speed for N=200 for decomposition without reconstruction
    N = 200
    benchmark(utilis.get_B_one, N)


def test_speed_gen_B2(benchmark):
    # Benchmark speed for N=200 for decomposition without reconstruction
    N = 200
    benchmark(utilis.get_B_one, N)
