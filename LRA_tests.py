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
    # Full rank reconstruction should have low frob norm error
    N = 50
    tensor = np.asarray(utilis.get_B_one(N))
    Core, U = utilis.compute_core(tensor, ranks=[N, N, N])
    recon_tensor = utilis.reconstruct_tensor(U, Core, tensor)
    error = utilis.frobenius_norm_tensor(recon_tensor - tensor)
    assert(error < 10e-10)


def test_hosvd_B2():
    # Full rank reconstruction should have low frob norm error
    N = 50
    tensor = np.asarray(utilis.get_B_two(N))
    Core, U = utilis.compute_core(tensor, ranks=[N, N, N])
    recon_tensor = utilis.reconstruct_tensor(U, Core, tensor)
    error = utilis.frobenius_norm_tensor(recon_tensor - tensor)
    assert(error < 10e-10)

##########################
#         Project 2      #
##########################


def test_full_pivot():
    # Full rank reconstruction should have low frob norm error
    test_mat = np.random.uniform(-5, 5, ((50, 50)))
    C, U, R = utilis.aca_full_pivoting(test_mat, 10e-19)
    recon = np.dot(C, np.dot(U, R))
    error = utilis.frobenius_norm_mat(recon - test_mat)
    del(C, U, R, recon)
    C, U, R = utilis.aca_full_pivoting(test_mat, 10e-2)
    recon = np.dot(C, np.dot(U, R))
    print(f"Shape of reconstructed CUR (full) {recon.shape}")
    error = utilis.frobenius_norm_mat(recon - test_mat)
    assert(error < 10e-1 * utilis.frobenius_norm_mat(test_mat))


def test_generator_B1():
    # check whether functional and loop B1 generator are equal
    N = 50
    tensor = np.asarray(utilis.get_B_one(N))
    gen_tensor = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            for z in range(N):
                gen_tensor[i, j, z] = utilis.f1(i, j, z, N)
    assert(np.allclose(tensor, gen_tensor))


def test_generator_B2():
    # check whether functional and loop B2 generator are equal
    N = 50
    tensor = np.asarray(utilis.get_B_two(N))
    gen_tensor = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            for z in range(N):
                gen_tensor[i, j, z] = utilis.f2(i, j, z, N)
    assert(np.allclose(tensor, gen_tensor))

##########################
#      Benchmarking      #
##########################


@pytest.mark.slow
def test_speed_hosvd_B1(benchmark):
    # Benchmark speed for N=200 for decomposition without reconstruction
    N = 100
    tensor = np.asarray(utilis.get_B_one(N))
    benchmark(utilis.compute_core, tensor, ranks=[N, N, N])


@pytest.mark.slow
def test_speed_hosvd_B2(benchmark):
    # Benchmark speed for N=200 for decomposition without reconstruction
    N = 100
    tensor = np.asarray(utilis.get_B_two(N))
    benchmark(utilis.compute_core, tensor, ranks=[N, N, N])


@pytest.mark.slow
def test_speed_gen_B1_py(benchmark):
    # Benchmark speed for N=200 for decomposition without reconstruction
    N = 100
    benchmark(py_gen_B1, N)


@pytest.mark.slow
def test_speed_gen_B2_py(benchmark):
    # Benchmark speed for N=200 for decomposition without reconstruction
    N = 100
    benchmark(py_gen_B2, N)


def test_speed_gen_B1(benchmark):
    # Benchmark speed for N=200 for decomposition without reconstruction
    N = 100
    benchmark(utilis.get_B_one, N)


def test_speed_gen_B2(benchmark):
    # Benchmark speed for N=200 for decomposition without reconstruction
    N = 100
    benchmark(utilis.get_B_one, N)
