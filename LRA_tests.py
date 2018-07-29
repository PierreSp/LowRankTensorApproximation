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
                gen_tensor[i, j, z] = utilis.b1(i, j, z, N)
    assert(np.allclose(tensor, gen_tensor))


def test_generator_B2():
    # check whether functional and loop B2 generator are equal
    N = 50
    tensor = np.asarray(utilis.get_B_two(N))
    gen_tensor = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            for z in range(N):
                gen_tensor[i, j, z] = utilis.b2(i, j, z, N)
    assert(np.allclose(tensor, gen_tensor))


def test_fun_matriziation_b1():
    # check whether functional matrizitation is identical with original one
    N = 100
    for mode in range(3):
        tensor = np.asarray(utilis.get_B_one(N))
        unfold_mat = tl.unfold(tensor, mode)
        functional_generator = utilis.mode_m_matricization_fun(
            utilis.b1, N, N, N, mode)
        gen_mat = np.zeros((N, N * N))
        for i in range(N):
            for j in range(N * N):
                gen_mat[i, j] = functional_generator(i, j, N)
        assert(np.allclose(unfold_mat, gen_mat))


def test_fun_matriziation_b2():
    # check whether functional matrizitation is identical with original one
    N = 100
    for mode in range(3):
        tensor = np.asarray(utilis.get_B_two(N))
        unfold_mat = tl.unfold(tensor, mode)
        functional_generator = utilis.mode_m_matricization_fun(
            utilis.b2, N, N, N, mode)
        gen_mat = np.zeros((N, N * N))
        for i in range(N):
            for j in range(N * N):
                gen_mat[i, j] = functional_generator(i, j, N)
        assert(np.allclose(unfold_mat, gen_mat))


def test_aca_part_b2():
    N = 100
    mode = 0  # 24558
    tensor = np.asarray(utilis.get_B_two(N))
    test_mat = tl.unfold(tensor, mode)
    # C, U, R = utilis.aca_full_pivoting(test_mat, 10e-10)
    # recon_full = np.dot(C, np.dot(U, R))
    # del(C, U, R)

    functional_generator = utilis.mode_m_matricization_fun(
        utilis.b2, N, N, N, mode)
    C, U, R = utilis.aca_partial_pivoting(
        functional_generator, N, N * N, N, 10e-10)
    recon_part = np.dot(C, np.dot(U, R))
    error = utilis.frobenius_norm_mat(
        recon_part - test_mat) / utilis.frobenius_norm_mat(test_mat)
    assert(error < 10e-10)


def test_aca_part_b1():
    N = 100
    mode = 0  # 24558
    tensor = np.asarray(utilis.get_B_one(N))
    test_mat = tl.unfold(tensor, mode)
    # C, U, R = utilis.aca_full_pivoting(test_mat, 10e-10)
    # recon_full = np.dot(C, np.dot(U, R))
    # del(C, U, R)

    functional_generator = utilis.mode_m_matricization_fun(
        utilis.b1, N, N, N, mode)
    C, U, R = utilis.aca_partial_pivoting(
        functional_generator, N, N * N, N, 10e-10)
    recon_part = np.dot(C, np.dot(U, R))
    error = utilis.frobenius_norm_mat(
        recon_part - test_mat) / utilis.frobenius_norm_mat(test_mat)
    assert(error < 10e-10)


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


@pytest.mark.slow
def test_speed_aca_part_b1(benchmark):
    N = 100
    mode = 0  # 24558
    functional_generator = utilis.mode_m_matricization_fun(
        utilis.b1, N, N, N, mode)
    C, U, R = utilis.aca_partial_pivoting(
        functional_generator, N, N * N, N, 10e-10)
