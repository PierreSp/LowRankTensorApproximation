import pytest
import lra.utilis as utilis
import numpy as np
import tensorly as tl
import lra.aca as aca_fun


def _tsi(x, N):
    # Needed for generation of B1 and B2 with python
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
    N = 50
    for obj in ["B1", "B2"]:
        C_list = []
        ranks = np.array([N, N, N])
        if obj == "B1":
            tensor = np.asarray(utilis.get_B_one(N))
        else:
            tensor = np.asarray(utilis.get_B_two(N))
        for mode in range(3):
            if mode == 0:
                # Start with original matrix
                Core_mat = tl.unfold(tensor, mode)
            else:
                Core_mat = tl.unfold(Core_ten, mode)
            C, U, R = aca_fun.aca_full_pivoting(Core_mat, 10e-8)
            ranks[mode] = U.shape[0]
            Core_ten = tl.fold(np.dot(U, R), mode, ranks)
            C_list.append(C)

        recon = utilis.reconstruct_tensor(C_list, Core_ten, tensor)
        error = utilis.frobenius_norm_tensor(recon - tensor)
        assert(error < 10e-8 * utilis.frobenius_norm_tensor(tensor))


def test_generator_B1():
    # check whether functional and loop B1 generator are equal
    N = 50
    tensor = np.asarray(utilis.get_B_one(N))
    gen_tensor = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            for z in range(N):
                gen_tensor[i, j, z] = aca_fun.b1(i, j, z, N)
    assert(np.allclose(tensor, gen_tensor))


def test_generator_B2():
    # check whether functional and loop B2 generator are equal
    N = 50
    tensor = np.asarray(utilis.get_B_two(N))
    gen_tensor = np.zeros((N, N, N))
    for i in range(N):
        for j in range(N):
            for z in range(N):
                gen_tensor[i, j, z] = aca_fun.b2(i, j, z, N)
    assert(np.allclose(tensor, gen_tensor))


def test_fun_matriziation_b1():
    # check whether functional matrizitation is identical with original one
    N = 50
    for mode in range(3):
        tensor = np.asarray(utilis.get_B_one(N))
        unfold_mat = tl.unfold(tensor, mode)
        functional_generator = aca_fun.mode_m_matricization_fun(
            aca_fun.b1, N, N, N)
        gen_mat = np.zeros((N, N * N))
        for i in range(N):
            for j in range(N * N):
                gen_mat[i, j] = functional_generator(i, j, N)
        assert(np.allclose(unfold_mat, gen_mat))


def test_fun_matriziation_b2():
    # check whether functional matrizitation is identical with original one
    N = 50
    for mode in range(3):
        tensor = np.asarray(utilis.get_B_two(N))
        unfold_mat = tl.unfold(tensor, mode)
        functional_generator = aca_fun.mode_m_matricization_fun(
            aca_fun.b2, N, N, N)
        gen_mat = np.zeros((N, N * N))
        for i in range(N):
            for j in range(N * N):
                gen_mat[i, j] = functional_generator(i, j, N)
        assert(np.allclose(unfold_mat, gen_mat))


def test_aca_func():
    # check whether functional matrizitation is identical with original one
    N = 50
    for obj in ["B1", "B2"]:
        C_list = []
        ranks = np.array([N, N, N])
        for mode in range(3):
            if mode == 0:
                if obj == "B1":
                    functional_generator = aca_fun.mode_m_matricization_fun(
                        aca_fun.b1, N, N, N)
                    C, U, R = aca_fun.aca_partial_pivoting(
                        functional_generator, N, N * N, N, 10e-8 / 3)
                    tensor = np.asarray(utilis.get_B_one(N))
                else:
                    functional_generator = aca_fun.mode_m_matricization_fun(
                        aca_fun.b2, N, N, N)
                    C, U, R = aca_fun.aca_partial_pivoting(
                        functional_generator, N, N * N, N, 10e-8 / 3)
                    tensor = np.asarray(utilis.get_B_two(N))
            else:
                Core_mat = tl.unfold(Core_ten, mode)
                C, U, R = aca_fun.aca_full_pivoting(Core_mat, 10e-8 / 3)
            ranks[mode] = U.shape[0]
            Core_ten = tl.fold(np.dot(U, R), mode, ranks)
            C_list.append(C)

        recon = utilis.reconstruct_tensor(C_list, Core_ten, tensor)
        error = utilis.frobenius_norm_tensor(recon - tensor)
        assert(error < 10e-8 * utilis.frobenius_norm_tensor(tensor))


# def test_aca_part_b2():
#     N = 50
#     mode = 0  # 24558
#     tensor = np.asarray(utilis.get_B_two(N))
#     test_mat = tl.unfold(tensor, mode)
#     # C, U, R = utilis.aca_full_pivoting(test_mat, 10e-10)
#     # recon_full = np.dot(C, np.dot(U, R))
#     # del(C, U, R)

#     functional_generator = aca_fun.mode_m_matricization_fun(
#         aca_fun.b2, N, N, N)
#     C, U, R = aca_fun.aca_partial_pivoting(
#         functional_generator, N, N * N, N, 10e-10)
#     recon_part = np.dot(C, np.dot(U, R))
#     error = utilis.frobenius_norm_mat(
#         recon_part - test_mat) / utilis.frobenius_norm_mat(test_mat)
#     assert(error < 10e-10)


# def test_aca_part_b1():
#     N = 50
#     mode = 0  # 24558
#     tensor = np.asarray(utilis.get_B_one(N))
#     test_mat = tl.unfold(tensor, mode)
#     # C, U, R = utilis.aca_full_pivoting(test_mat, 10e-10)
#     # recon_full = np.dot(C, np.dot(U, R))
#     # del(C, U, R)

#     functional_generator = aca_fun.mode_m_matricization_fun(
#         aca_fun.b1, N, N, N)
#     C, U, R = aca_fun.aca_partial_pivoting(
#         functional_generator, N, N * N, N, 10e-10)
#     recon_part = np.dot(C, np.dot(U, R))
#     error = utilis.frobenius_norm_mat(
#         recon_part - test_mat) / utilis.frobenius_norm_mat(test_mat)
#     assert(error < 10e-10)


##########################
#      Benchmarking      #
##########################


@pytest.mark.slow
def test_speed_hosvd_B1_N200(benchmark):
    # Benchmark speed for N=200 for decomposition without reconstruction
    N = 400
    tensor = np.asarray(utilis.get_B_one(N))
    benchmark(utilis.compute_core, tensor, max_rel_error=10e-5)


@pytest.mark.slow
def test_speed_hosvd_B2_N200(benchmark):
    # Benchmark speed for N=200 for decomposition without reconstruction
    N = 400
    tensor = np.asarray(utilis.get_B_two(N))
    benchmark(utilis.compute_core, tensor, max_rel_error=10e-5)


# @pytest.mark.slow
# def test_speed_gen_B1_py_N200(benchmark):
#     # Benchmark speed for N=200 for decomposition without reconstruction
#     N = 200
#     benchmark(py_gen_B1, N)


# @pytest.mark.slow
# def test_speed_gen_B2_py_N200(benchmark):
#     # Benchmark speed for N=200 for decomposition without reconstruction
#     N = 200
#     benchmark(py_gen_B2, N)


def test_speed_gen_B1_N200(benchmark):
    # Benchmark speed for N=200 for decomposition without reconstruction
    N = 200
    benchmark(utilis.get_B_one, N)


def test_speed_gen_B2_N200(benchmark):
    # Benchmark speed for N=200 for decomposition without reconstruction
    N = 200
    benchmark(utilis.get_B_one, N)


# @pytest.mark.slow
# def test_speed_aca_part_b1_N200(benchmark):
#     N = 200
#     mode = 0  # 24558
#     functional_generator = aca_fun.mode_m_matricization_fun(
#         aca_fun.b1, N, N, N)
#     C, U, R = aca_fun.aca_partial_pivoting(
#         functional_generator, N, N * N, N, 10e-10)

@pytest.mark.slow
def test_speed_aca_full_b1_N200(benchmark):
    # Benchmark first mode 1 aca for B1
    def run_aca_full():
        N = 200
        C_list = []
        ranks = np.array([N, N, N])
        tensor = np.asarray(utilis.get_B_one(N))
        for mode in range(3):
            if mode == 0:
                # Start with original matrix
                Core_mat = tl.unfold(tensor, mode)
            else:
                Core_mat = tl.unfold(Core_ten, mode)
                C, U, R = aca_fun.aca_full_pivoting(Core_mat, 10e-5)
            ranks[mode] = U.shape[0]
            print(f'Current ranks: {ranks}')
            Core_ten = tl.fold(np.dot(U, R), mode, ranks)
            C_list.append(C)
    benchmark(run_aca_full)


@pytest.mark.slow
def test_speed_aca_full_b2_N200(benchmark):
    # Benchmark first mode 1 aca for B2
    def run_aca_full():
        N = 200
        C_list = []
        ranks = np.array([N, N, N])
        tensor = np.asarray(utilis.get_B_two(N))
        for mode in range(3):
            print(f'Currently in mode {mode + 1} step')
            if mode == 0:
                # Start with original matrix
                Core_mat = tl.unfold(tensor, mode)
            else:
                Core_mat = tl.unfold(Core_ten, mode)
            C, U, R = aca_fun.aca_full_pivoting(Core_mat, 10e-5)

            ranks[mode] = U.shape[0]
            Core_ten = tl.fold(np.dot(U, R), mode, ranks)
            C_list.append(C)
    benchmark(run_aca_full)
