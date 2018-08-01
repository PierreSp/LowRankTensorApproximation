import utilis
import numpy as np
import argparse
import tensorly as tl
import aca as aca_fun
import time

# parser to provide parameters from terminal
parser = argparse.ArgumentParser(
    description='Calculate ACA and tucker decomposition for the LRA task4.')
parser.add_argument('dimension', type=int, default=200,
                    help='Dimension for the tensor')
parser.add_argument(
    '--full', help='Calculates full ACA', action='store_true')
parser.add_argument(
    '--speed', help='Calculates ACA without relative error (and B1)', action='store_true')
parser.add_argument('--acc', type=float, default=10e-4,
                    help='Allowed error wrt. frobenius norm between tucker decomposition and original tensor')
args = parser.parse_args()


# Runscript
if __name__ == "__main__":
    N = args.dimension
    print(f"Number of elements per dimension: {N}")
    print(f"Demanded error is: {args.acc}")

    if args.full:
        # Calculate full ACA
        for obj in ["B1", "B2"]:
            timer = time.time()
            all_C = []
            ranks = np.array([N, N, N])
            print(f"    Working on {obj} ")
            if obj == "B1":
                tensor = np.asarray(utilis.get_B_one(N))
            else:
                tensor = np.asarray(utilis.get_B_two(N))
            for mode in range(3):
                print(f'        Currently in mode {mode + 1} step')
                if mode == 0:
                    # Start with original matrix
                    Core_mat = tl.unfold(tensor, mode)
                else:
                    Core_mat = tl.unfold(Core_ten, mode)
                C, U, R = aca_fun.aca_full_pivoting(Core_mat, args.acc / 3)
                ranks[mode] = U.shape[0]
                print(f'        Current ranks: {ranks}')
                Core_ten = tl.fold(np.dot(U, R), mode, ranks)
                all_C.append(C)
            utilis.reconstruct_tensor(all_C, Core_ten, tensor)
            print(f"Time needed: {time.time() - timer}")
    else:

        for obj in ["B1", "B2"]:
            timer = time.time()
            print(f"    Working on {obj} ")
            all_C = []
            ranks = np.array([N, N, N])
            for mode in range(3):
                print(f'        Currently in mode {mode + 1} step')
                if mode == 0:
                    if obj == "B1" and not args.speed:
                        functional_generator = aca_fun.mode_m_matricization_fun(
                            aca_fun.b1, N, N, N)  # Initializes the functional B
                        C, U, R = aca_fun.aca_partial_pivoting(
                            functional_generator, N, N * N, N, args.acc / 3)
                        if not args.speed:
                            print("        Speed option activated")
                            tensor = np.asarray(utilis.get_B_one(N))
                    else:
                        functional_generator = aca_fun.mode_m_matricization_fun(
                            aca_fun.b2, N, N, N)  # Initializes the functional B
                        C, U, R = aca_fun.aca_partial_pivoting(
                            functional_generator, N, N * N, N, args.acc / 3)
                        if not args.speed:
                            tensor = np.asarray(utilis.get_B_two(N))
                else:
                    Core_mat = tl.unfold(Core_ten, mode)
                    C, U, R = aca_fun.aca_full_pivoting(Core_mat, args.acc / 3)
                ranks[mode] = U.shape[0]
                print(f'        Current ranks: {ranks}')
                Core_ten = tl.fold(np.dot(U, R), mode, ranks)
                all_C.append(C)
            if not args.speed:
                recon = utilis.reconstruct_tensor(all_C, Core_ten, tensor)
            print(f"Time needed: {time.time() - timer}")
