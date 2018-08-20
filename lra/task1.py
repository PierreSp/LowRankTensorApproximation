import utilis
import numpy as np
import argparse
import matplotlib.pyplot as plt

# parser to provide parameters from terminal
parser = argparse.ArgumentParser(
    description="Calculate HOSVD and tucker decomposition for task1."
)
parser.add_argument("dimension", type=int, default=200, help="Dimension for the tensor")
parser.add_argument(
    "--plot", help="Activates plotting of the eigenvalues", action="store_true"
)
parser.add_argument(
    "--acc",
    type=float,
    default=10e-4,
    help="Allowed error wrt. frobenius norm between tucker decomposition and original tensor",
)
args = parser.parse_args()


def plot_singular_values(tensor):
    """ Plot singular values of HOSVD
    Calculate the three mode-m matrizitation and plot all singular values
    """
    m = 0
    _, sigmas = utilis.calculate_SVD(tensor)
    for matrix in sigmas:
        plt.semilogy(range(len(matrix)), matrix, ".", markersize=7)
        plt.title(f"Singular values of {m + 1}-mode matricization")
        plt.show()
        plt.close()
        m += 1


# Runscript
if __name__ == "__main__":
    N = args.dimension
    print(f"Number of elements per dimension: {N}")

    # HOSVD of B1
    print(f"\nCalculation with B1")
    tensor = np.asarray(utilis.get_B_one(N))
    print(f"Calculation HOSVD with maximal relative error of {args.acc}")
    Core, U = utilis.compute_core(tensor, max_rel_error=args.acc)
    utilis.reconstruct_tensor(U, Core, tensor)
    if args.plot:
        print("Plotting singular values for B1")
        plot_singular_values(tensor)
    del (Core, U, tensor)

    # HOSVD of B2
    print(f"\nCalculation with B2")
    tensor = np.asarray(utilis.get_B_two(N))
    print(f"Calculation HOSVD with maximal relative error of {args.acc}")
    Core, U = utilis.compute_core(tensor, max_rel_error=args.acc)
    utilis.reconstruct_tensor(U, Core, tensor)
    if args.plot:
        print("Plotting singular values for B2")
        plot_singular_values(tensor)
    del (Core, U, tensor)
