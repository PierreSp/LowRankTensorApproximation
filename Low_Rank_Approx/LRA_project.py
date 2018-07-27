import utilis
import numpy as np
import argparse
# Add parser to run homework from terminal
parser = argparse.ArgumentParser(
    description='Calculate HOSVD and tucker decomposition for the LRA homework.')
parser.add_argument('dimension', type=int, default=200,
                    help='Dimension for the tensor')
parser.add_argument('--acc', type=float, default=10e-4,
                    help='Allowed error wrt. frobenius norm between tucker decomposition and original tensor')
args = parser.parse_args()


# Runscript
if __name__ == "__main__":
    N = args.dimension
    print(f"Number of dimension: {N}")
    print(f"Calculation rhs B with sin")
    tensor = np.asarray(utilis.get_B_one(N))
    print(f"Calculated B1")
    Core, U = utilis.compute_core(tensor, max_rel_error=args.acc)
    utilis.reconstruct_tensor(U, Core, tensor)
    print(f"Calculation rhs B with L2")
    tensor = np.asarray(utilis.get_B_two(N))
    print(f"Calculated B2")
    Core, U = utilis.compute_core(tensor, max_rel_error=args.acc)
    utilis.reconstruct_tensor(U, Core, tensor)
