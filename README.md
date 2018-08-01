# Low Rank Tensor Approximation - Mini Project
High Order SVD and ACA project of the [Low Rank approximation lecture](https://www5.in.tum.de/wiki/index.php/Low_Rank_Approximation)

### How to run:
1. Install all requirements '''pip install -r requirements.txt'''
2. Run '''python python setup.py build_ext --inplace'''
3. Task1 / HOSVD: run python LRA_project.py domension --acc rel_error with the demanded dimension and relative error. --plot creates plots of the singular values
4. Task5 / ACA: run python LRA_project.py domension --acc rel_error with the demanded dimension and relative error
5. Benchmarks and tests are in LRA_tests.py and have to be run with pytest with its [benchmarking module](https://github.com/ionelmc/pytest-benchmark). RUn with --slowrun to run all benchmarks (takes a lot of time)