from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "utilis",
        ["utilis.pyx"],
        extra_compile_args=['-fopenmp', '-O3'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='Low_rank_approximation_project',
    version='0.5',
    description='Hosvd with python',
    ext_modules=cythonize(ext_modules),
)
