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
    name='fast_tools',
    ext_modules=cythonize(ext_modules),
)