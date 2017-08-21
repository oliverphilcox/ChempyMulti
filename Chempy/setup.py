from distutils.core import setup
from Cython.Build import cythonize

setup(name="CyChempy", ext_modules=cythonize('CyChempy.pyx'),)