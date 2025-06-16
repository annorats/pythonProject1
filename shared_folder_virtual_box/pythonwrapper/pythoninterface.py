#! /usr/bin/env python

# System imports
from distutils.core import *
from distutils      import sysconfig

# NumPy supplies numpy arrays that can be converted to C arrays and passed to and from C functions
import numpy

# Obtain the numpy include directory.  This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

# decimate extension module
#_demomathslibrary = Extension('_demomathslibrary',
#    extra_compile_args=['-I.'],
#    sources=['pythoninterface.i',   
#            './demolibrarysource.c'],
#    include_dirs=[numpy_include])

# decimate extension module
_demomathslibrary = Extension('_demomathslibrary',
    extra_compile_args=['-I.'],
    sources=['pythoninterface.i'],
    libraries=['demolibrary'],   
    library_dirs=['.'],
    include_dirs=[numpy_include])

# test wrapper setup
setup(  name        = 'wrapper for doubleing function',
        description = 'Function for doubling a number',
        author      = 'Ed Daw',
        version     = '0.1',
        ext_modules = [_demomathslibrary],)  
