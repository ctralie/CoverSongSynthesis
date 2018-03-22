from distutils.core import setup, Extension
import numpy

c_ext = Extension("_Driedger", ["_Driedger.c", "Driedger.c"], include_dirs=[numpy.get_include()])

setup(
    ext_modules=[c_ext],
    include_dirs=[numpy.get_include()],
)
