import io
import os
import re

from setuptools import find_packages
from setuptools import setup

from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np


extensions = [Extension('rapidearth.cython.utils', ["rapidearth/cython/utils.pyx"]),
            Extension('rapidearth.cython.functions', ["rapidearth/cython/functions.pyx"],
            include_dirs=[np.get_include()],extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'])]

def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="py_kdtree",
    version="0.1.0",
    url="https://github.com/cluel01/py_kdtree",
    license='MIT',

    author="Christian Lülf",
    author_email="christian.luelf@uni-muenster.de",

    description="Python implementation of KD-Tree with range search",
    long_description=read("README.rst"),

    packages=find_packages(exclude=('tests',)),

    ext_modules=cythonize(extensions),

    zip_safe=False,

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
