#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Get the directory of this file
here = os.path.abspath(os.path.dirname(__file__))

extensions = [
    Extension(
        "euler_newton_cython",
        ["euler_newton_cython.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-ffast-math"],
        extra_link_args=["-O3"],
    ),
    Extension(
        "newton_cython",
        ["newton_cython.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-ffast-math"],
        extra_link_args=["-O3"],
    ),
]

setup(
    name="pacopy-cython",
    version="0.1.0",
    description="Cython-optimized version of pacopy for fast continuation",
    author="Nico Schlömer",
    author_email="nico.schloemer@gmail.com",
    url="https://github.com/schlomen/pacopy",
    packages=["."],
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
        },
    ),
    install_requires=[
        "numpy",
        "scipy",
    ],
    zip_safe=False,
)