#!/usr/bin/env python

'''
setup.py file for biosaur2
'''
import os
from setuptools import setup, find_packages, Extension

version = open('VERSION', encoding="utf8").readline().strip()


def make_extensions():
    is_ci = bool(os.getenv("CI", ""))
    include_diagnostics = False
    try:
        import numpy
    except ImportError:
        print("C Extensions require `numpy`")
        raise
    try:
        from Cython.Build import cythonize
        cython_directives = {
            'embedsignature': True,
            "profile": include_diagnostics
        }
        macros = []
        if include_diagnostics:
            macros.append(("CYTHON_TRACE_NOGIL", "1"))
        if is_ci and include_diagnostics:
            cython_directives['linetrace'] = True

        extensions = cythonize([
            Extension(name='biosaur2.cutils', sources=['biosaur2/cutils.pyx'],
                      include_dirs=[numpy.get_include(), ])
        ], compiler_directives=cython_directives)
    except ImportError:
        extensions = [
            Extension(name='biosaur2.cutils', sources=['biosaur2/cutils.c'],
                      include_dirs=[numpy.get_include(), ])
        ]
    return extensions

setup(
    name                 = 'biosaur2',
    version              = version,
    description          = '''A feature detection LC-MS1 spectra.''',
    long_description     = (''.join(open('README.md', encoding="utf8").readlines())),
    long_description_content_type = 'text/markdown',
    author               = 'Mark Ivanov',
    author_email         = 'markmipt@gmail.com',
    install_requires     = [line.strip() for line in open('requirements.txt', encoding="utf8")],
    ext_modules          = make_extensions(),
    classifiers          = ['Intended Audience :: Science/Research',
                            'Programming Language :: Python :: 3.9',
                            'Topic :: Education',
                            'Topic :: Scientific/Engineering :: Bio-Informatics',
                            'Topic :: Scientific/Engineering :: Chemistry',
                            'Topic :: Scientific/Engineering :: Physics'],
    license              = 'License :: OSI Approved :: Apache Software License',
    packages             = find_packages(),
    entry_points         = {'console_scripts': ['biosaur2 = biosaur2.search:run',]}
    )
