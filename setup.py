#!/usr/bin/env python

'''
setup.py file for biosaur2
'''

from setuptools import setup, find_packages

version = open('VERSION').readline().strip()

setup(
    name                 = 'biosaur2',
    version              = version,
    description          = '''A feature detection LC-MS1 spectra.''',
    long_description     = (''.join(open('README.md').readlines())),
    long_description_content_type = 'text/markdown',
    author               = 'Mark Ivanov',
    author_email         = 'markmipt@gmail.com',
    install_requires     = [line.strip() for line in open('requirements.txt')],
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
