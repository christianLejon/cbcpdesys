#!/usr/bin/env python

import os, sys, platform
from setuptools import setup

# Version number
major = 1
minor = 7
maintenance = 0

setup(name = "cbcpdesys",
      version = "%d.%d.%d" % (major, minor, maintenance),
      description = "CBC.PDESys -- Framework for solving systems of PDEs from the Center of Biomedical Computing",
      author = "Mikael Mortensen and Hans Petter Langtangen",
      author_email = "mikaem@math.uio.no", 
      url = 'https://bitbucket.org/simula_cbc/cbcpdesys',
      classifiers = [
          'Development Status :: 5 - Production/Stable',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Intended Audience :: Education',
          'Programming Language :: Python :: 2.7',
          'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
          'Topic :: Scientific/Engineering :: Mathematics',
          'Topic :: Software Development :: Libraries :: Python Modules',
          ],
      packages = ["cbc",
                  "cbc.pdesys",
                  "cbc.demo",
                  "cbc.doc",
                  ],
      package_dir = {"cbcpdesys": "cbcpdesys"},
    )