import os
from setuptools import setup
from setuptools.dist import Distribution
import sys

NAME = "esa_cascade"
VERSION = "@cascade_VERSION@"
DESCRIPTION = "N-body simulation for the evolution of orbital environments"
LONG_DESCRIPTION = "A Python library to propagate the evolution of a large number of orbiting objects while detecting reliably close encounters and collisions."
URL = "https://github.com/esa/cascade"
AUTHOR = "Francesco Biscani, Dario Izzo"
AUTHOR_EMAIL = "bluescarni@gmail.com"
LICENSE = "MPL-2.0"
CLASSIFIERS = [
    # How mature is this project? Common values are
    #   3 - Alpha
    #   4 - Beta
    #   5 - Production/Stable
    "Development Status :: 5 - Production/Stable",
    "Operating System :: OS Independent",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Scientific/Engineering :: Physics",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
    "Programming Language :: Python :: 3",
]
KEYWORDS = "science math physics ode"
INSTALL_REQUIRES = ["heyoka==@heyoka_VERSION_MAJOR@.@heyoka_VERSION_MINOR@.*", "numpy"]
PLATFORMS = ["Unix"]


class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    platforms=PLATFORMS,
    install_requires=INSTALL_REQUIRES,
    packages=["cascade", "cascade.dynamics"],
    # Include pre-compiled extension
    package_data={"cascade": [f for f in os.listdir("cascade/") if f.endswith(".so")]},
    distclass=BinaryDistribution,
)
