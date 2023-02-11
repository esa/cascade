cascade
=========

[![Code Coverage](https://img.shields.io/codecov/c/github/esa/cascade.svg?style=for-the-badge)](https://codecov.io/github/esa/cascade?branch=main)

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/esa/cascade">
    <img src="doc/_static/images/logo.png" alt="An illustration of the Earth surrounded by space debris; the Earth has a face on it with a worried look." width="280">
  </a>
  <p align="center">
    n-body simulation for the evolution of orbital environments
    <br />
    <a href="https://esa.github.io/cascade/index.html"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/esa/cascade/issues/new/choose">Report bug</a>
    ·
    <a href="https://github.com/esa/cascade/issues/new/choose">Request feature</a>
    ·
    <a href="https://github.com/esa/cascade/discussions">Discuss</a>
  </p>
</p>

Cascade is a C++/Python library developed to propagate the evolution of large
number of orbiting objects while detecting reliably close encounters and
collisions. It is coded in modern C++20 with focus on the efficiency of the
underlying N-body simulation with collision/conjunction detection.

Installation
===============
Install conda env:

```
conda env create -f cascade_devel.yml
conda activate cascade_devel
```

Compile:
```
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DBoost_NO_BOOST_CMAKE=ON -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python -DCASCADE_INSTALL_LIBDIR=lib -DCASCADE_BUILD_PYTHON_BINDINGS=yes -DCMAKE_BUILD_TYPE=Debug
make install
```
