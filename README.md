cascade
=========

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/esa/cascade">
    <img src="doc/_static/images/logo.png" alt="Logo" width="280">
  </a>
  <p align="center">
    Long term evolution of orbital environments.
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

cascade is a Python library ...

Installation
===============
Install conda env:

```
conda env create -f environment.yml
conda activate cascade
```

Compile:
```
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DBoost_NO_BOOST_CMAKE=ON -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python -DCASCADE_INSTALL_LIBDIR=lib -DCASCADE_BUILD_PYTHON_BINDINGS=yes -DCMAKE_BUILD_TYPE=Debug
make install
```
