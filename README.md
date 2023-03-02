cascade
=========

[![Code Coverage](https://img.shields.io/codecov/c/github/esa/cascade.svg?style=for-the-badge)](https://codecov.io/github/esa/cascade?branch=main)

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/esa/cascade">
    <img src="doc/_static/images/logo.png" alt="Logo" width="280">
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

*As the number of artificial satellites orbiting our planet grows, the likelihood of collisions increases, potentially triggering a cascade effect making certain orbits unusable.*

**cascade** is a C++/Python library developed to propagate the evolution of large number of orbiting objects
while detecting reliably close encounters and collisions. It is coded in modern C++20 with focus on the
efficency of the underlying N-body simulation with collision/conjunction detection. Its development was
motivated to help conjunction tracking and collision detection of orbiting space debris populations.

Notable features include:

* an original collision detection algorithm exploiting high order Taylor expansions.
* guaranteed detection of all occuring collisions and conjunctions.
* high precision orbital propagation via Taylor integration.
* possibility to define custom dynamics.
* seamless usage of modern SIMD instruction sets (including AVX/AVX2/AVX-512/Neon/VSX).
* seamless multi-threaded parallelisation.

Installation
------------

Via pip:

```console
$ pip install esa_cascade
```

Via conda + [conda-forge](https://conda-forge.org/):

```console
$ conda install cascade
```
