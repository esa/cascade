.. cascade documentation master file, created by
   sphinx-quickstart on Fri Jan 20 17:24:51 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Cascade
===================================
   As the number of artificial satellites orbiting our planet grows, the likelihood of collisions increases, which can create even more debris and
   potentially render certain orbits unusable.

Cascade is a Python library developed to propagate the evolution of large number of orbiting objects while detecting reliably close encounters and collisions. 

Notable features include:

- guaranteed detection of all occuring collisions.
- high precision orbital propagation via the use of Taylor integration.
- possibility to define custom dynamics.
- automatic usage of modern SIMD instruction sets (including AVX/AVX2/AVX-512/Neon/VSX).
- automatic multi-threaded parallelisation.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install.rst

