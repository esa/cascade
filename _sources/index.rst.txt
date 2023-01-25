.. cascade documentation master file, created by
   sphinx-quickstart on Fri Jan 20 17:24:51 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Cascade
===================================
   As the number of artificial satellites orbiting our planet grows, the likelihood of collisions increases,
   potentially leading to a chain reaction that could make certain orbits unusable.

Cascade is a Python library developed to propagate the evolution of large number of orbiting objects while detecting
reliably close encounters and collisions. It is coded in modern C++20 with focus on the efficency of the underlying N-body 
simulation with collision detection. 

Notable features include:

- guaranteed detection of all occuring collisions.
- high precision orbital propagation via Taylor integration.
- possibility to define custom dynamics.
- automatic usage of modern SIMD instruction sets (including AVX/AVX2/AVX-512/Neon/VSX).
- automatic multi-threaded parallelisation.

cascade is released under the MPL-2.0 license. The authors are Francesco Biscani and Dario Izzo (European Space Agency).

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   install.rst

.. toctree::
   :maxdepth: 1
   :caption: Utilities:

   utilities/leo_population
   utilities/simple_atmosphere
   utilities/cubes



