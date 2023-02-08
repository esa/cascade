.. cascade documentation master file, created by
   sphinx-quickstart on Fri Jan 20 17:24:51 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

cascade
=======

   As the number of artificial satellites orbiting our planet grows, the likelihood of collisions increases,
   potentially leading to a chain reaction that could make certain orbits unusable.

cascade is a C++/Python library developed to propagate the evolution of large number of orbiting objects while detecting
reliably close encounters and collisions. It is coded in modern C++20 with focus on the efficency of the underlying N-body 
simulation with collision/conjunction detection. Its development was motivated to help conjunction tracking and
collision detection of orbiting space debris populations.

Notable features include:

- guaranteed detection of all occuring collisions and conjunctions,
- high precision orbital propagation via Taylor integration,
- possibility to define custom dynamics,
- automatic usage of modern SIMD instruction sets (including AVX/AVX2/AVX-512/Neon/VSX),
- automatic multi-threaded parallelisation.

cascade is released under the MPL-2.0 license. The authors are Francesco Biscani and Dario Izzo (European Space Agency).

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Contents:

   install.rst
   user_guide.rst
   api_reference.rst

.. grid::

    .. grid-item-card::
        :img-top: _static/images/getting_started.svg
        :text-align: center
        :padding: 2 2 2 2

        **Installation**

        How to install the software and how to get started.

        +++

        .. button-ref:: installation
            :ref-type: ref
            :expand:
            :color: secondary
            :click-parent:

            To installation

    .. grid-item-card::
        :img-top: _static/images/user_guide.svg
        :text-align: center
        :padding: 2 2 2 2

        **User Guide**

        TODO what goes here?

        +++

        .. button-ref:: user_guide
            :ref-type: ref
            :expand:
            :color: secondary
            :click-parent:

            To the User Guide

    .. grid-item-card::
        :img-top: _static/images/api.svg
        :text-align: center
        :padding: 2 2 2 2

        **API Reference**

        The API reference contains a full description of the functions, classes and modules in cascade.

        +++

        .. button-ref:: api_reference
            :ref-type: ref
            :expand:
            :color: secondary
            :click-parent:

            To the API reference

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:
   
   tutorials/quickstart
   tutorials/conjandcoll

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   examples/20yearsofLEO_coll
   examples/oneweekofLEO_conj
   examples/itokawa_cubesats

.. toctree::
   :maxdepth: 1
   :caption: Utilities:

   utilities/leo_population
   utilities/simple_atmosphere
   utilities/cubes



