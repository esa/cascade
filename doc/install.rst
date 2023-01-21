Installation
============

Installing from a package manager
---------------------------------

Conda
^^^^^

cascade is available via the `conda <https://docs.conda.io/en/latest/>`__
package manager for Linux, OSX and Windows
thanks to the infrastructure provided by `conda-forge <https://conda-forge.org/>`__.
In order to install cascade via conda, you need to add ``conda-forge``
to the channels. Assuming a working installation of conda:

.. code-block:: console

   $ conda config --add channels conda-forge
   $ conda config --set channel_priority strict
   $ conda install cascade

.. note::
    It is useful to check the scripts we use to build and test cascade since they use conda. You can find the ones for OSX and linux in
    `the tool folder <https://github.com/esa/cascade/tree/main/tools>`__ of our github repository.

The conda packages for cascade are maintained by the core development team,
and they are regularly updated when new cascade versions are released.

Please refer to the `conda documentation <https://docs.conda.io/en/latest/>`__
for instructions on how to setup and manage
your conda installation.


Installation from source
------------------------

cascade is written in modern C++, and it requires a compiler able to understand
many new idioms introduced in C++21. The library is regularly tested on
a continuous integration pipeline which currently includes:

* GCC 12 on Linux,
* Clang 11 on OSX,
* MSVC 2019 on Windows.

Its a good idea to check the scripts used to build cascade on these architectures, you can find the ones for OSX and linux in
`the tool folder <https://github.com/esa/cascade/tree/main/tools>`__ of our github repository.

cascade has several Python and C++ dependencies. On the C++ side, cascade depends on:

* the `heyoka C++ library <https://github.com/bluescarni/heyoka>`__,
  version 0.20.1 or later (**mandatory**),
* the `Boost <https://www.boost.org/>`__ C++ libraries version 1.73 or later (**mandatory**),
* the `{fmt} <https://fmt.dev/latest/index.html>`__ library (**mandatory**),
* the `TBB <https://github.com/oneapi-src/oneTBB>`__ library (**mandatory**),
* the `spdlog <https://github.com/gabime/spdlog>`__ library (**mandatory**).

On the Python side, cascade requires at least Python 3.5
(Python 2.x is **not** supported) and depends on:

* `NumPy <https://numpy.org/>`__ (**mandatory**),

The tested and supported CPU architectures at this time are x86-64.

In addition to the C++ dependencies enumerated :ref:`earlier <installation_deps>`,
installation from source requires also:

* `pybind11 <https://github.com/pybind/pybind11>`__ (version >= 2.10),
* `CMake <https://cmake.org/>`__, version 3.19 or later.

After making sure the dependencies are installed on your system, you can
download the cascade source code from the
`GitHub release page <https://github.com/esa/cascade/releases>`__. Alternatively,
and if you like living on the bleeding edge, you can get the very latest
version of cascade via ``git``:

.. code-block:: console

   $ git clone https://github.com/esa/cascade.git

We follow the usual PR-based development workflow, thus cascade's ``main``
branch is normally kept in a working state.

After downloading and/or unpacking cascade's
source code, go to cascade's
source tree, create a ``build`` directory and ``cd`` into it. E.g.,
on a Unix-like system:

.. code-block:: console

   $ cd /path/to/cascade
   $ mkdir build
   $ cd build

Once you are in the ``build`` directory, you must configure your build
using ``cmake``. There are various useful CMake variables you can set,
such as:

* ``CMAKE_BUILD_TYPE``: the build type (``Release``, ``Debug``, etc.),
  defaults to ``Release``.
* ``CMAKE_PREFIX_PATH``: additional paths that will be searched by CMake
  when looking for dependencies.

Please consult `CMake's documentation <https://cmake.org/cmake/help/latest/>`_
for more details about CMake's variables and options.

In order for the python module to be built the corresponding option will need to be activated,
else cascade build system will only build the dynamic library, which you can still use from the C++ side
as done by the various benchmarks you can also build and find in the corresponding directory.

* ``CASCADE_BUILD_PYTHON_BINDINGS``: builds also the python module.

After configuring the build with CMake, we can then proceed to actually
building cascade:

.. code-block:: console

   $ cmake --build .

Finally, we can install cascade with the command:

.. code-block:: console

   $ cmake  --build . --target install

Verifying the installation
--------------------------

You can verify that cascade was successfully compiled and
installed by running the test suite with the following command:

.. code-block:: bash

   $ python -c "import cascade; cascade.test.run_test_suite();"

If this command executes without any error, then
your cascade installation is ready for use.

Getting help
------------

If you run into troubles installing cascade, please do not hesitate
to contact us by opening an issue report on `github <https://github.com/esa/cascade/issues>`__.