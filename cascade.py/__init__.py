# Copyright 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the cascade.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

""":mod:`cascade`
=================

Classes
-------

.. autosummary::
    :toctree: generated/
    :template: custom-class-template.rst

    sim
    outcome
    set_nthreads
    get_nthreads
    set_logger_level_trace
    set_logger_level_debug
    set_logger_level_info
    set_logger_level_warn
    set_logger_level_err
    set_logger_level_critical

"""

# Version setup.
from ._version import __version__

# NOTE: import heyoka to ensure that
# cascade knows how to convert to/from
# expressions.
import heyoka as _hy

# We import core into the root namespace.
from .core import *

del _hy

# We import the sub-modules.
import cascade.dynamics
import cascade.test
