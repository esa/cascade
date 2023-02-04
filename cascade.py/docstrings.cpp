// Copyright 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <string>

#include "docstrings.hpp"

namespace cascade_py::docstrings
{

// __init__(state: numpy.ndarray[numpy.double] = numpy.array([], shape=(0, 7), dtype=float64), ct: float = 1.0, dyn:
// typing.Optional[typing.List[typing.Tuple[heyoka.core.expression, heyoka.core.expression]]] = None, reentry_radius:
// typing.Optional[float | typing.List[float]] = None, exit_radius: Optional[float] = None, pars:
// Optional[numpy.ndarray[numpy.float64]] = None, tol: Optional[float] = None, high_accuracy: bool = False, n_par_ct:
// int = 1, conj_thresh: float = 0.0, min_coll_radius: float = 0.0, coll_whitelist: Set[int] = set(), conj_whitelist:
// typing.Set[int] = set())

std::string sim_docstring()
{
    return R"(The simulation class

This is the class that....

)";
}

std::string sim_init_docstring()
{
    return R"(__init__(state: numpy.ndarray[numpy.double], ct: float, **kwargs)

Constructor

More details here...

Parameters
----------

state: numpy.ndarray[numpy.double]
    The initial state vector for the simulation
ct: float
    The number of digits for printing fraction of seconds

Other Parameters
----------------

dyn: typing.Optional[typing.List[typing.Tuple[heyoka.core.expression, heyoka.core.expression]]] = None
    sadas
reentry_radius: typing.Optional[float | typing.List[float]] = None
    asdd

)";
}

} // namespace cascade_py::docstrings
