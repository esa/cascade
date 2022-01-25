// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CASCADE_PY_LOGGING_HPP
#define CASCADE_PY_LOGGING_HPP

#include <pybind11/pybind11.h>

namespace cascade_py
{

namespace py = pybind11;

void expose_logging_setters(py::module_ &);

} // namespace cascade_py

#endif
