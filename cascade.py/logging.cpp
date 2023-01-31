// Copyright 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <pybind11/pybind11.h>

#include <cascade/logging.hpp>

#include "logging.hpp"

namespace cascade_py
{

void expose_logging_setters(py::module_ &m)
{
    namespace csc = cascade;

    m.def("set_logger_level_trace", &csc::set_logger_level_trace);
    m.def("set_logger_level_debug", &csc::set_logger_level_debug);
    m.def("set_logger_level_info", &csc::set_logger_level_info);
    m.def("set_logger_level_warn", &csc::set_logger_level_warn);
    m.def("set_logger_level_err", &csc::set_logger_level_err);
    m.def("set_logger_level_critical", &csc::set_logger_level_critical);
}

} // namespace cascade_py
