// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CASCADE_LOGGING_HPP
#define CASCADE_LOGGING_HPP

#include <cascade/detail/visibility.hpp>

namespace cascade
{

CASCADE_DLL_PUBLIC void *create_logger();

CASCADE_DLL_PUBLIC void set_logger_level_trace();
CASCADE_DLL_PUBLIC void set_logger_level_debug();
CASCADE_DLL_PUBLIC void set_logger_level_info();
CASCADE_DLL_PUBLIC void set_logger_level_warn();
CASCADE_DLL_PUBLIC void set_logger_level_err();
CASCADE_DLL_PUBLIC void set_logger_level_critical();

} // namespace cascade

#endif
