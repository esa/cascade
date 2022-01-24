// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CASCADE_DETAIL_LOGGING_IMPL_HPP
#define CASCADE_DETAIL_LOGGING_IMPL_HPP

#if !defined(NDEBUG)

// NOTE: this means that in release builds all SPDLOG_LOGGER_DEBUG() calls
// will be elided (so that they won't show up in the log even if
// the log level is set to spdlog::level::debug).
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG

#endif

#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

namespace cascade::detail
{

spdlog::logger *get_logger();

} // namespace cascade::detail

#endif
