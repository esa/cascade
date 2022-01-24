// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <memory>
#include <mutex>
#include <utility>

#include <spdlog/sinks/base_sink.h>
#include <spdlog/spdlog.h>

#include <fmt/core.h>

#include <pybind11/pybind11.h>

#include <cascade/logging.hpp>

#include "logging.hpp"

namespace cascade_py
{

namespace py = pybind11;
namespace csc = cascade;

namespace detail
{

namespace
{

template <typename Mutex>
class py_sink : public spdlog::sinks::base_sink<Mutex>
{
protected:
    void sink_it_(const spdlog::details::log_msg &msg) override
    {
        // NOTE: grab the raw message and convert it to string
        // without applying any spdlog-specific formatting.
        const auto str = fmt::to_string(msg.payload);

        // Make sure we lock the GIL before calling into the
        // interpreter, as log messages may be produced from
        // TBB threads.
        py::gil_scoped_acquire acquire;

        // Fetch the Python logger.
        auto py_logger = py::module_::import("logging").attr("getLogger")("cascade");

        switch (msg.level) {
            case spdlog::level::trace:
                [[fallthrough]];
            case spdlog::level::debug:
                py_logger.attr("debug")(str);
                break;
            case spdlog::level::info:
                py_logger.attr("info")(str);
                break;
            case spdlog::level::warn:
                py_logger.attr("warning")(str);
                break;
            case spdlog::level::err:
                py_logger.attr("error")(str);
                break;
            case spdlog::level::critical:
                py_logger.attr("critical")(str);
                break;
            default:;
        }
    }

    void flush_() override {}
};

// Utility helper to synchronize the logging levels
// of the cascade C++ logger and the Python one.
void log_sync_levels()
{
    // Fetch the C++ logger.
    auto logger = spdlog::get("cascade");
    assert(logger);

    // Fetch the Python logger.
    auto log_mod = py::module_::import("logging");
    auto py_logger = log_mod.attr("getLogger")("cascade");

    // Do the matching.
    switch (logger->level()) {
        case spdlog::level::trace:
            [[fallthrough]];
        case spdlog::level::debug:
            py_logger.attr("setLevel")(log_mod.attr("DEBUG"));
            break;
        case spdlog::level::info:
            py_logger.attr("setLevel")(log_mod.attr("INFO"));
            break;
        case spdlog::level::warn:
            py_logger.attr("setLevel")(log_mod.attr("WARNING"));
            break;
        case spdlog::level::err:
            py_logger.attr("setLevel")(log_mod.attr("ERROR"));
            break;
        case spdlog::level::critical:
            py_logger.attr("setLevel")(log_mod.attr("CRITICAL"));
            break;
        default:;
    }
}

} // namespace

} // namespace detail

void enable_logging()
{
    // Force the creation of the cascade logger.
    csc::create_logger();

    // Fetch it.
    auto logger = spdlog::get("cascade");
    assert(logger);

    // Initial creation of the cascade logger on the
    // Python side.
    auto log_mod = py::module_::import("logging");
    auto py_logger = log_mod.attr("getLogger")("cascade");

    // Set the initial logging level.
    detail::log_sync_levels();

    // Add the Python sink to the cascade logger.
    auto sink = std::make_shared<detail::py_sink<std::mutex>>();
    logger->sinks().push_back(std::move(sink));
}

void expose_logging_setters(py::module_ &m)
{
    m.def("set_logger_level_trace", []() {
        csc::set_logger_level_trace();
        detail::log_sync_levels();
    });

    m.def("set_logger_level_debug", []() {
        csc::set_logger_level_debug();
        detail::log_sync_levels();
    });

    m.def("set_logger_level_info", []() {
        csc::set_logger_level_info();
        detail::log_sync_levels();
    });

    m.def("set_logger_level_warning", []() {
        csc::set_logger_level_warn();
        detail::log_sync_levels();
    });

    m.def("set_logger_level_error", []() {
        csc::set_logger_level_err();
        detail::log_sync_levels();
    });

    m.def("set_logger_level_critical", []() {
        csc::set_logger_level_critical();
        detail::log_sync_levels();
    });
}

} // namespace cascade_py
