// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cstddef>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <oneapi/tbb/global_control.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <heyoka/expression.hpp>

#include <cascade/sim.hpp>

#include "logging.hpp"

namespace cascade_py::detail
{

namespace
{

std::optional<oneapi::tbb::global_control> tbb_gc;

}

} // namespace cascade_py::detail

PYBIND11_MODULE(core, m)
{
    namespace py = pybind11;
    using namespace pybind11::literals;

    namespace hy = heyoka;

    using namespace cascade;
    namespace cpy = cascade_py;

    // Expose the logging setter functions.
    cpy::expose_logging_setters(m);

    // outcome enum.
    py::enum_<outcome>(m, "outcome")
        .value("success", outcome::success)
        .value("time_limit", outcome::time_limit)
        .value("interrupt", outcome::interrupt);

    // Dynamics submodule.
    auto dynamics_module = m.def_submodule("dynamics");
    dynamics_module.def("kepler", &dynamics::kepler, "mu"_a = 1.);

    // sim class.
    py::class_<sim>(m, "sim", py::dynamic_attr{})
        .def(py::init<>())
        .def(py::init([](std::vector<double> x, std::vector<double> y, std::vector<double> z, std::vector<double> vx,
                         std::vector<double> vy, std::vector<double> vz, std::vector<double> sizes, double ct,
                         std::vector<std::pair<hy::expression, hy::expression>> dyn) {
                 // NOTE: might have to re-check this if we ever offer the
                 // option to define event callbacks in the dynamics.
                 py::gil_scoped_release release;

                 return sim(std::move(x), std::move(y), std::move(z), std::move(vx), std::move(vy), std::move(vz),
                            std::move(sizes), ct, kw::dyn = std::move(dyn));
             }),
             "x"_a, "y"_a, "z"_a, "vx"_a, "vy"_a, "vz"_a, "sizes"_a, "ct"_a, "dyn"_a = py::list{})
        .def_property_readonly("interrupt_info", &sim::get_interrupt_info)
        .def_property("time", &sim::get_time, &sim::set_time)
        .def_property("ct", &sim::get_ct, &sim::set_ct)
        .def(
            "step",
            [](sim &s, double dt) {
                // NOTE: might have to re-check this if we ever offer the
                // option to define event callbacks in the dynamics.
                py::gil_scoped_release release;

                return s.step(dt);
            },
            "dt"_a = 0.)
        .def(
            "propagate_until",
            [](sim &s, double t, double dt) {
                // NOTE: might have to re-check this if we ever offer the
                // option to define event callbacks in the dynamics.
                py::gil_scoped_release release;

                return s.propagate_until(t, dt);
            },
            "t"_a, "dt"_a = 0.)
        // Expose the state getters.
        .def_property_readonly("x",
                               [](const sim &s) {
                                   auto ret = py::array_t<double>(
                                       py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(s.get_nparts())},
                                       s.get_x().data());

                                   // Ensure the returned array is read-only.
                                   ret.attr("flags").attr("writeable") = false;

                                   return ret;
                               })
        .def_property_readonly("y",
                               [](const sim &s) {
                                   auto ret = py::array_t<double>(
                                       py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(s.get_nparts())},
                                       s.get_y().data());

                                   // Ensure the returned array is read-only.
                                   ret.attr("flags").attr("writeable") = false;

                                   return ret;
                               })
        .def_property_readonly("z",
                               [](const sim &s) {
                                   auto ret = py::array_t<double>(
                                       py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(s.get_nparts())},
                                       s.get_z().data());

                                   // Ensure the returned array is read-only.
                                   ret.attr("flags").attr("writeable") = false;

                                   return ret;
                               })
        .def_property_readonly("vx",
                               [](const sim &s) {
                                   auto ret = py::array_t<double>(
                                       py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(s.get_nparts())},
                                       s.get_vx().data());

                                   // Ensure the returned array is read-only.
                                   ret.attr("flags").attr("writeable") = false;

                                   return ret;
                               })
        .def_property_readonly("vy",
                               [](const sim &s) {
                                   auto ret = py::array_t<double>(
                                       py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(s.get_nparts())},
                                       s.get_vy().data());

                                   // Ensure the returned array is read-only.
                                   ret.attr("flags").attr("writeable") = false;

                                   return ret;
                               })
        .def_property_readonly("vz",
                               [](const sim &s) {
                                   auto ret = py::array_t<double>(
                                       py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(s.get_nparts())},
                                       s.get_vz().data());

                                   // Ensure the returned array is read-only.
                                   ret.attr("flags").attr("writeable") = false;

                                   return ret;
                               })
        .def_property_readonly("sizes",
                               [](const sim &s) {
                                   auto ret = py::array_t<double>(
                                       py::array::ShapeContainer{boost::numeric_cast<py::ssize_t>(s.get_nparts())},
                                       s.get_sizes().data());

                                   // Ensure the returned array is read-only.
                                   ret.attr("flags").attr("writeable") = false;

                                   return ret;
                               })
        .def("set_new_state",
             [](sim &s, std::vector<double> x, std::vector<double> y, std::vector<double> z, std::vector<double> vx,
                std::vector<double> vy, std::vector<double> vz, std::vector<double> sizes) {
                 py::gil_scoped_release release;

                 s.set_new_state(std::move(x), std::move(y), std::move(z), std::move(vx), std::move(vy), std::move(vz),
                                 std::move(sizes));
             })
        // Repr.
        .def("__repr__", [](const sim &s) {
            std::ostringstream oss;
            oss << s;
            return oss.str();
        });

    m.def("set_nthreads", [](std::size_t n) {
        if (n == 0u) {
            cpy::detail::tbb_gc.reset();
        } else {
            cpy::detail::tbb_gc.emplace(oneapi::tbb::global_control::max_allowed_parallelism, n);
        }
    });

    m.def("get_nthreads", []() {
        return oneapi::tbb::global_control::active_value(oneapi::tbb::global_control::max_allowed_parallelism);
    });

    // Make sure the TBB control structure is cleaned
    // up before shutdown.
    auto atexit = py::module_::import("atexit");
    atexit.attr("register")(py::cpp_function([]() {
#if !defined(NDEBUG)
        std::cout << "Cleaning up the TBB control structure" << std::endl;
#endif
        cpy::detail::tbb_gc.reset();
    }));
}
