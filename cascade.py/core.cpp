// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <sstream>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cascade/sim.hpp>

#include "logging.hpp"

PYBIND11_MODULE(core, m)
{
    namespace py = pybind11;
    using namespace pybind11::literals;

    using namespace cascade;
    namespace cpy = cascade_py;

    // Connect cascade's logging to Python's logging.
    cpy::enable_logging();

    // Expose the logging setter functions.
    cpy::expose_logging_setters(m);

    // outcome enum.
    py::enum_<outcome>(m, "outcome")
        .value("success", outcome::success)
        .value("time_limit", outcome::time_limit)
        .value("interrupt", outcome::interrupt);

    // sim class.
    py::class_<sim>(m, "sim", py::dynamic_attr{})
        .def(py::init<>())
        .def(py::init<std::vector<double>, std::vector<double>, std::vector<double>, std::vector<double>,
                      std::vector<double>, std::vector<double>, std::vector<double>, double>())
        .def_property_readonly("interrupt_info", &sim::get_interrupt_info)
        .def_property("time", &sim::get_time, &sim::set_time)
        .def_property("ct", &sim::get_ct, &sim::set_ct)
        .def(
            "step", [](sim &s, double dt) { s.step(dt); }, "dt"_a = 0.)
        .def(
            "propagate_until", [](sim &s, double t, double dt) { s.propagate_until(t, dt); }, "t"_a, "dt"_a = 0.)
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
                 s.set_new_state(std::move(x), std::move(y), std::move(z), std::move(vx), std::move(vy), std::move(vz),
                                 std::move(sizes));
             })
        // Repr.
        .def("__repr__", [](const sim &s) {
            std::ostringstream oss;
            oss << s;
            return oss.str();
        });
}
