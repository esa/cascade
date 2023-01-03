// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <tuple>
#include <variant>
#include <vector>

#include <cascade/sim.hpp>

#include "catch.hpp"

using namespace cascade;

TEST_CASE("nf state step")
{
    sim s({1.01, 0, 0, -100., 0, 0, 0, -1.01, 0, 0, 99., 0, 0, 0}, 0.23);

    auto oc = s.step(10.);

    REQUIRE(oc == outcome::err_nf_state);
    REQUIRE(std::get<0>(std::get<2>(*s.get_interrupt_info())) == 0u);
    REQUIRE(std::get<1>(std::get<2>(*s.get_interrupt_info())) != 0);

    // Ensure correctness also when using the batch integrator.
    s.set_new_state({1.01, 0, 0, -100., 0, 0, 0,    -1.01, 0, 0, 99.,  0, 0, 0,    1.02, 0, 0, 1.02,
                     0,    0, 0, 1.03,  0, 0, 1.03, 0,     0, 0, 1.04, 0, 0, 1.04, 0,    0, 0});
    s.set_time(0.);

    oc = s.step(10.);

    REQUIRE(oc == outcome::err_nf_state);
    REQUIRE(std::get<0>(std::get<2>(*s.get_interrupt_info())) == 0u);
    REQUIRE(std::get<1>(std::get<2>(*s.get_interrupt_info())) != 0);
}

TEST_CASE("nf state propagate")
{
    sim s({1.01, 0, 0, -100., 0, 0, 0, -1.01, 0, 0, 101., 0, 0, 0}, 0.23);

    auto oc = s.propagate_until(1000., 10.);

    REQUIRE(oc == outcome::err_nf_state);
    REQUIRE(std::get<0>(std::get<2>(*s.get_interrupt_info())) == 1u);
    REQUIRE(std::get<1>(std::get<2>(*s.get_interrupt_info())) != 0);

    // Ensure correctness also when using the batch integrator.
    s.set_new_state({1.01, 0, 0, -100., 0, 0, 0,    -1.01, 0, 0, 101., 0, 0, 0,    1.02, 0, 0, 1.02,
                     0,    0, 0, 1.03,  0, 0, 1.03, 0,     0, 0, 1.04, 0, 0, 1.04, 0,    0, 0});
    s.set_time(0.);

    oc = s.propagate_until(1000., 10.);

    REQUIRE(oc == outcome::err_nf_state);
    // NOTE: here the index is zero instead of 1 because both errors
    // occurr within the same step, but the error for particle 0 is recorded
    // *before* particle 1 in the status vector. Thus, all else being equa,
    // std::min_element will return index 0.
    REQUIRE(std::get<0>(std::get<2>(*s.get_interrupt_info())) == 0u);
    REQUIRE(std::get<1>(std::get<2>(*s.get_interrupt_info())) != 0);
}
