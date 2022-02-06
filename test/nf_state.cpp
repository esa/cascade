// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <variant>
#include <vector>

#include <cascade/sim.hpp>

#include "catch.hpp"

using namespace cascade;

TEST_CASE("nf state step")
{
    sim s(std::vector<double>{1.01, -1.01}, std::vector<double>{0, 0}, std::vector<double>{0, 0},
          std::vector<double>{-100., 99.}, std::vector<double>{0.0, 0}, std::vector<double>{0., 0},
          std::vector<double>{0, 0}, 0.23, kw::c_radius = 1.);

    auto oc = s.step(10.);

    REQUIRE(oc == outcome::err_nf_state);
    REQUIRE(std::get<1>(*s.get_interrupt_info()) == 0u);
}

TEST_CASE("nf state propagate")
{
    sim s(std::vector<double>{1.01, -1.01}, std::vector<double>{0, 0}, std::vector<double>{0, 0},
          std::vector<double>{-100., 101}, std::vector<double>{0.0, 0}, std::vector<double>{0., 0},
          std::vector<double>{0, 0}, 0.23, kw::c_radius = 1.);

    auto oc = s.propagate_until(1000., 10.);

    REQUIRE(oc == outcome::err_nf_state);
    REQUIRE(std::get<1>(*s.get_interrupt_info()) == 1u);
}
