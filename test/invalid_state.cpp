// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include <cascade/sim.hpp>

#include "catch.hpp"
#include "keputils.hpp"

using namespace cascade;
using namespace cascade_test;

// Test to check the behaviour when a simulation contains
// invalid particle states.
TEST_CASE("invalid state")
{
    using Catch::Matchers::Message;

    const auto psize = 1.57e-8;
    const auto [x1, v1] = kep_to_cart<double>({1.05, 0., 0., 0., 0., 0}, 1);
    auto x2 = x1;
    x2[0] = -x2[0];
    auto v2 = v1;

    const auto s0
        = std::vector{x1[0], x1[1], x1[2], v1[0], v1[1], v1[2], psize, x2[0], x2[1], x2[2], v2[0], v2[1], v2[2], psize};

    sim s(s0, 0.23);

    auto sv = xt::adapt(s.get_state_data(), {2, 7});

    // Set invalid p radius.
    sv(0, 6) = -1;

    REQUIRE_THROWS_MATCHES(s.step(), std::invalid_argument,
                           Message("An invalid particle size of -1 was detected for the particle at index 0"));

    sv(0, 6) = psize;

    REQUIRE(s.get_state() == s0);
    REQUIRE(s.get_time() == 0.);

    sv(0, 0) = std::numeric_limits<double>::infinity();

    REQUIRE(s.step() == outcome::err_nf_state);

    sv(0, 0) = x1[0];

    REQUIRE(s.get_state() == s0);
    REQUIRE(s.get_time() == 0.);
}
