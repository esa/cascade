// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <random>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <cascade/sim.hpp>

#include "catch.hpp"
#include "keputils.hpp"

using namespace cascade;
using namespace cascade_test;

TEST_CASE("particle removal")
{
    std::mt19937 rng;

    // Create random particles.
    std::uniform_real_distribution<double> a_dist(1.02, 1.3), e_dist(0., 0.02), i_dist(0., 0.05),
        ang_dist(0., 2 * boost::math::constants::pi<double>());

    std::vector<double> state;

    const auto nparts = 100ull;

    for (auto i = 0ull; i < nparts; ++i) {
        const auto a = a_dist(rng);
        const auto e = e_dist(rng);
        const auto inc = i_dist(rng);
        const auto om = ang_dist(rng);
        const auto Om = ang_dist(rng);
        const auto nu = ang_dist(rng);

        auto [r, v] = kep_to_cart<double>({a, e, inc, om, Om, nu}, 1.);

        state.push_back(r[0]);
        state.push_back(r[1]);
        state.push_back(r[2]);

        state.push_back(v[0]);
        state.push_back(v[1]);
        state.push_back(v[2]);

        state.push_back(0.);
    }

    sim s(state, 0.23);
    REQUIRE(s.get_nparts() == 100u);

    for (auto i = 0; i < 10; ++i) {
        REQUIRE(s.step() == outcome::success);
    }

    // Remove half of the paricles.
    state.resize(state.size() / 2u);

    s.set_state(state);
    REQUIRE(s.get_nparts() == 50u);

    for (auto i = 0; i < 10; ++i) {
        REQUIRE(s.step() == outcome::success);
        REQUIRE(s.get_nparts() == 50u);
    }
}
