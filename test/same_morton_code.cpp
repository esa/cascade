// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <cascade/sim.hpp>

#include "catch.hpp"
#include "keputils.hpp"

using namespace cascade;
using namespace cascade_test;

// Simulation in which some particles end up having the same
// Morton code.
TEST_CASE("same morton code")
{
    std::mt19937 rng;
    const auto seed = std::random_device{}();
    rng.seed(seed);
    std::cout << "Seed set to: " << seed << '\n';

    // Create random particles.
    std::uniform_real_distribution<double> a_dist(1.02, 1.3), e_dist(0., 0.02), i_dist(0., 0.05),
        ang_dist(0., 2 * boost::math::constants::pi<double>());

    std::vector<double> x, y, z, vx, vy, vz, size;

    const auto nparts = 2000ull;

    for (auto i = 0ull; i < nparts; ++i) {
        const auto a = a_dist(rng);
        const auto e = e_dist(rng);
        const auto inc = i_dist(rng);
        const auto om = ang_dist(rng);
        const auto Om = ang_dist(rng);
        const auto nu = ang_dist(rng);

        auto [r, v] = kep_to_cart<double>({a, e, inc, om, Om, nu}, 1.);

        x.push_back(r[0]);
        y.push_back(r[1]);
        z.push_back(r[2]);

        vx.push_back(v[0]);
        vy.push_back(v[1]);
        vz.push_back(v[2]);

        size.push_back(0.);
    }

    // Add a couple of particles very close to other particles in the simulation.
    for (auto idx : {1u, 123u, 456u}) {
        x.push_back(x[idx] + std::numeric_limits<double>::epsilon() * 1000);
        y.push_back(y[idx] + std::numeric_limits<double>::epsilon() * 1000);
        z.push_back(z[idx] + std::numeric_limits<double>::epsilon() * 1000);

        vx.push_back(vx[idx]);
        vy.push_back(vy[idx]);
        vz.push_back(vz[idx]);

        size.push_back(0.);
    }

    sim s(x, y, z, vx, vy, vz, size, 0.23);

    REQUIRE(s.step() == outcome::success);
}
