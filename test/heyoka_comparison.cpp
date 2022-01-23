// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <initializer_list>
#include <random>
#include <utility>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <heyoka/expression.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/math/sum_sq.hpp>
#include <heyoka/taylor.hpp>

#include <cascade/sim.hpp>

#include "catch.hpp"
#include "keputils.hpp"

using namespace cascade;
using namespace cascade_test;
namespace hy = heyoka;

// Compare the numerical integration with heyoka.
TEST_CASE("heyoka comparison")
{
    using Catch::Detail::Approx;

    std::mt19937 rng;

    // Create random particles.
    std::uniform_real_distribution<double> a_dist(1.02, 1.3), e_dist(0., 0.02), i_dist(0., 0.05),
        ang_dist(0., 2 * boost::math::constants::pi<double>());

    std::vector<double> xv, yv, zv, vxv, vyv, vzv, sizev;

    const auto nparts = 10ull;

    for (auto i = 0ull; i < nparts; ++i) {
        const auto a = a_dist(rng);
        const auto e = e_dist(rng);
        const auto inc = i_dist(rng);
        const auto om = ang_dist(rng);
        const auto Om = ang_dist(rng);
        const auto nu = ang_dist(rng);

        auto [r, v] = kep_to_cart<double>({a, e, inc, om, Om, nu}, 1.);

        xv.push_back(r[0]);
        yv.push_back(r[1]);
        zv.push_back(r[2]);

        vxv.push_back(v[0]);
        vyv.push_back(v[1]);
        vzv.push_back(v[2]);

        sizev.push_back(0.);
    }

    sim s(xv, yv, zv, vxv, vyv, vzv, sizev, 0.23);

    for (auto i = 0; i < 10; ++i) {
        REQUIRE(s.step() == outcome::success);
    }

    auto [x, y, z, vx, vy, vz, r] = hy::make_vars("x", "y", "z", "vx", "vy", "vz", "r");

    const auto dynamics = std::vector<std::pair<hy::expression, hy::expression>>{
        hy::prime(x) = vx,
        hy::prime(y) = vy,
        hy::prime(z) = vz,
        hy::prime(vx) = -x * hy::pow(hy::sum_sq({x, y, z}), -1.5),
        hy::prime(vy) = -y * hy::pow(hy::sum_sq({x, y, z}), -1.5),
        hy::prime(vz) = -z * hy::pow(hy::sum_sq({x, y, z}), -1.5)};

    hy::taylor_adaptive<double> ta(dynamics, std::vector<double>(6u));

    for (auto i = 0ull; i < nparts; ++i) {
        ta.get_state_data()[0] = xv[i];
        ta.get_state_data()[1] = yv[i];
        ta.get_state_data()[2] = zv[i];
        ta.get_state_data()[3] = vxv[i];
        ta.get_state_data()[4] = vyv[i];
        ta.get_state_data()[5] = vzv[i];

        ta.set_time(0);

        ta.propagate_until(s.get_time());

        REQUIRE(ta.get_state()[0] == Approx(s.get_x()[i]).epsilon(0.).margin(1e-13));
        REQUIRE(ta.get_state()[1] == Approx(s.get_y()[i]).epsilon(0.).margin(1e-13));
        REQUIRE(ta.get_state()[2] == Approx(s.get_z()[i]).epsilon(0.).margin(1e-13));
        REQUIRE(ta.get_state()[3] == Approx(s.get_vx()[i]).epsilon(0.).margin(1e-13));
        REQUIRE(ta.get_state()[4] == Approx(s.get_vy()[i]).epsilon(0.).margin(1e-13));
        REQUIRE(ta.get_state()[5] == Approx(s.get_vz()[i]).epsilon(0.).margin(1e-13));
    }
}
