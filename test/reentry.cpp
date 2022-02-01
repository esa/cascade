// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>
#include <vector>

#include <cascade/sim.hpp>

#include "catch.hpp"

using namespace cascade;

TEST_CASE("reentry sphere")
{
    using Catch::Detail::Approx;

    sim s(std::vector<double>{1.5}, std::vector<double>{0}, std::vector<double>{0}, std::vector<double>{0},
          std::vector<double>{0.01}, std::vector<double>{0.}, std::vector<double>{0}, 0.23, kw::c_radius = 1.);

    auto oc = s.propagate_until(1000.);

    REQUIRE(oc == outcome::reentry);

    auto x = s.get_x()[0];
    auto y = s.get_y()[0];
    auto z = s.get_z()[0];

    REQUIRE(std::sqrt(x * x + y * y + z * z) == Approx(1.).epsilon(0.).margin(1e-14));

    s.set_new_state(std::vector<double>{0.}, std::vector<double>{1.5}, std::vector<double>{0}, std::vector<double>{0},
                    std::vector<double>{0.}, std::vector<double>{0.}, std::vector<double>{0});
    s.set_time(0.);

    oc = s.propagate_until(1000, 0.1);

    REQUIRE(oc == outcome::reentry);

    x = s.get_x()[0];
    y = s.get_y()[0];
    z = s.get_z()[0];

    REQUIRE(std::sqrt(x * x + y * y + z * z) == Approx(1.).epsilon(0.).margin(1e-14));
}

TEST_CASE("reentry ellipsoid")
{
    using Catch::Detail::Approx;

    sim s(std::vector<double>{1.5}, std::vector<double>{0}, std::vector<double>{0}, std::vector<double>{0},
          std::vector<double>{0.0}, std::vector<double>{0.}, std::vector<double>{0}, 0.23,
          kw::c_radius = std::vector<double>{1., 1.1, 1.2});

    auto oc = s.propagate_until(1000., 0.1);

    REQUIRE(oc == outcome::reentry);

    auto x = s.get_x()[0];
    auto y = s.get_y()[0];
    auto z = s.get_z()[0];

    REQUIRE(std::sqrt(x * x + y * y + z * z) == Approx(1.).epsilon(0.).margin(1e-14));

    s.set_new_state(std::vector<double>{0.}, std::vector<double>{1.5}, std::vector<double>{0}, std::vector<double>{0},
                    std::vector<double>{0.}, std::vector<double>{0.}, std::vector<double>{0});
    s.set_time(0.);

    oc = s.propagate_until(1000, 0.1);

    REQUIRE(oc == outcome::reentry);

    x = s.get_x()[0];
    y = s.get_y()[0];
    z = s.get_z()[0];

    REQUIRE(std::sqrt(x * x + y * y + z * z) == Approx(1.1).epsilon(0.).margin(1e-14));

    s.set_new_state(std::vector<double>{0.}, std::vector<double>{0.}, std::vector<double>{1.5}, std::vector<double>{0},
                    std::vector<double>{0.}, std::vector<double>{0.}, std::vector<double>{0});
    s.set_time(0.);

    oc = s.propagate_until(1000, 0.1);

    REQUIRE(oc == outcome::reentry);

    x = s.get_x()[0];
    y = s.get_y()[0];
    z = s.get_z()[0];

    REQUIRE(std::sqrt(x * x + y * y + z * z) == Approx(1.2).epsilon(0.).margin(1e-14));
}
