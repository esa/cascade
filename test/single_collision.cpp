// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <iostream>
#include <limits>
#include <variant>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <boost/math/constants/constants.hpp>

#include <cascade/sim.hpp>

#include "catch.hpp"
#include "keputils.hpp"

using namespace cascade;
using namespace cascade_test;

// Create two particles of 10cm size on polar Keplerian orbits.
// All orbital elements are equal apart from the longitude of the ascending node.
TEST_CASE("single collision polar")
{
    const auto psize = 1.57e-8;
    const auto [x1, v1] = kep_to_cart<double>({1.05, .05, boost::math::constants::pi<double>() / 2, 0, 1.23, 0}, 1);
    const auto [x2, v2] = kep_to_cart<double>({1.05, .05, boost::math::constants::pi<double>() / 2, 0, 4.56, 0}, 1);

    sim s(std::vector<double>{x1[0], x2[0]}, std::vector<double>{x1[1], x2[1]}, std::vector<double>{x1[2], x2[2]},
          std::vector<double>{v1[0], v2[0]}, std::vector<double>{v1[1], v2[1]}, std::vector<double>{v1[2], v2[2]},
          std::vector<double>{psize, psize}, 0.23);

    while (s.step() != outcome::interrupt) {
    }

    const auto &i_info = std::get<0>(*s.get_interrupt_info());
    std::cout << fmt::format("Simulation end time: {}\n", s.get_time());
    std::cout << fmt::format("Colliding particles: {}\n", i_info);
    std::cout << fmt::format("x/y/z coordinates at collision time:\n{}\n{}\n{}\n", s.get_x(), s.get_y(), s.get_z());

    const auto dx = s.get_x()[0] - s.get_x()[1];
    const auto dy = s.get_y()[0] - s.get_y()[1];
    const auto dz = s.get_z()[0] - s.get_z()[1];

    REQUIRE(dx * dx + dy * dy + dz * dz - 4 * psize * psize < std::numeric_limits<double>::epsilon() * 10);
}

TEST_CASE("single collision propagate_until")
{
    using Catch::Detail::Approx;

    const auto psize = 1.57e-8;
    const auto [x1, v1] = kep_to_cart<double>({1.05, .05, boost::math::constants::pi<double>() / 2, 0, 1.23, 0}, 1);
    const auto [x2, v2] = kep_to_cart<double>({1.05, .05, boost::math::constants::pi<double>() / 2, 0, 4.56, 0}, 1);

    sim s(std::vector<double>{x1[0], x2[0]}, std::vector<double>{x1[1], x2[1]}, std::vector<double>{x1[2], x2[2]},
          std::vector<double>{v1[0], v2[0]}, std::vector<double>{v1[1], v2[1]}, std::vector<double>{v1[2], v2[2]},
          std::vector<double>{psize, psize}, 0.23);

    // Propagate until *after* the collision time.
    auto oc = s.propagate_until(10);

    REQUIRE(oc == outcome::interrupt);
    REQUIRE(s.get_time() < 9);

    const auto &i_info = std::get<0>(*s.get_interrupt_info());
    std::cout << fmt::format("Simulation end time: {}\n", s.get_time());
    std::cout << fmt::format("Colliding particles: {}\n", i_info);
    std::cout << fmt::format("x/y/z coordinates at collision time:\n{}\n{}\n{}\n", s.get_x(), s.get_y(), s.get_z());

    const auto dx = s.get_x()[0] - s.get_x()[1];
    const auto dy = s.get_y()[0] - s.get_y()[1];
    const auto dz = s.get_z()[0] - s.get_z()[1];

    REQUIRE(dx * dx + dy * dy + dz * dz - 4 * psize * psize < std::numeric_limits<double>::epsilon() * 10);

    // Propagate until *before* the collision time.
    s.set_new_state(std::vector<double>{x1[0], x2[0]}, std::vector<double>{x1[1], x2[1]},
                    std::vector<double>{x1[2], x2[2]}, std::vector<double>{v1[0], v2[0]},
                    std::vector<double>{v1[1], v2[1]}, std::vector<double>{v1[2], v2[2]},
                    std::vector<double>{psize, psize});
    s.set_time(0);

    oc = s.propagate_until(0.5);
    REQUIRE(oc == outcome::time_limit);
    REQUIRE(!s.get_interrupt_info());
    REQUIRE(s.get_time() == .5);

    // Propagate until shortly *before* the collision time.
    s.set_new_state(std::vector<double>{x1[0], x2[0]}, std::vector<double>{x1[1], x2[1]},
                    std::vector<double>{x1[2], x2[2]}, std::vector<double>{v1[0], v2[0]},
                    std::vector<double>{v1[1], v2[1]}, std::vector<double>{v1[2], v2[2]},
                    std::vector<double>{psize, psize});
    s.set_time(0);

    oc = s.propagate_until(1.57);
    REQUIRE(oc == outcome::time_limit);
    REQUIRE(s.get_time() == Approx(1.57).epsilon(0.).margin(1e-15));
    REQUIRE(!s.get_interrupt_info());
}

// Same as above, but equatorial orbits. This tests a configuration
// in which a dimension of the global bounding box (i.e., the z coordinate)
// is zero.
TEST_CASE("single collision equatorial")
{
    const auto psize = 1.57e-8;
    const auto [x1, v1] = kep_to_cart<double>({1.05, 0., 0., 0., 0., 0}, 1);
    auto x2 = x1;
    x2[0] = -x2[0];
    auto v2 = v1;

    sim s(std::vector<double>{x1[0], x2[0]}, std::vector<double>{x1[1], x2[1]}, std::vector<double>{x1[2], x2[2]},
          std::vector<double>{v1[0], v2[0]}, std::vector<double>{v1[1], v2[1]}, std::vector<double>{v1[2], v2[2]},
          std::vector<double>{psize, psize}, 0.23);

    while (s.step() != outcome::interrupt) {
    }

    const auto &i_info = std::get<0>(*s.get_interrupt_info());
    std::cout << fmt::format("Simulation end time: {}\n", s.get_time());
    std::cout << fmt::format("Colliding particles: {}\n", i_info);
    std::cout << fmt::format("x/y/z coordinates at collision time:\n{}\n{}\n{}\n", s.get_x(), s.get_y(), s.get_z());

    const auto dx = s.get_x()[0] - s.get_x()[1];
    const auto dy = s.get_y()[0] - s.get_y()[1];
    const auto dz = s.get_z()[0] - s.get_z()[1];

    REQUIRE(dx * dx + dy * dy + dz * dz - 4 * psize * psize < std::numeric_limits<double>::epsilon() * 10);
}

TEST_CASE("single collision equatorial propagate_until")
{
    const auto psize = 1.57e-8;
    const auto [x1, v1] = kep_to_cart<double>({1.05, 0., 0., 0., 0., 0}, 1);
    auto x2 = x1;
    x2[0] = -x2[0];
    auto v2 = v1;

    sim s(std::vector<double>{x1[0], x2[0]}, std::vector<double>{x1[1], x2[1]}, std::vector<double>{x1[2], x2[2]},
          std::vector<double>{v1[0], v2[0]}, std::vector<double>{v1[1], v2[1]}, std::vector<double>{v1[2], v2[2]},
          std::vector<double>{psize, psize}, 0.23);

    const auto oc = s.propagate_until(0.5);

    REQUIRE(oc == outcome::time_limit);
    REQUIRE(!s.get_interrupt_info());
    REQUIRE(s.get_time() == .5);
}
