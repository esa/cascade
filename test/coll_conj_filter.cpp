// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <array>
#include <initializer_list>
#include <limits>
#include <unordered_set>
#include <variant>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <cascade/sim.hpp>

#include "catch.hpp"
#include "keputils.hpp"

using namespace cascade;
using namespace cascade_test;

// Polar collision filtered out by minimum collisional radius and whitelist.
TEST_CASE("coll filtering")
{
    const auto psize = 1.57e-8;
    const auto [x1, v1] = kep_to_cart<double>({1., .000005, boost::math::constants::pi<double>() / 2, 0, 1.23, 0}, 1);
    const auto [x2, v2] = kep_to_cart<double>({1., .000005, boost::math::constants::pi<double>() / 2, 0, 4.56, 0}, 1);

    const auto [x3, v3] = kep_to_cart<double>({1.05, 0., 0., 0., 0., 0}, 1);
    auto x4 = x3;
    x4[0] = -x4[0];
    auto v4 = v3;

    for (auto n_par_ct : {1u, 3u}) {
        std::vector ic
            = {x1[0], x1[1], x1[2], v1[0], v1[1], v1[2], psize, x2[0], x2[1], x2[2], v2[0], v2[1], v2[2], psize};

        sim s(ic, 0.23, kw::min_coll_radius = psize * 2, kw::n_par_ct = n_par_ct);

        auto oc = s.propagate_until(10.);

        REQUIRE(oc == outcome::time_limit);

        // Make only 1 particle too small.
        s.set_time(0);
        s.set_new_state_pars(ic);
        s.get_state_data()[6] = psize * 3;

        oc = s.propagate_until(10.);

        REQUIRE(oc == outcome::collision);

        // Try whitelisting only 1.
        s.set_time(0);
        s.set_new_state_pars(ic);
        s.get_state_data()[6] = psize * 3;
        s.set_coll_whitelist({0});

        oc = s.propagate_until(10.);

        REQUIRE(oc == outcome::collision);

        // Add an equatorial collision.
        ic.insert(ic.end(),
                  {x3[0], x3[1], x3[2], v3[0], v3[1], v3[2], psize, x4[0], x4[1], x4[2], v4[0], v4[1], v4[2], psize});

        s.set_time(0);
        s.set_new_state_pars(ic);
        s.get_state_data()[20] = psize * 3;
        s.set_coll_whitelist({});

        oc = s.propagate_until(10.);

        REQUIRE(oc == outcome::collision);
        REQUIRE(std::get<0>(*s.get_interrupt_info()) == std::array<sim::size_type, 2>{2, 3});

        // Filter out the only collision via whitelist.
        s.set_time(0);
        s.set_new_state_pars(ic);
        s.get_state_data()[20] = psize * 3;
        s.set_coll_whitelist({0});

        oc = s.propagate_until(10.);

        REQUIRE(oc == outcome::time_limit);
    }
}

// Conjunction filtering via collisional threshold and whitelisting.
TEST_CASE("conj filtering")
{
    const auto psize = 1.57e-8;
    const auto [x1, v1] = kep_to_cart<double>({1., .000005, boost::math::constants::pi<double>() / 2, 0, 1.23, 0}, 1);
    const auto [x2, v2] = kep_to_cart<double>({1., .000005, boost::math::constants::pi<double>() / 2, 0, 4.56, 0}, 1);

    const auto [x3, v3] = kep_to_cart<double>({1.05, 0., 0., 0., 0., 0}, 1);
    auto x4 = x3;
    x4[0] = -x4[0];
    auto v4 = v3;

    for (auto n_par_ct : {1u, 3u}) {
        std::vector ic
            = {x1[0], x1[1], x1[2], v1[0], v1[1], v1[2], psize, x2[0], x2[1], x2[2], v2[0], v2[1], v2[2], psize};

        // Start of disabling both collisions (with +inf min_coll_radius) and conjunctions (off by default)
        sim s(ic, 0.23, kw::min_coll_radius = std::numeric_limits<double>::infinity(), kw::n_par_ct = n_par_ct);

        auto oc = s.propagate_until(10.);

        REQUIRE(oc == outcome::time_limit);
        REQUIRE(s.get_conjunctions().empty());

        // Turn on conjunction detection.
        s.set_time(0);
        s.set_new_state_pars(ic);
        s.set_conj_thresh(psize / 2);

        oc = s.propagate_until(10.);

        REQUIRE(oc == outcome::time_limit);
        REQUIRE(s.get_conjunctions().size() == 3u);

        // Try whitelisting only 1.
        s.set_time(0);
        s.set_new_state_pars(ic);
        s.set_conj_whitelist({0});
        s.reset_conjunctions();

        oc = s.propagate_until(10.);

        REQUIRE(oc == outcome::time_limit);
        REQUIRE(s.get_conjunctions().size() == 3u);

        // Add an equatorial conjunction.
        ic.insert(ic.end(),
                  {x3[0], x3[1], x3[2], v3[0], v3[1], v3[2], psize, x4[0], x4[1], x4[2], v4[0], v4[1], v4[2], psize});

        s.set_time(0);
        s.set_new_state_pars(ic);
        s.set_conj_whitelist({});
        s.reset_conjunctions();

        oc = s.propagate_until(10.);

        REQUIRE(oc == outcome::time_limit);
        REQUIRE(s.get_conjunctions().size() == 6u);

        // Disable one conjunction via whitelist.
        s.set_time(0);
        s.set_new_state_pars(ic);
        s.set_conj_whitelist({0});
        s.reset_conjunctions();

        oc = s.propagate_until(10.);

        REQUIRE(oc == outcome::time_limit);
        REQUIRE(s.get_conjunctions().size() == 3u);
        REQUIRE(std::all_of(s.get_conjunctions().begin(), s.get_conjunctions().end(),
                            [](const auto &c) { return c.i == 0u && c.j == 1u; }));

        // Same as above, other conjunction.
        s.set_time(0);
        s.set_new_state_pars(ic);
        s.set_conj_whitelist({2, 3});
        s.reset_conjunctions();

        oc = s.propagate_until(10.);

        REQUIRE(oc == outcome::time_limit);
        REQUIRE(s.get_conjunctions().size() == 3u);
        REQUIRE(std::all_of(s.get_conjunctions().begin(), s.get_conjunctions().end(),
                            [](const auto &c) { return c.i == 2u && c.j == 3u; }));
    }
}
