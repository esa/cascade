// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <initializer_list>
#include <variant>
#include <vector>

#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include <cascade/sim.hpp>

#include "catch.hpp"

using namespace cascade;

// This test checks that we deal correctly with different interrupt
// conditions arising in the same step.
TEST_CASE("interrupt order")
{
    using Catch::Detail::Approx;

    // Test 1: 2 particles colliding right before one exits the domain.
    sim s({0.99, 0, 0, 1, 0, 0, 1e-6, 1.01, 0, 0, -1, 0, 0, 1e-6, 4.9, 0, 0, 1, 0, 0, 1e-6}, .23, kw::d_radius = 5.,
          kw::c_radius = 0.75);

    auto sv = xt::adapt(s.get_state_data(), {3, 7});
    auto pos = xt::view(sv, xt::all(), xt::range(0, 3));

    auto oc = s.propagate_until(1000.);

    auto x0 = pos(0, 0);
    auto y0 = pos(0, 1);
    auto z0 = pos(0, 2);

    auto x1 = pos(1, 0);
    auto y1 = pos(1, 1);
    auto z1 = pos(1, 2);

    REQUIRE(oc == outcome::collision);
    REQUIRE(std::get<0>(*s.get_interrupt_info()) == std::array<sim::size_type, 2>{0, 1});
    REQUIRE((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) + (z0 - z1) * (z0 - z1)
            == Approx(4e-12).epsilon(0.).margin(1e-14));

    // The opposite, particle exiting before collision.
    sv = xt::xarray<double>{{0.99, 0, 0, 1, 0, 0, 1e-6}, {1.01, 0, 0, -1, 0, 0, 1e-6}, {4.9, 0, 0, 1000, 0, 0, 1e-6}};

    s.set_time(0.);

    oc = s.propagate_until(1000.);

    auto x2 = pos(2, 0);
    auto y2 = pos(2, 1);
    auto z2 = pos(2, 2);

    REQUIRE(oc == outcome::exit);
    REQUIRE(std::get<1>(*s.get_interrupt_info()) == 2u);
    REQUIRE(x2 * x2 + y2 * y2 + z2 * z2 == Approx(25.).epsilon(0.).margin(1e-14));

    // Particles colliding before reentry.
    sv = xt::xarray<double>{{0.99, 0, 0, 1, 0, 0, 1e-6}, {1.01, 0, 0, -1, 0, 0, 1e-6}, {0.76, 0, 0, 0, 0, 0, 1e-6}};
    s.set_time(0.);

    oc = s.propagate_until(1000.);

    x0 = pos(0, 0);
    y0 = pos(0, 1);
    z0 = pos(0, 2);

    x1 = pos(1, 0);
    y1 = pos(1, 1);
    z1 = pos(1, 2);

    REQUIRE(oc == outcome::collision);
    REQUIRE(std::get<0>(*s.get_interrupt_info()) == std::array<sim::size_type, 2>{0, 1});
    REQUIRE((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) + (z0 - z1) * (z0 - z1)
            == Approx(4e-12).epsilon(0.).margin(1e-14));

    // Particle reentry before collision.
    sv = xt::xarray<double>{{0.99, 0, 0, 1, 0, 0, 1e-6}, {1.01, 0, 0, -1, 0, 0, 1e-6}, {0.76, 0, 0, -1, 0, 0, 1e-6}};
    s.set_time(0.);

    oc = s.propagate_until(1000.);

    x2 = pos(2, 0);
    y2 = pos(2, 1);
    z2 = pos(2, 2);

    REQUIRE(oc == outcome::reentry);
    REQUIRE(std::get<1>(*s.get_interrupt_info()) == 2u);
    REQUIRE(x2 * x2 + y2 * y2 + z2 * z2 == Approx(0.75 * 0.75).epsilon(0.).margin(1e-14));

    // Reentry happening before domain exit, no collision.
    sv = xt::xarray<double>{{0.76, 0, 0, -1, 0, 0, 1e-6}, {1., 0, 0, 0, 1, 0, 1e-6}, {2, 0, 0, 1, 0, 0, 1e-6}};
    s.set_time(0.);

    oc = s.propagate_until(1000., 10.);

    x0 = pos(0, 0);
    y0 = pos(0, 1);
    z0 = pos(0, 2);

    REQUIRE(oc == outcome::reentry);
    REQUIRE(std::get<1>(*s.get_interrupt_info()) == 0u);
    REQUIRE(x0 * x0 + y0 * y0 + z0 * z0 == Approx(0.75 * 0.75).epsilon(0.).margin(1e-14));

    // The opposite.
    sv = xt::xarray<double>{{0.76, 0, 0, 0, 0, 0, 1e-6}, {1., 0, 0, 0, 1, 0, 1e-6}, {4.9, 0, 0, 1, 0, 0, 1e-6}};
    s.set_time(0.);

    oc = s.propagate_until(1000., 10.);

    x2 = pos(2, 0);
    y2 = pos(2, 1);
    z2 = pos(2, 2);

    REQUIRE(oc == outcome::exit);
    REQUIRE(std::get<1>(*s.get_interrupt_info()) == 2u);
    REQUIRE(x2 * x2 + y2 * y2 + z2 * z2 == Approx(25.).epsilon(0.).margin(1e-14));
}
