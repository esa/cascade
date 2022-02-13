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

#include <cascade/sim.hpp>

#include "catch.hpp"

using namespace cascade;

// This test checks that we deal correctly with different interrupt
// conditions arising in the same step.
TEST_CASE("interrupt order")
{
    using Catch::Detail::Approx;

    using vec_t = std::vector<double>;

    // Test 1: 2 particles colliding right before one exits the domain.
    sim s(vec_t{0.99, 1.01, 4.9}, vec_t(3u, 0.), vec_t(3u, 0.), vec_t{1., -1., 1.}, vec_t(3u, 0.), vec_t(3u, 0.),
          vec_t(3u, 1e-6), 0.23, kw::d_radius = 5., kw::c_radius = 0.75);

    auto oc = s.propagate_until(1000.);

    auto x0 = s.get_x()[0];
    auto y0 = s.get_y()[0];
    auto z0 = s.get_z()[0];

    auto x1 = s.get_x()[1];
    auto y1 = s.get_y()[1];
    auto z1 = s.get_z()[1];

    REQUIRE(oc == outcome::collision);
    REQUIRE(std::get<0>(*s.get_interrupt_info()) == std::array<sim::size_type, 2>{0, 1});
    REQUIRE((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) + (z0 - z1) * (z0 - z1)
            == Approx(4e-12).epsilon(0.).margin(1e-14));

    // The opposite, particle exiting before collision.
    s.set_new_state(vec_t{0.99, 1.01, 4.9}, vec_t(3u, 0.), vec_t(3u, 0.), vec_t{1., -1., 1000.}, vec_t(3u, 0.),
                    vec_t(3u, 0.), vec_t(3u, 1e-6));
    s.set_time(0.);

    oc = s.propagate_until(1000.);

    auto x2 = s.get_x()[2];
    auto y2 = s.get_y()[2];
    auto z2 = s.get_z()[2];

    REQUIRE(oc == outcome::exit);
    REQUIRE(std::get<1>(*s.get_interrupt_info()) == 2u);
    REQUIRE(x2 * x2 + y2 * y2 + z2 * z2 == Approx(25.).epsilon(0.).margin(1e-14));

    // Particles colliding before reentry.
    s.set_new_state(vec_t{0.99, 1.01, 0.76}, vec_t(3u, 0.), vec_t(3u, 0.), vec_t{1., -1., 0.}, vec_t(3u, 0.),
                    vec_t(3u, 0.), vec_t(3u, 1e-6));
    s.set_time(0.);

    oc = s.propagate_until(1000.);

    x0 = s.get_x()[0];
    y0 = s.get_y()[0];
    z0 = s.get_z()[0];

    x1 = s.get_x()[1];
    y1 = s.get_y()[1];
    z1 = s.get_z()[1];

    REQUIRE(oc == outcome::collision);
    REQUIRE(std::get<0>(*s.get_interrupt_info()) == std::array<sim::size_type, 2>{0, 1});
    REQUIRE((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1) + (z0 - z1) * (z0 - z1)
            == Approx(4e-12).epsilon(0.).margin(1e-14));

    // Particle reentry before collision.
    s.set_new_state(vec_t{0.99, 1.01, .76}, vec_t(3u, 0.), vec_t(3u, 0.), vec_t{1., -1., -1.}, vec_t(3u, 0.),
                    vec_t(3u, 0.), vec_t(3u, 1e-6));
    s.set_time(0.);

    oc = s.propagate_until(1000.);

    x2 = s.get_x()[2];
    y2 = s.get_y()[2];
    z2 = s.get_z()[2];

    REQUIRE(oc == outcome::reentry);
    REQUIRE(std::get<1>(*s.get_interrupt_info()) == 2u);
    REQUIRE(x2 * x2 + y2 * y2 + z2 * z2 == Approx(0.75 * 0.75).epsilon(0.).margin(1e-14));

    // Reentry happening before domain exit, no collision.
    s.set_new_state(vec_t{0.76, 1., 2.}, vec_t(3u, 0.), vec_t(3u, 0.), vec_t{-1., 0., 1.}, vec_t{0., 1., 0.},
                    vec_t(3u, 0.), vec_t(3u, 1e-6));
    s.set_time(0.);

    oc = s.propagate_until(1000., 10.);

    x0 = s.get_x()[0];
    y0 = s.get_y()[0];
    z0 = s.get_z()[0];

    REQUIRE(oc == outcome::reentry);
    REQUIRE(std::get<1>(*s.get_interrupt_info()) == 0u);
    REQUIRE(x0 * x0 + y0 * y0 + z0 * z0 == Approx(0.75 * 0.75).epsilon(0.).margin(1e-14));

    // The opposite.
    s.set_new_state(vec_t{0.76, 1., 4.9}, vec_t(3u, 0.), vec_t(3u, 0.), vec_t{0., 0., 1.}, vec_t{0., 1., 0.},
                    vec_t(3u, 0.), vec_t(3u, 1e-6));
    s.set_time(0.);

    oc = s.propagate_until(1000., 10.);

    x2 = s.get_x()[2];
    y2 = s.get_y()[2];
    z2 = s.get_z()[2];

    REQUIRE(oc == outcome::exit);
    REQUIRE(std::get<1>(*s.get_interrupt_info()) == 2u);
    REQUIRE(x2 * x2 + y2 * y2 + z2 * z2 == Approx(25.).epsilon(0.).margin(1e-14));
}
