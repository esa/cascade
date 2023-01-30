// Copyright 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <variant>
#include <vector>

#include <boost/algorithm/string/predicate.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include <cascade/logging.hpp>
#include <cascade/sim.hpp>

#include "catch.hpp"

using namespace cascade;

TEST_CASE("reentry sphere")
{
    using Catch::Detail::Approx;

    sim s({1.5, 0, 0, 0, 0.01, 0., 0}, 0.23, kw::c_radius = 1.);

    auto sv = xt::adapt(s.get_state_data(), {1, 7});
    auto pos = xt::view(sv, xt::all(), xt::range(0, 3));

    auto oc = s.propagate_until(1000.);

    REQUIRE(oc == outcome::reentry);

    auto x = pos(0, 0);
    auto y = pos(0, 1);
    auto z = pos(0, 2);

    REQUIRE(std::sqrt(x * x + y * y + z * z) == Approx(1.).epsilon(0.).margin(1e-14));

    sv = xt::xarray<double>{{0., 1.5, 0, 0, 0., 0., 0.}};
    s.set_time(0.);

    oc = s.propagate_until(1000);

    REQUIRE(oc == outcome::reentry);

    x = pos(0, 0);
    y = pos(0, 1);
    z = pos(0, 2);

    REQUIRE(std::sqrt(x * x + y * y + z * z) == Approx(1.).epsilon(0.).margin(1e-14));

    // Test possible re-trigger of the same event if we do not resolve it.
    try {
        s.step();
    } catch (const std::invalid_argument &ia) {
        REQUIRE(boost::algorithm::contains(
            ia.what(), "The recomputed number of chunks after the triggering of a stopping terminal"));
    }

    // Ensure correctness also when using the batch integrator.
    s.set_new_state_pars({1.1, 0., 0., 0., .953, 0., 0., 1.5, 0., 0., 0., .01,  0., 0., 1.1, 0., 0., 0., .953, 0., 0.,
                          1.1, 0., 0., 0., .953, 0., 0., 1.1, 0., 0., 0., .953, 0., 0., 1.1, 0., 0., 0., .953, 0., 0.});
    s.set_time(0.);

    oc = s.propagate_until(1000);

    REQUIRE(oc == outcome::reentry);
    REQUIRE(std::get<1>(*s.get_interrupt_info()) == 1u);

    auto sv2 = xt::adapt(s.get_state_data(), {6, 7});
    auto pos2 = xt::view(sv2, xt::all(), xt::range(0, 3));

    x = pos2(1, 0);
    y = pos2(1, 1);
    z = pos2(1, 2);

    REQUIRE(std::sqrt(x * x + y * y + z * z) == Approx(1.).epsilon(0.).margin(1e-14));

    // Test possible re-trigger of the same event if we do not resolve it.
    try {
        s.step();
    } catch (const std::invalid_argument &ia) {
        REQUIRE(boost::algorithm::contains(
            ia.what(), "The recomputed number of chunks after the triggering of a stopping terminal"));
    }
}

TEST_CASE("reentry ellipsoid")
{
    using Catch::Detail::Approx;

    sim s({1.5, 0, 0, 0, 0., 0., 0}, 0.23, kw::c_radius = std::vector<double>{1., 1.1, 1.2});

    auto sv = xt::adapt(s.get_state_data(), {1, 7});
    auto pos = xt::view(sv, xt::all(), xt::range(0, 3));

    auto oc = s.propagate_until(1000.);

    REQUIRE(oc == outcome::reentry);

    auto x = pos(0, 0);
    auto y = pos(0, 1);
    auto z = pos(0, 2);

    REQUIRE(std::sqrt(x * x + y * y + z * z) == Approx(1.).epsilon(0.).margin(1e-14));

    sv = xt::xarray<double>{{0., 1.5, 0, 0, 0., 0., 0}};

    s.set_time(0.);

    oc = s.propagate_until(1000);

    REQUIRE(oc == outcome::reentry);

    x = pos(0, 0);
    y = pos(0, 1);
    z = pos(0, 2);

    REQUIRE(std::sqrt(x * x + y * y + z * z) == Approx(1.1).epsilon(0.).margin(1e-14));

    sv = xt::xarray<double>{{0., 0, 1.5, 0, 0., 0., 0}};

    s.set_time(0.);

    oc = s.propagate_until(1000);

    REQUIRE(oc == outcome::reentry);

    x = pos(0, 0);
    y = pos(0, 1);
    z = pos(0, 2);

    REQUIRE(std::sqrt(x * x + y * y + z * z) == Approx(1.2).epsilon(0.).margin(1e-14));

    try {
        s.step();
    } catch (const std::invalid_argument &ia) {
        REQUIRE(boost::algorithm::contains(
            ia.what(), "The recomputed number of chunks after the triggering of a stopping terminal"));
    }

    // Ensure correctness also when using the batch integrator.
    s.set_new_state_pars({1.1, 0., 0., 0., .953, 0., 0., 1.5, 0., 0., 0., .0,   0., 0., 1.1, 0., 0., 0., .953, 0., 0.,
                          1.1, 0., 0., 0., .953, 0., 0., 1.1, 0., 0., 0., .953, 0., 0., 1.1, 0., 0., 0., .953, 0., 0.});
    s.set_time(0.);

    oc = s.propagate_until(1000);

    REQUIRE(oc == outcome::reentry);
    REQUIRE(std::get<1>(*s.get_interrupt_info()) == 1u);

    auto sv2 = xt::adapt(s.get_state_data(), {6, 7});
    auto pos2 = xt::view(sv2, xt::all(), xt::range(0, 3));

    x = pos2(1, 0);
    y = pos2(1, 1);
    z = pos2(1, 2);

    REQUIRE(std::sqrt(x * x + y * y + z * z) == Approx(1.).epsilon(0.).margin(1e-14));

    try {
        s.step();
    } catch (const std::invalid_argument &ia) {
        REQUIRE(boost::algorithm::contains(
            ia.what(), "The recomputed number of chunks after the triggering of a stopping terminal"));
    }

    sv2 = xt::xarray<double>{{0., 0., 2.1, .69, 0., 0., 0.}, {0., 0., 1.5, .0, 0., 0., 0.},
                             {0., 0., 2.1, .69, 0., 0., 0.}, {0., 0., 2.1, .69, 0., 0., 0.},
                             {0., 0., 2.1, .69, 0., 0., 0.}, {0., 0., 2.1, .69, 0., 0., 0.}};

    s.set_time(0.);

    oc = s.propagate_until(1000);

    REQUIRE(oc == outcome::reentry);
    REQUIRE(std::get<1>(*s.get_interrupt_info()) == 1u);

    x = pos2(1, 0);
    y = pos2(1, 1);
    z = pos2(1, 2);

    REQUIRE(std::sqrt(x * x + y * y + z * z) == Approx(1.2).epsilon(0.).margin(1e-14));

    sv2 = xt::xarray<double>{{0., 2.1, 0., .69, 0., 0., 0.}, {0., 2.1, 0., .69, 0., 0., 0.},
                             {0., 2.1, 0., .69, 0., 0., 0.}, {0., 1.5, 0., .0, 0., 0., 0.},
                             {0., 2.1, 0., .69, 0., 0., 0.}, {0., 2.1, 0., .69, 0., 0., 0.}};

    s.set_time(0.);

    oc = s.propagate_until(1000);

    REQUIRE(oc == outcome::reentry);
    REQUIRE(std::get<1>(*s.get_interrupt_info()) == 3u);

    x = pos2(3, 0);
    y = pos2(3, 1);
    z = pos2(3, 2);

    REQUIRE(std::sqrt(x * x + y * y + z * z) == Approx(1.1).epsilon(0.).margin(1e-14));
}
