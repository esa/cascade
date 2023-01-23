// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

#include <heyoka/expression.hpp>

#include <cascade/sim.hpp>

#include "catch.hpp"

using namespace cascade;

TEST_CASE("basic")
{
    using Catch::Matchers::Message;

    using namespace heyoka::literals;

    // Default construction.
    {
        sim s;

        REQUIRE(s.get_state().empty());
        REQUIRE(s.get_pars().empty());
        REQUIRE(s.get_nparts() == 0u);
        REQUIRE(s.get_time() == 0);
        REQUIRE(s.get_ct() == 1);
        REQUIRE(s.get_tol() == std::numeric_limits<double>::epsilon());
        REQUIRE(!s.get_high_accuracy());
        REQUIRE(s.get_npars() == 0u);
        REQUIRE(std::get<0>(s.get_c_radius()) == 0.);
        REQUIRE(s.get_d_radius() == 0.);
    }

    // Construction with non-default parameters.
    {
        auto dyn = dynamics::kepler();
        dyn[0].second += heyoka::par[1];

        sim s({1., .001, .001, .001, 1., .001, .001}, .5, kw::dyn = dyn, kw::pars = {.002, .001},
              kw::c_radius = {.1, .2, .3}, kw::d_radius = 100., kw::tol = 1e-12, kw::high_accuracy = true);

        REQUIRE(!s.get_state().empty());
        REQUIRE(s.get_state() == std::vector{1., .001, .001, .001, 1., .001, .001});
        REQUIRE(!s.get_pars().empty());
        REQUIRE(s.get_pars() == std::vector{.002, .001});
        REQUIRE(s.get_nparts() == 1u);
        REQUIRE(s.get_time() == 0);
        REQUIRE(s.get_ct() == .5);
        REQUIRE(s.get_tol() == 1e-12);
        REQUIRE(s.get_high_accuracy());
        REQUIRE(s.get_npars() == 2u);
        REQUIRE(std::get<1>(s.get_c_radius()) == std::vector{.1, .2, .3});
        REQUIRE(s.get_d_radius() == 100.);
        REQUIRE(s.get_time() == 0.);

        // Take a single step.
        s.step();

        // Make a copy.
        auto s2 = s;

        REQUIRE(s2.get_state() == s.get_state());
        REQUIRE(s2.get_pars() == std::vector{.002, .001});
        REQUIRE(s2.get_nparts() == 1u);
        REQUIRE(s2.get_time() == s.get_time());
        REQUIRE(s2.get_ct() == .5);
        REQUIRE(s2.get_tol() == 1e-12);
        REQUIRE(s2.get_high_accuracy());
        REQUIRE(s2.get_npars() == 2u);
        REQUIRE(std::get<1>(s2.get_c_radius()) == std::vector{.1, .2, .3});
        REQUIRE(s2.get_d_radius() == 100.);

        // Test also with move.
        auto s3 = std::move(s2);

        REQUIRE(s3.get_state() == s.get_state());
        REQUIRE(s3.get_pars() == std::vector{.002, .001});
        REQUIRE(s3.get_nparts() == 1u);
        REQUIRE(s3.get_time() == s.get_time());
        REQUIRE(s3.get_ct() == .5);
        REQUIRE(s3.get_tol() == 1e-12);
        REQUIRE(s3.get_high_accuracy());
        REQUIRE(s3.get_npars() == 2u);
        REQUIRE(std::get<1>(s3.get_c_radius()) == std::vector{.1, .2, .3});
        REQUIRE(s3.get_d_radius() == 100.);

        // Revive s2 via assignment.
        s2 = std::move(s3);

        REQUIRE(s2.get_state() == s.get_state());
        REQUIRE(s2.get_pars() == std::vector{.002, .001});
        REQUIRE(s2.get_nparts() == 1u);
        REQUIRE(s2.get_time() == s.get_time());
        REQUIRE(s2.get_ct() == .5);
        REQUIRE(s2.get_tol() == 1e-12);
        REQUIRE(s2.get_high_accuracy());
        REQUIRE(s2.get_npars() == 2u);
        REQUIRE(std::get<1>(s2.get_c_radius()) == std::vector{.1, .2, .3});
        REQUIRE(s2.get_d_radius() == 100.);

        // Revive s3 via assignment.
        s3 = s2;

        REQUIRE(s3.get_state() == s.get_state());
        REQUIRE(s3.get_pars() == std::vector{.002, .001});
        REQUIRE(s3.get_nparts() == 1u);
        REQUIRE(s3.get_time() == s.get_time());
        REQUIRE(s3.get_ct() == .5);
        REQUIRE(s3.get_tol() == 1e-12);
        REQUIRE(s3.get_high_accuracy());
        REQUIRE(s3.get_npars() == 2u);
        REQUIRE(std::get<1>(s3.get_c_radius()) == std::vector{.1, .2, .3});
        REQUIRE(s3.get_d_radius() == 100.);
    }

    // Error modes.
    REQUIRE_THROWS_MATCHES(sim({1., .001, .001, .001, 1., .001}, .5), std::invalid_argument,
                           Message("The size of the state vector is 6, which is not a multiple of 7"));

    REQUIRE_THROWS_MATCHES(sim({}, 0), std::invalid_argument,
                           Message("The collisional timestep must be finite and positive, but it is 0 instead"));
    REQUIRE_THROWS_AS(sim({}, std::numeric_limits<double>::infinity()), std::invalid_argument);

    REQUIRE_THROWS_MATCHES(sim({1., .001, .001, .001, 1., .001, .001}, .5,
                               kw::dyn = std::vector<std::pair<heyoka::expression, heyoka::expression>>{{}}),
                           std::invalid_argument,
                           Message("6 dynamical equations are expected, but 1 were provided instead"));
    auto dyn = dynamics::kepler();
    dyn[0].first = "foo"_var;
    REQUIRE_THROWS_MATCHES(sim({1., .001, .001, .001, 1., .001, .001}, .5, kw::dyn = dyn), std::invalid_argument,
                           Message("The LHS of the dynamics at index 0 must be a variable named \"x\", but instead it "
                                   "is the expression \"foo\""));
    dyn[0].first = "x"_var;
    dyn[0].second += "a"_var;
    REQUIRE_THROWS_MATCHES(
        sim({1., .001, .001, .001, 1., .001, .001}, .5, kw::dyn = dyn), std::invalid_argument,
        Message("The RHS of the differential equation for the variable \"x\" contains the invalid variables [\"a\"] "
                "(the allowed variables are [\"x\", \"y\", \"z\", \"vx\", \"vy\", \"vz\"])"));

    REQUIRE_THROWS_MATCHES(
        sim({}, .5, kw::c_radius = std::vector{.1}), std::invalid_argument,
        Message("The c_radius argument must be either a scalar (for a spherical central body) or a vector of 3 "
                "elements (for a triaxial ellipsoid), but instead it is a vector of 1 element(s)"));
    REQUIRE_THROWS_MATCHES(
        sim({}, .5, kw::c_radius = std::vector{1., 2., 0.}), std::invalid_argument,
        Message("A non-finite or non-positive value was detected among the 3 semiaxes of the central body: [1, 2, 0]"));
    REQUIRE_THROWS_MATCHES(
        sim({}, .5, kw::c_radius = -1), std::invalid_argument,
        Message("The radius of the central body must be finite and non-negative, but it is -1 instead"));

    REQUIRE_THROWS_MATCHES(sim({}, .5, kw::d_radius = -1), std::invalid_argument,
                           Message("The domain radius must be finite and non-negative, but it is -1 instead"));
}

TEST_CASE("remove particles")
{
    using Catch::Matchers::Message;

    // Empty sim first.
    {
        sim s;
        s.remove_particles({});

        REQUIRE(s.get_state().empty());
        REQUIRE(s.get_pars().empty());

        REQUIRE_THROWS_MATCHES(
            s.remove_particles({3, 1, 2}), std::invalid_argument,
            Message("An invalid vector of indices was passed to the function for particle removal: [1, 2, 3]"));
    }

    // Sim with Keplerian dynamics (i.e., no pars)
    // and a few particles.
    {
        std::vector st = {.1, .1, .1, .1, .1, .1, .1, .2, .2, .2, .2, .2, .2, .2};

        sim s(st, .5);
        s.remove_particles({});

        REQUIRE(s.get_state() == st);
        REQUIRE(s.get_pars().empty());

        // NOTE: check repeated indices.
        s.remove_particles({1, 1});

        REQUIRE(s.get_state() == std::vector{.1, .1, .1, .1, .1, .1, .1});
        REQUIRE(s.get_pars().empty());

        s.remove_particles({0, 0});

        REQUIRE(s.get_state().empty());
        REQUIRE(s.get_pars().empty());
    }

    // Sim with a couple of pars in the dynamics.
    {
        std::vector st = {.1, .1, .1, .1, .1, .1, .1, .2, .2, .2, .2, .2, .2, .2};
        std::vector pars = {.3, .3, .4, .4};

        auto dyn = dynamics::kepler();
        dyn[0].second += heyoka::par[1];

        sim s(st, .5, kw::dyn = dyn, kw::pars = pars);
        s.remove_particles({});

        REQUIRE(s.get_state() == st);
        REQUIRE(s.get_pars() == pars);

        // NOTE: check repeated indices.
        s.remove_particles({1, 1});

        REQUIRE(s.get_state() == std::vector{.1, .1, .1, .1, .1, .1, .1});
        REQUIRE(s.get_pars() == std::vector{.3, .3});

        s.remove_particles({0, 0});

        REQUIRE(s.get_state().empty());
        REQUIRE(s.get_pars().empty());
    }
}

TEST_CASE("set new state pars")
{
    using Catch::Matchers::Message;

    // Empty sim first.
    {
        sim s;

        std::vector st = {.1, .1, .1, .1, .1, .1, .1, .2, .2, .2, .2, .2, .2, .2};

        s.set_new_state_pars(st);

        REQUIRE(s.get_state() == st);
        REQUIRE(s.get_pars().empty());
        REQUIRE(s.get_nparts() == 2u);

        REQUIRE_THROWS_MATCHES(s.set_new_state_pars({.1}), std::invalid_argument,
                               Message("The size of the state vector is 1, which is not a multiple of 7"));

        REQUIRE(s.get_state() == st);
        REQUIRE(s.get_pars().empty());
        REQUIRE(s.get_nparts() == 2u);

        REQUIRE_THROWS_MATCHES(
            s.set_new_state_pars(st, {.1}), std::invalid_argument,
            Message("The input array of parameter values must be empty when the number of parameters "
                    "in the dynamics is zero"));
    }

    // Sim with a couple of pars in the dynamics.
    {
        std::vector st = {.1, .1, .1, .1, .1, .1, .1, .2, .2, .2, .2, .2, .2, .2};
        std::vector pars = {.3, .3, .4, .4};

        auto dyn = dynamics::kepler();
        dyn[0].second += heyoka::par[1];

        sim s(st, .5, kw::dyn = dyn, kw::pars = pars);

        s.set_new_state_pars(
            {
                .1,
                .1,
                .1,
                .1,
                .1,
                .1,
                .1,
            },
            {.3, .3});

        REQUIRE(s.get_state()
                == std::vector{
                    .1,
                    .1,
                    .1,
                    .1,
                    .1,
                    .1,
                    .1,
                });
        REQUIRE(s.get_pars() == std::vector{.3, .3});
        REQUIRE(s.get_nparts() == 1u);

        // Verify that leaving the pars vector empty sets
        // all pars to zero.
        s.set_new_state_pars({
            .2,
            .2,
            .2,
            .2,
            .2,
            .2,
            .2,
        });

        REQUIRE(s.get_state()
                == std::vector{
                    .2,
                    .2,
                    .2,
                    .2,
                    .2,
                    .2,
                    .2,
                });

        REQUIRE(std::all_of(s.get_pars().cbegin(), s.get_pars().cend(), [](auto val) { return val == 0; }));

        // Incorrect pars vector.
        REQUIRE_THROWS_MATCHES(s.set_new_state_pars(
                                   {
                                       .2,
                                       .2,
                                       .2,
                                       .2,
                                       .2,
                                       .2,
                                       .2,
                                   },
                                   {.1}),
                               std::invalid_argument,
                               Message("The input array of parameter values must have shape (1, 2), "
                                       "but instead its flattened size is 1"));
    }
}
