// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <sstream>
#include <unordered_map>

#include <boost/math/constants/constants.hpp>

#include <fmt/core.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include <heyoka/taylor.hpp>

#include <cascade/sim.hpp>

#include "catch.hpp"
#include "keputils.hpp"

using namespace cascade;
using namespace cascade_test;
namespace hy = heyoka;

// Conjunction (without collision) of two particles of size 10cm, one on a polar orbit,
// the other on an almost polar orbit. The conjunction is approximately on the north pole,
// approximately at 1/4 of the orbital period.
TEST_CASE("polar conj")
{
    using namespace hy::literals;

    const auto psize = 1.57e-8;
    const auto [x1, v1]
        = kep_to_cart<double>({1., .000005, boost::math::constants::pi<double>() / 2 + 1e-6, 0, 1.23, 0}, 1);
    const auto [x2, v2] = kep_to_cart<double>({1., .000005, boost::math::constants::pi<double>() / 2, 0, 4.56, 0}, 1);

    for (auto n_par_ct : {1u, 3u}) {
        sim s({x1[0], x1[1], x1[2], v1[0], v1[1], v1[2], psize, x2[0], x2[1], x2[2], v2[0], v2[1], v2[2], psize}, 0.23,
              kw::conj_thresh = psize * 100, kw::n_par_ct = n_par_ct);

        auto sv = xt::adapt(s.get_state().data(), {2, 7});
        auto pos = xt::view(sv, xt::all(), xt::range(0, 3));

        while (s.get_conjunctions().empty()) {
            REQUIRE(s.step() == outcome::success);
        }

        REQUIRE(s.get_conjunctions().size() == 1u);
        REQUIRE(s.get_conjunctions()[0].i == 0u);
        REQUIRE(s.get_conjunctions()[0].j == 1u);
        REQUIRE(std::abs(s.get_conjunctions()[0].time - boost::math::constants::pi<double>() / 2) < 1e-4);
        REQUIRE(s.get_conjunctions()[0].dist < psize * 100);

        // Run a corresponding heyoka integration with event detection
        // and compare.
        auto dyn = dynamics::kepler();
        auto subs_dict = std::unordered_map<std::string, hy::expression>{
            {"x", "x0"_var}, {"y", "y0"_var}, {"z", "z0"_var}, {"vx", "vx0"_var}, {"vy", "vy0"_var}, {"vz", "vz0"_var}};
        for (auto &[lhs, rhs] : dyn) {
            lhs = hy::subs(lhs, subs_dict);
            rhs = hy::subs(rhs, subs_dict);
        }

        subs_dict = std::unordered_map<std::string, hy::expression>{
            {"x", "x1"_var}, {"y", "y1"_var}, {"z", "z1"_var}, {"vx", "vx1"_var}, {"vy", "vy1"_var}, {"vz", "vz1"_var}};
        for (const auto &[lhs, rhs] : dynamics::kepler()) {
            dyn.emplace_back(hy::subs(lhs, subs_dict), hy::subs(rhs, subs_dict));
        }

        auto ev_eq = ("x1"_var - "x0"_var) * ("vx1"_var - "vx0"_var) + ("y1"_var - "y0"_var) * ("vy1"_var - "vy0"_var)
                     + ("z1"_var - "z0"_var) * ("vz1"_var - "vz0"_var);

        auto ta = hy::taylor_adaptive<double>{
            dyn,
            {x1[0], x1[1], x1[2], v1[0], v1[1], v1[2], x2[0], x2[1], x2[2], v2[0], v2[1], v2[2]},
            hy::kw::t_events = {hy::t_event<double>{ev_eq}}};

        const auto oc = std::get<0>(ta.propagate_until(100));

        REQUIRE(oc == hy::taylor_outcome{-1});
        REQUIRE(std::abs(ta.get_time() - s.get_conjunctions()[0].time) < 1e-15);

        const auto &st = ta.get_state();

        REQUIRE(std::abs(std::sqrt((st[0] - st[6]) * (st[0] - st[6]) + (st[1] - st[7]) * (st[1] - st[7])
                                   + (st[2] - st[8]) * (st[2] - st[8]))
                         - s.get_conjunctions()[0].dist)
                < 1e-9);

        // Compare the states.
        REQUIRE(std::abs(st[0] - s.get_conjunctions()[0].state_i[0]) < 1e-15);
        REQUIRE(std::abs(st[1] - s.get_conjunctions()[0].state_i[1]) < 1e-15);
        REQUIRE(std::abs(st[2] - s.get_conjunctions()[0].state_i[2]) < 1e-15);
        REQUIRE(std::abs(st[3] - s.get_conjunctions()[0].state_i[3]) < 1e-15);
        REQUIRE(std::abs(st[4] - s.get_conjunctions()[0].state_i[4]) < 1e-15);
        REQUIRE(std::abs(st[5] - s.get_conjunctions()[0].state_i[5]) < 1e-15);

        REQUIRE(std::abs(st[6] - s.get_conjunctions()[0].state_j[0]) < 1e-15);
        REQUIRE(std::abs(st[7] - s.get_conjunctions()[0].state_j[1]) < 1e-15);
        REQUIRE(std::abs(st[8] - s.get_conjunctions()[0].state_j[2]) < 1e-15);
        REQUIRE(std::abs(st[9] - s.get_conjunctions()[0].state_j[3]) < 1e-15);
        REQUIRE(std::abs(st[10] - s.get_conjunctions()[0].state_j[4]) < 1e-15);
        REQUIRE(std::abs(st[11] - s.get_conjunctions()[0].state_j[5]) < 1e-15);

        // Test also reset_conjunctions().
        s.reset_conjunctions();
        REQUIRE(s.get_conjunctions().empty());
    }
}

// Like above, but the IC are set such that the conjunction happens barely outside the limit
// and thus it is not recorded.
TEST_CASE("polar conj near miss")
{
    const auto psize = 1.57e-8;
    const auto [x1, v1]
        = kep_to_cart<double>({1., .000005, boost::math::constants::pi<double>() / 2 + 1.5775e-6, 0, 1.23, 0}, 1);
    const auto [x2, v2] = kep_to_cart<double>({1., .000005, boost::math::constants::pi<double>() / 2, 0, 4.56, 0}, 1);

    for (auto n_par_ct : {1u, 3u}) {
        sim s({x1[0], x1[1], x1[2], v1[0], v1[1], v1[2], psize, x2[0], x2[1], x2[2], v2[0], v2[1], v2[2], psize}, 0.23,
              kw::conj_thresh = psize * 100, kw::n_par_ct = n_par_ct);

        auto sv = xt::adapt(s.get_state().data(), {2, 7});
        auto pos = xt::view(sv, xt::all(), xt::range(0, 3));

        while (true) {
            REQUIRE(s.step() == outcome::success);
            REQUIRE(s.get_conjunctions().empty());
            if (s.get_time() > 3.14) {
                break;
            }
        }
    }
}

// Similar to above, but the conjunction is just barely happening (instead of just
// barely missing).
TEST_CASE("polar conj barely")
{
    using namespace hy::literals;

    const auto psize = 1.57e-8;
    const auto [x1, v1]
        = kep_to_cart<double>({1., .000005, boost::math::constants::pi<double>() / 2 + 1.5765e-6, 0, 1.23, 0}, 1);
    const auto [x2, v2] = kep_to_cart<double>({1., .000005, boost::math::constants::pi<double>() / 2, 0, 4.56, 0}, 1);

    for (auto n_par_ct : {1u, 3u}) {
        sim s({x1[0], x1[1], x1[2], v1[0], v1[1], v1[2], psize, x2[0], x2[1], x2[2], v2[0], v2[1], v2[2], psize}, 0.23,
              kw::conj_thresh = psize * 100, kw::n_par_ct = n_par_ct);

        auto sv = xt::adapt(s.get_state().data(), {2, 7});
        auto pos = xt::view(sv, xt::all(), xt::range(0, 3));

        while (s.get_conjunctions().empty()) {
            REQUIRE(s.step() == outcome::success);
        }

        REQUIRE(s.get_conjunctions().size() == 1u);
        REQUIRE(s.get_conjunctions()[0].i == 0u);
        REQUIRE(s.get_conjunctions()[0].j == 1u);
        REQUIRE(std::abs(s.get_conjunctions()[0].time - boost::math::constants::pi<double>() / 2) < 1e-4);
        REQUIRE(s.get_conjunctions()[0].dist < psize * 100);

        // Run a corresponding heyoka integration with event detection
        // and compare.
        auto dyn = dynamics::kepler();
        auto subs_dict = std::unordered_map<std::string, hy::expression>{
            {"x", "x0"_var}, {"y", "y0"_var}, {"z", "z0"_var}, {"vx", "vx0"_var}, {"vy", "vy0"_var}, {"vz", "vz0"_var}};
        for (auto &[lhs, rhs] : dyn) {
            lhs = hy::subs(lhs, subs_dict);
            rhs = hy::subs(rhs, subs_dict);
        }

        subs_dict = std::unordered_map<std::string, hy::expression>{
            {"x", "x1"_var}, {"y", "y1"_var}, {"z", "z1"_var}, {"vx", "vx1"_var}, {"vy", "vy1"_var}, {"vz", "vz1"_var}};
        for (const auto &[lhs, rhs] : dynamics::kepler()) {
            dyn.emplace_back(hy::subs(lhs, subs_dict), hy::subs(rhs, subs_dict));
        }

        auto ev_eq = ("x1"_var - "x0"_var) * ("vx1"_var - "vx0"_var) + ("y1"_var - "y0"_var) * ("vy1"_var - "vy0"_var)
                     + ("z1"_var - "z0"_var) * ("vz1"_var - "vz0"_var);

        auto ta = hy::taylor_adaptive<double>{
            dyn,
            {x1[0], x1[1], x1[2], v1[0], v1[1], v1[2], x2[0], x2[1], x2[2], v2[0], v2[1], v2[2]},
            hy::kw::t_events = {hy::t_event<double>{ev_eq}}};

        const auto oc = std::get<0>(ta.propagate_until(100));

        REQUIRE(oc == hy::taylor_outcome{-1});
        REQUIRE(std::abs(ta.get_time() - s.get_conjunctions()[0].time) < 1e-15);

        const auto &st = ta.get_state();

        REQUIRE(std::abs(std::sqrt((st[0] - st[6]) * (st[0] - st[6]) + (st[1] - st[7]) * (st[1] - st[7])
                                   + (st[2] - st[8]) * (st[2] - st[8]))
                         - s.get_conjunctions()[0].dist)
                < 1e-9);

        REQUIRE(std::abs(st[0] - s.get_conjunctions()[0].state_i[0]) < 1e-15);
        REQUIRE(std::abs(st[1] - s.get_conjunctions()[0].state_i[1]) < 1e-15);
        REQUIRE(std::abs(st[2] - s.get_conjunctions()[0].state_i[2]) < 1e-15);
        REQUIRE(std::abs(st[3] - s.get_conjunctions()[0].state_i[3]) < 1e-15);
        REQUIRE(std::abs(st[4] - s.get_conjunctions()[0].state_i[4]) < 1e-15);
        REQUIRE(std::abs(st[5] - s.get_conjunctions()[0].state_i[5]) < 1e-15);

        REQUIRE(std::abs(st[6] - s.get_conjunctions()[0].state_j[0]) < 1e-15);
        REQUIRE(std::abs(st[7] - s.get_conjunctions()[0].state_j[1]) < 1e-15);
        REQUIRE(std::abs(st[8] - s.get_conjunctions()[0].state_j[2]) < 1e-15);
        REQUIRE(std::abs(st[9] - s.get_conjunctions()[0].state_j[3]) < 1e-15);
        REQUIRE(std::abs(st[10] - s.get_conjunctions()[0].state_j[4]) < 1e-15);
        REQUIRE(std::abs(st[11] - s.get_conjunctions()[0].state_j[5]) < 1e-15);
    }
}

// Test in which a conjunction is discarded because
// it happens just after a collision event.
TEST_CASE("polar conj discard")
{
    const auto psize = 1.57e-8;
    const auto [x1, v1]
        = kep_to_cart<double>({1., .000005, boost::math::constants::pi<double>() / 2 + 1e-6, 0, 1.23, 0}, 1);
    const auto [x2, v2] = kep_to_cart<double>({1., .000005, boost::math::constants::pi<double>() / 2, 0, 4.56, 0}, 1);

    const auto [x3, v3] = kep_to_cart<double>({1. + 1e-7, 0., 0., 0., 0., 0}, 1);
    auto x4 = x3;
    x4[0] = -x4[0];
    auto v4 = v3;

    for (auto n_par_ct : {1u, 3u}) {
        sim s({x1[0], x1[1], x1[2], v1[0], v1[1], v1[2], psize, x2[0], x2[1], x2[2], v2[0], v2[1], v2[2], psize,
               x3[0], x3[1], x3[2], v3[0], v3[1], v3[2], psize, x4[0], x4[1], x4[2], v4[0], v4[1], v4[2], psize},
              0.23, kw::conj_thresh = psize * 100, kw::n_par_ct = n_par_ct);

        while (true) {
            const auto oc = s.step();

            if (oc != outcome::success) {
                REQUIRE(oc == outcome::collision);
                break;
            }
        }

        // NOTE: only the conjunction between 0 and 1 is reported.
        REQUIRE(s.get_conjunctions().size() == 1u);
        REQUIRE(s.get_conjunctions()[0].i == 0u);
        REQUIRE(s.get_conjunctions()[0].j == 1u);

        // The conjunction time must be before the simulation time.
        REQUIRE(s.get_conjunctions()[0].time < s.get_time());
    }
}

// Test multiple steps triggering conjunctions: check that the conjunctions
// are kept in chrono order and that the logic for resizing the
// conj vector works properly.
TEST_CASE("multiple conjs")
{
    const auto psize = 1.57e-8;
    const auto [x1, v1]
        = kep_to_cart<double>({1., .000005, boost::math::constants::pi<double>() / 2 + 1e-6, 0, 1.23, 0}, 1);
    const auto [x2, v2] = kep_to_cart<double>({1., .000005, boost::math::constants::pi<double>() / 2, 0, 4.56, 0}, 1);

    const auto [x3, v3] = kep_to_cart<double>({1. + 1e-7, 0., 0., 0., 0., 0}, 1);
    auto x4 = x3;
    x4[0] = -x4[0] + 1e-6;
    auto v4 = v3;

    for (auto n_par_ct : {1u, 3u}) {
        sim s({x1[0], x1[1], x1[2], v1[0], v1[1], v1[2], psize, x2[0], x2[1], x2[2], v2[0], v2[1], v2[2], psize,
               x3[0], x3[1], x3[2], v3[0], v3[1], v3[2], psize, x4[0], x4[1], x4[2], v4[0], v4[1], v4[2], psize},
              0.23, kw::conj_thresh = psize * 100, kw::n_par_ct = n_par_ct);

        while (true) {
            const auto oc = s.step();

            REQUIRE(oc == outcome::success);

            if (s.get_time() > 26.) {
                break;
            }
        }

        REQUIRE(s.get_conjunctions().size() == 8u);
        REQUIRE(std::is_sorted(s.get_conjunctions().begin(), s.get_conjunctions().end(),
                               [](const auto &c1, const auto &c2) { return c1.time < c2.time; }));
        REQUIRE(s.get_conjunctions().back().time < s.get_time());

        // Stream/format operator for the conjunction struct.
        std::ostringstream oss;
        oss << s.get_conjunctions()[0];
        REQUIRE(!oss.str().empty());

        oss.str("");
        oss << fmt::format("{}", s.get_conjunctions()[0]);
        REQUIRE(!oss.str().empty());
    }
}
