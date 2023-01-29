// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>
#include <tuple>
#include <unordered_map>

#include <boost/math/constants/constants.hpp>

#include <fmt/core.h>
#include <fmt/ranges.h>

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

    sim s({x1[0], x1[1], x1[2], v1[0], v1[1], v1[2], psize, x2[0], x2[1], x2[2], v2[0], v2[1], v2[2], psize}, 0.23,
          kw::conj_thresh = psize * 100);

    auto sv = xt::adapt(s.get_state().data(), {2, 7});
    auto pos = xt::view(sv, xt::all(), xt::range(0, 3));

    const auto &conj = s.get_conjunctions();

    while (conj.empty()) {
        REQUIRE(s.step() == outcome::success);
    }

    REQUIRE(conj.size() == 1u);
    REQUIRE(std::get<0>(conj[0]) == 0u);
    REQUIRE(std::get<1>(conj[0]) == 1u);
    REQUIRE(std::abs(std::get<2>(conj[0]) - boost::math::constants::pi<double>() / 2) < 1e-4);
    REQUIRE(std::get<3>(conj[0]) < psize * 100);

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
    REQUIRE(std::abs(ta.get_time() - std::get<2>(conj[0])) < 1e-15);

    const auto &st = ta.get_state();

    REQUIRE(std::abs(std::sqrt((st[0] - st[6]) * (st[0] - st[6]) + (st[1] - st[7]) * (st[1] - st[7])
                               + (st[2] - st[8]) * (st[2] - st[8]))
                     - std::get<3>(conj[0]))
            < 1e-9);
}
