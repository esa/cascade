// Copyright 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>
#include <vector>

#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xview.hpp>

#include <cascade/sim.hpp>

#include "catch.hpp"

using namespace cascade;

TEST_CASE("domain exit")
{
    using Catch::Detail::Approx;

    for (auto n_par_ct : {1u, 3u}) {
        sim s({1., 0, 0, 150., 0., 0., 0}, 0.23, kw::exit_radius = 10., kw::n_par_ct = n_par_ct);

        auto sv = xt::adapt(s.get_state_data(), {1, 7});
        auto pos = xt::view(sv, xt::all(), xt::range(0, 3));

        auto oc = s.propagate_until(1000.);

        REQUIRE(oc == outcome::exit);

        auto x = pos(0, 0);
        auto y = pos(0, 1);
        auto z = pos(0, 2);

        REQUIRE(std::sqrt(x * x + y * y + z * z) == Approx(10.).epsilon(0.).margin(1e-14));

        sv = xt::xarray<double>{{1., 0, 0, 150, 150., 0., 0.}};

        s.set_time(0.);

        oc = s.propagate_until(1000.);

        REQUIRE(oc == outcome::exit);

        x = pos(0, 0);
        y = pos(0, 1);
        z = pos(0, 2);

        REQUIRE(std::sqrt(x * x + y * y + z * z) == Approx(10.).epsilon(0.).margin(1e-14));

        // Ensure correctness also when using the batch integrator.
        s.set_new_state_pars({1.1, 0, 0, 0, .953, 0, 0, 1.5, 0, 0, 0, .953, 0, 0, 1.1, 0, 0, 0, .953, 0, 0,
                              1.1, 0, 0, 0, 150,  0, 0, 1.1, 0, 0, 0, .953, 0, 0, 1.1, 0, 0, 0, .953, 0, 0});

        s.set_time(0.);

        oc = s.propagate_until(1000);

        REQUIRE(oc == outcome::exit);
        REQUIRE(std::get<1>(*s.get_interrupt_info()) == 3u);

        auto sv2 = xt::adapt(s.get_state_data(), {6, 7});
        auto pos2 = xt::view(sv2, xt::all(), xt::range(0, 3));

        x = pos2(3, 0);
        y = pos2(3, 1);
        z = pos2(3, 2);

        REQUIRE(std::sqrt(x * x + y * y + z * z) == Approx(10.).epsilon(0.).margin(1e-14));
    }
}
