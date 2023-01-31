// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <fstream>
#include <initializer_list>
#include <string>
#include <vector>

#include <cascade/sim.hpp>

#include "catch.hpp"

using namespace cascade;

// Test for bug in the tree construction
// when a node contains particles differing
// only in the last bit (i.e., the least
// significant one).
TEST_CASE("last bit flip")
{
    const auto GMe = 398600441800000.0;

    for (auto n_par_ct : {1u, 3u}) {
        std::array<std::vector<double>, 7> data;

        auto names = std::vector<std::string>{"x_lbf.txt",  "y_lbf.txt",  "z_lbf.txt",    "vx_lbf.txt",
                                              "vy_lbf.txt", "vz_lbf.txt", "sizes_lbf.txt"};

        for (auto i = 0u; i < 7u; ++i) {
            std::fstream in(names[i]);

            std::string line;

            while (std::getline(in, line)) {
                double value = 0;
                std::stringstream ss(line);

                ss >> value;

                data[i].push_back(value);
            }
        }

        std::vector<double> state;
        for (decltype(state.size()) i = 0; i < data[0].size(); ++i) {
            state.push_back(data[0][i]);
            state.push_back(data[1][i]);
            state.push_back(data[2][i]);
            state.push_back(data[3][i]);
            state.push_back(data[4][i]);
            state.push_back(data[5][i]);
            state.push_back(data[6][i]);
        }

        sim s(state, 0.23 * 806.81, kw::dyn = dynamics::kepler(GMe), kw::n_par_ct = n_par_ct);

        auto oc = s.step();
        REQUIRE((oc == outcome::success || oc == outcome::collision));
    }
}
