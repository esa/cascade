// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <oneapi/tbb/global_control.h>

#include <cascade/logging.hpp>
#include <cascade/sim.hpp>

template <typename T>
inline T dot(std::array<T, 3> a, std::array<T, 3> b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

template <typename T>
inline std::pair<std::array<T, 3>, std::array<T, 3>> kep_to_cart(std::array<T, 6> kep, T mu)
{
    using std::atan;
    using std::cos;
    using std::sin;
    using std::sqrt;
    using std::tan;

    auto [a, e, i, Om, om, nu] = kep;

    const auto E = 2 * atan(sqrt((1 - e) / (1 + e)) * tan(nu / 2));

    const auto n = sqrt(mu / (a * a * a));

    const std::array<T, 3> q = {a * (cos(E) - e), a * sqrt(1 - e * e) * sin(E), T(0)};
    const std::array<T, 3> vq
        = {-n * a * sin(E) / (1 - e * cos(E)), n * a * sqrt(1 - e * e) * cos(E) / (1 - e * cos(E)), T(0)};

    const std::array<T, 3> r1 = {cos(Om) * cos(om) - sin(Om) * cos(i) * sin(om),
                                 -cos(Om) * sin(om) - sin(Om) * cos(i) * cos(om), sin(Om) * sin(i)};
    const std::array<T, 3> r2 = {sin(Om) * cos(om) + cos(Om) * cos(i) * sin(om),
                                 -sin(Om) * sin(om) + cos(Om) * cos(i) * cos(om), -cos(Om) * sin(i)};
    const std::array<T, 3> r3 = {sin(i) * sin(om), sin(i) * cos(om), cos(i)};

    std::array<T, 3> x = {dot(r1, q), dot(r2, q), dot(r3, q)};
    std::array<T, 3> v = {dot(r1, vq), dot(r2, vq), dot(r3, vq)};

    return std::pair{x, v};
}

std::mt19937 rng;

using namespace cascade;

int main()
{
    create_logger();

    set_logger_level_trace();

    const auto seed = std::random_device{}();
    rng.seed(seed);
    std::cout << "Seed set to: " << seed << '\n';

    std::uniform_real_distribution<double> a_dist(1.02, 1.3), e_dist(0., 0.02), i_dist(0., 0.05),
        ang_dist(0., 2 * boost::math::constants::pi<double>()), size_dist(1.57e-8, 1.57e-7);

    std::vector<double> state;

    const auto nparts = 17378ull;

    for (auto i = 0ull; i < nparts; ++i) {
        const auto a = a_dist(rng);
        const auto e = e_dist(rng);
        const auto inc = i_dist(rng);
        const auto om = ang_dist(rng);
        const auto Om = ang_dist(rng);
        const auto nu = ang_dist(rng);

        auto [r, v] = kep_to_cart<double>({a, e, inc, Om, om, nu}, 1.);

        state.push_back(r[0]);
        state.push_back(r[1]);
        state.push_back(r[2]);

        state.push_back(v[0]);
        state.push_back(v[1]);
        state.push_back(v[2]);

        state.push_back(size_dist(rng));
    }

    auto state_data = state.data();

    // Simulation with nparts with all default values for the kwargs (keplerian dynamics ... etc..)
    sim s(std::move(state), 0.23);

    // Performs 20 steps of the simulation
    for (auto i = 0; i < 20; ++i) {
        std::cout << "\nStep: " << i << '\n';
        s.step();
    }
}
