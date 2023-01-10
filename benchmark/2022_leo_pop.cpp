// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

#include <heyoka/math/cos.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sum_sq.hpp>
#include <heyoka/math/time.hpp>

#include <cascade/logging.hpp>
#include <cascade/sim.hpp>

std::mt19937 rng;

using namespace cascade;

int main()
{
    namespace hy = heyoka;
    set_logger_level_trace();

    // Constuct the initial state from file "test_ic_19647.txt". This was created
    // using all catalogued objects from spacetrack as of 2022
    std::vector<double> state;
    std::string line;
    std::ifstream file_in;
    file_in.open("test_ic_19647.txt");
    if (file_in.is_open()) {
        while (std::getline(file_in, line)) {
            double value = 0;
            std::stringstream ss(line);
            ss >> value;
            state.push_back(value);
        }
    } else {
        std::cout << " unable to open file test_ic_19647.txt" << std::endl;
    }

    // Create the dynamics
    // Dynamical variables.
    const auto [x, y, z, vx, vy, vz] = hy::make_vars("x", "y", "z", "vx", "vy", "vz");

    // Constants.
    const auto GMe = 398600441800000.0;
    const auto C20 = -4.84165371736e-4;
    const auto C22 = 2.43914352398e-6;
    const auto S22 = -1.40016683654e-6;
    const auto Re = 6378137.0;

    // Create Keplerian dynamics.
    auto dyn = dynamics::kepler(GMe);

    // Add the J2 terms.
    auto magr2 = hy::sum_sq({x, y, z});
    auto J2term1 = GMe * (Re * Re) * std::sqrt(5) * C20 / (2. * hy::pow(magr2, 0.5));
    auto J2term2 = 3. / (magr2 * magr2);
    auto J2term3 = 15. * (z * z) / (magr2 * magr2 * magr2);
    auto fJ2x = J2term1 * x * (J2term2 - J2term3);
    auto fJ2y = J2term1 * y * (J2term2 - J2term3);
    auto fJ2z = J2term1 * z * (3. * J2term2 - J2term3);

    dyn[3].second += fJ2x;
    dyn[4].second += fJ2y;
    dyn[5].second += fJ2z;

    // Create the simulation.
    auto ct = 0.23 * 806.81;
    sim s(state, ct, kw::dyn = dyn);

    while (true) {
        auto oc = s.step();

        if (oc == outcome::collision) {
            // Fetch the indices of the collision.
            const auto [i, j] = std::get<0>(*s.get_interrupt_info());

            xt::xtensor_fixed<double, xt::xshape<3>> ri{s.get_state()[i * 7], s.get_state()[i * 7 + 1],
                                                        s.get_state()[i * 7 + 2]};
            xt::xtensor_fixed<double, xt::xshape<3>> rj{s.get_state()[j * 7], s.get_state()[j * 7 + 1],
                                                        s.get_state()[j * 7 + 2]};

            xt::xtensor_fixed<double, xt::xshape<3>> vi{s.get_state()[i * 7 + 3], s.get_state()[i * 7 + 4],
                                                        s.get_state()[i * 7 + 5]};
            xt::xtensor_fixed<double, xt::xshape<3>> vj{s.get_state()[j * 7 + 3], s.get_state()[j * 7 + 4],
                                                        s.get_state()[j * 7 + 5]};

            auto rij = rj - ri;
            auto uij = rij / xt::linalg::norm(rij);

            auto vu_i = xt::linalg::dot(vi, uij);
            auto vu_j = xt::linalg::dot(vj, uij);

            auto new_vu_i = (nu * (vu_j - vu_i) + vu_i + vu_j) / 2.;
            auto new_vu_j = (nu * (vu_i - vu_j) + vu_i + vu_j) / 2.;
            vi -= vu_i * uij;
            vj -= vu_j * uij;
            vi += new_vu_i * uij;
            vj += new_vu_j * uij;

            auto new_state = s.get_state();

            new_state[7 * i + 3] = vi(0);
            new_state[7 * i + 4] = vi(1);
            new_state[7 * i + 5] = vi(2);

            new_state[7 * j + 3] = vj(0);
            new_state[7 * j + 4] = vj(1);
            new_state[7 * j + 5] = vj(2);

            s.set_new_state(new_state);
        } else if (oc != outcome::success) {
            std::cout << "Interrupting due to terminal event detected\n";
            break;
        }

        if (s.get_time() > 30. * 86400) {
            std::cout << "Final time reached, exiting\n";
            break;
        }
    }
}
