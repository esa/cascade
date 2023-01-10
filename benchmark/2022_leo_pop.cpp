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
#include <xtensor/xadapt.hpp>
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

constexpr double pi = boost::math::constants::pi<double>();

std::vector<double> read_file(std::string filename)
{
    std::vector<double> retval;
    std::string line;
    std::ifstream file_in;
    file_in.open(filename);
    if (file_in.is_open()) {
        while (std::getline(file_in, line)) {
            double value = 0;
            std::stringstream ss(line);
            ss >> value;
            retval.push_back(value);
        }
    } else {
        std::cout << " unable to open file test_ic_19647.txt" << std::endl;
    }
    return retval;
}

void remove_particles(std::vector<double> &state, std::vector<double> &pars, const std::vector<std::size_t> &idxs)
{
    auto idxs_copy = idxs;
    std::sort(idxs_copy.begin(), idxs_copy.end(), std::greater<int>());
    for (auto idx : idxs_copy) {
        state.erase(state.begin() + idx);
        pars.erase(pars.begin() + idx);
    }
}

using namespace cascade;

int main()
{
    namespace hy = heyoka;
    set_logger_level_trace();

    // Constuct the initial state from file "test_ic_19647.txt". This was created
    // using all catalogued objects from spacetrack as of 2022
    std::vector<double> state;
    state = read_file("test_ic_19647.txt");
    std::vector<double> c_sections;
    c_sections = read_file("test_par_19647.txt");

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

    // Add the Earth's C22 and S22 terms.
    // This value represents the rotation of the Earth fixed system at t0
    auto theta_g = (pi / 180) * 280.4606;
    // This value represents the magnitude of the Earth rotation
    auto nu_e = (pi / 180) * (4.178074622024230e-3);

    auto X = x * hy::cos(theta_g + nu_e * hy::time) + y * hy::sin(theta_g + nu_e * hy::time);
    auto Y = -x * hy::sin(theta_g + nu_e * hy::time) + y * hy::cos(theta_g + nu_e * hy::time);
    auto Z = z;

    auto C22term1 = 5. * GMe * (Re * Re) * std::sqrt(15) * C22 / (2. * hy::pow(magr2, (7. / 2)));
    auto C22term2 = GMe * (Re * Re) * std::sqrt(15) * C22 / (hy::pow(magr2, (5. / 2)));
    auto fC22X = C22term1 * X * (Y * Y - X * X) + C22term2 * X;
    auto fC22Y = C22term1 * Y * (Y * Y - X * X) - C22term2 * Y;
    auto fC22Z = C22term1 * Z * (Y * Y - X * X);

    auto S22term1 = 5 * GMe * (Re * Re) * std::sqrt(15) * S22 / (hy::pow(magr2, (7. / 2)));
    auto S22term2 = GMe * (Re * Re) * std::sqrt(15) * S22 / (hy::pow(magr2, (5. / 2)));
    auto fS22X = -S22term1 * (X * X) * Y + S22term2 * Y;
    auto fS22Y = -S22term1 * X * (Y * Y) + S22term2 * X;
    auto fS22Z = -S22term1 * X * Y * Z;

    auto fC22x = fC22X * hy::cos(theta_g + nu_e * hy::time) - fC22Y * hy::sin(theta_g + nu_e * hy::time);
    auto fC22y = fC22X * hy::sin(theta_g + nu_e * hy::time) + fC22Y * hy::cos(theta_g + nu_e * hy::time);
    auto fC22z = fC22Z;

    auto fS22x = fS22X * hy::cos(theta_g + nu_e * hy::time) - fS22Y * hy::sin(theta_g + nu_e * hy::time);
    auto fS22y = fS22X * hy::sin(theta_g + nu_e * hy::time) + fS22Y * hy::cos(theta_g + nu_e * hy::time);
    auto fS22z = fS22Z;

    dyn[3].second += fC22x + fS22x;
    dyn[4].second += fC22y + fS22y;
    dyn[5].second += fC22z + fS22z;

    // Create the simulation.
    auto ct = 0.23 * 806.81;
    sim s(state, ct, kw::dyn = dyn);

    // Performs 20 steps of the simulation
    for (auto step = 0; step < 20; ++step) {
        auto oc = s.step();

        if (oc == outcome::collision) {
            // Fetch the indices of the collision.
            const auto [i, j] = std::get<0>(*s.get_interrupt_info());

            std::cout << "Collision detected, deleting particles " << i << " and " << j << std::endl;

            auto new_state = s.get_state();
            auto new_pars = s.get_pars();
            remove_particles(new_state, new_pars, {i, j});
            s.set_new_state(new_state); // this will also reset all pars to zero and resize
            std::vector<std::size_t> shape = {s.get_nparts()};
            auto pars = xt::adapt(s.get_pars_data(), shape);
            pars = xt::adapt(new_pars.data(), shape);
        } else if (oc == outcome::reentry) {
            // Fetch the index of the re-entry.
            const auto i = std::get<1>(*s.get_interrupt_info());

            std::cout << "Reentry detected, particle " << i << std::endl;

            auto new_state = s.get_state();
            auto new_pars = s.get_pars();
            remove_particles(new_state, new_pars, {i});
            s.set_new_state(new_state); // this will also reset all pars to zero and resize
            std::vector<std::size_t> shape = {s.get_nparts()};
            auto pars = xt::adapt(s.get_pars_data(), shape);
            pars = xt::adapt(new_pars.data(), shape);
        }
    }
}
