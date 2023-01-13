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

#include <oneapi/tbb/global_control.h>

#include <boost/math/constants/constants.hpp>
#include <boost/program_options.hpp>

#include <xtensor/xadapt.hpp>

#include <heyoka/math/cos.hpp>
#include <heyoka/math/exp.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sqrt.hpp>
#include <heyoka/math/sum_sq.hpp>
#include <heyoka/math/time.hpp>

#include <cascade/logging.hpp>
#include <cascade/sim.hpp>

using namespace cascade;

constexpr double pi = boost::math::constants::pi<double>();

// Helper to read a file into an std vector. Assumes file formatted in one column
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

// Helper to remove from state and pars all the particles ids contained in a list.
void remove_particles(std::vector<double> &state, std::vector<double> &pars,
                      const std::vector<cascade::sim::size_type> &idxs)
{
    auto idxs_copy = idxs;
    std::sort(idxs_copy.begin(), idxs_copy.end(), std::greater<>());
    for (auto idx : idxs_copy) {
        state.erase(state.begin() + 7 * idx, state.begin() + 7 * idx + 7);
        pars.erase(pars.begin() + idx);
    }
}

// Helper returning the heyoka expression for the density using
// the results from the data interpolation
/*
    Returns the heyoka expression for the atmosheric density in kg.m^3.
    Input is the altitude in m.
    (when we fitted km were used here we change as to allow better expressions)
*/
heyoka::expression compute_density(heyoka::expression h, const std::vector<double> &best_x)
{

    heyoka::expression retval(0.);
    for (auto i = 0u; i < 4u; ++i) {
        double alpha = best_x[i];
        double beta = best_x[i + 4] / 1000;
        double gamma = best_x[i + 8] * 1000;
        retval += alpha * heyoka::exp(-(h - gamma) * beta);
    }
    return retval;
}

/*
This benchmark uses real LEO population (pre-computed) and runs a few steps on them

Allowed options:
  --help                                produce help message
  -n [ --cpus ] arg (=1)                set number of cpus to use
  -s [ --steps ] arg (=20)              set number of steps to perform
  -l [ --large ] arg (=0)               augments with >500000 small debris
  -r [ --rcs_factor ] arg (=1)          factor for the radius (collisions)
  -c [ --c_timestep ] arg (=64.544799999999995)
                                        collisional time step
  -S [ --s_size ] arg                   superstep size
*/

int main(int ac, char *av[])
{
    namespace hy = heyoka;
    set_logger_level_trace();

    // Program options
    // ------------------------------------------------------------------------------------------------------
    namespace po = boost::program_options;

    po::options_description desc("Allowed options");
    desc.add_options()("help", "produce help message")("cpus,n", po::value<int>()->default_value(1),
                                                       "set number of cpus to use")(
        "steps,s", po::value<int>()->default_value(20), "set number of steps to perform")(
        "large,l", po::value<bool>()->default_value(false), "augments with >500000 small debris")(
        "rcs_factor,r", po::value<double>()->default_value(1.), "factor for the radius (collisions)")(
        "c_timestep,c", po::value<double>()->default_value(64.5448),
        "collisional time step")("s_size,S", po::value<double>(), "superstep size");

    po::variables_map vm;
    po::store(po::parse_command_line(ac, av, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << desc << "\n";
        return 1;
    }

    int n_cpus, max_steps;
    if (vm.count("cpus")) {
        n_cpus = vm["cpus"].as<int>();
    }
    std::optional<oneapi::tbb::global_control> tbb_gc;
    tbb_gc.emplace(oneapi::tbb::global_control::max_allowed_parallelism, n_cpus);

    if (vm.count("steps")) {
        max_steps = vm["steps"].as<int>();
    }

    bool large_dataset;
    if (vm.count("large")) {
        large_dataset = vm["large"].as<bool>();
    }

    double rcs_factor;
    if (vm.count("rcs_factor")) {
        rcs_factor = vm["rcs_factor"].as<double>();
    }

    double c_timestep;
    if (vm.count("c_timestep")) {
        c_timestep = vm["c_timestep"].as<double>();
    }

    double s_size;
    if (vm.count("s_size")) {
        s_size = vm["s_size"].as<double>();
    }

    std::cout << "\nRunning " << max_steps << " steps with " << n_cpus << " cpus\n"
              << (large_dataset ? "Large" : "Small") << " dataset used\nRadius factor: " << rcs_factor
              << "\nCollisional time-step: " << c_timestep << "s\nSuper step size: ";
    if (vm.count("s_size")) {
        std::cout << s_size << "s";
    } else {
        std::cout << "auto";
    }
    std::cout << std::endl;

    // Construct the initial state from files. This was created using all catalogued objects from spacetrack as of 2022.
    // The large database instead adds also the debris as to match the LADDS test case.
    // ------------------------------------------------------------------------------------------------------
    std::cout << "\nReading data from files..." << std::endl;
    std::vector<double> state, pars;
    double drag_factor;
    if (large_dataset) {
        state = read_file("test_ic_612813.txt");
        pars = read_file("test_par_612813.txt");
        // We switch off drag as to avoid to see too many reentries (in connection to the event c_radius being halved)
        drag_factor = 0.;
    } else {
        state = read_file("test_ic_19647.txt");
        pars = read_file("test_par_19647.txt");
        drag_factor = 1;
    }
    // Account for factors
    for (decltype(state.size()) i = 6; i < state.size(); i = i + 7) {
        state[i] = state[i] * rcs_factor;
    }
    for (decltype(pars.size()) i = 0; i < pars.size(); ++i) {
        pars[i] = pars[i] * drag_factor;
    }
    std::vector<double> best_x;
    best_x = read_file("best_fit_density.txt");

    // Create the dynamics
    // ------------------------------------------------------------------------------------------------------

    // Dynamical variables.
    const auto [x, y, z, vx, vy, vz] = hy::make_vars("x", "y", "z", "vx", "vy", "vz");

    // Constants.
    const double GMe = 398600441800000.0;
    const double C20 = -4.84165371736e-4;
    const double C22 = 2.43914352398e-6;
    const double S22 = -1.40016683654e-6;
    const double Re = 6378137.0;
    const double min_radius = Re + 150000.;

    // Create Keplerian dynamics.
    auto dyn = dynamics::kepler(GMe);

    // Add the J2 terms.
    auto magr2 = hy::sum_sq({x, y, z});
    auto J2term1 = GMe * (Re * Re) * std::sqrt(5) * C20 / (2. * hy::sqrt(magr2));
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

    // Add the drag force.
    auto magv2 = hy::sum_sq({vx, vy, vz});
    auto magv = hy::sqrt(magv2);
    // Here we consider a spherical Earth ... would be easy to account for the oblateness effect on the altitude.
    auto altitude = hy::sqrt(magr2) - Re;
    auto density = compute_density(altitude, best_x);
    auto ref_density = 0.1570 / Re;
    auto fdrag = density / ref_density * hy::par[0] * magv;
    auto fdragx = -fdrag * vx;
    auto fdragy = -fdrag * vy;
    auto fdragz = -fdrag * vz;
    dyn[3].second += fC22x + fdragx;
    dyn[4].second += fC22y + fdragy;
    dyn[5].second += fC22z + fdragz;

    // Create the simulation.
    // ------------------------------------------------------------------------------------------------------

    std::cout << "\nConstructing the simulation object..." << std::endl;
    double c_rad;
    if (large_dataset) {
        c_rad = min_radius / 2;
    } else {
        c_rad = min_radius;
    }
    sim s(state, c_timestep, kw::dyn = dyn, kw::pars = pars, kw::c_radius = c_rad);
    // Perform steps of the simulation.
    // ------------------------------------------------------------------------------------------------------
    outcome oc;
    std::cout << "\nPerforming the simulation..." << std::endl;
    for (auto step = 0; step < max_steps; ++step) {
        std::cout << "\nStep: " << step << '\n';
        std::cout << "Time: " << s.get_time() / 60 / 60 / 24 << '\n';

        // Superstep is only used if passed as argument, else automatically detected.
        if (vm.count("s_size")) {
            oc = s.step(s_size);
        } else {
            oc = s.step();
        }
        if (oc == outcome::collision) {
            // Fetch the indices of the collision.
            const auto [i, j] = std::get<0>(*s.get_interrupt_info());

            std::cout << "\nCollision detected, deleting particles " << i << " and " << j << std::endl;

            auto new_state = s.get_state();
            auto new_pars = s.get_pars();
            remove_particles(new_state, new_pars, {i, j});
            s.set_new_state(new_state); // this will also reset all pars to zero and resize
            std::vector<cascade::sim::size_type> shape = {s.get_nparts()};
            auto pars = xt::adapt(s.get_pars_data(), shape);
            pars = xt::adapt(new_pars.data(), shape);
        } else if (oc == outcome::reentry) {
            // Fetch the index of the re-entry.
            const auto i = std::get<1>(*s.get_interrupt_info());

            std::cout << "\nReentry detected, particle " << i << std::endl;

            auto new_state = s.get_state();
            auto new_pars = s.get_pars();
            remove_particles(new_state, new_pars, {i});
            s.set_new_state(new_state); // this will also reset all pars to zero and resize
            std::vector<cascade::sim::size_type> shape = {s.get_nparts()};
            auto pars = xt::adapt(s.get_pars_data(), shape);
            pars = xt::adapt(new_pars.data(), shape);
        }
    }
}
