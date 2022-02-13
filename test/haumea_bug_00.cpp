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
#include <string>
#include <vector>

#include <boost/math/constants/constants.hpp>

#include <heyoka/math/cos.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sin.hpp>
#include <heyoka/math/sum_sq.hpp>
#include <heyoka/math/time.hpp>

#include <cascade/sim.hpp>

#include "catch.hpp"

using namespace cascade;

// Test a bug in a Haumea ring simulation in which
// the simulation will hang due to a singularity being
// encountered *after* a stopping terminal event.
TEST_CASE("haumea bug 00")
{
    namespace hy = heyoka;

    std::array<std::vector<double>, 6> state_data;

    auto names = std::vector<std::string>{"haumea_00_x.txt",  "haumea_00_y.txt",  "haumea_00_z.txt",
                                          "haumea_00_vx.txt", "haumea_00_vy.txt", "haumea_00_vz.txt"};

    for (auto i = 0u; i < 6u; ++i) {
        std::fstream in(names[i]);

        std::string line;

        while (std::getline(in, line)) {
            double value;
            std::stringstream ss(line);

            ss >> value;

            state_data[i].push_back(value);
        }
    }

    // Create the dynamics.
    const auto G = 6.674e-11;

    const auto m_c = 4.006e21;
    const auto mu_c = m_c * G;
    const auto ra = 2000e3;
    const auto rb = ra;
    const auto rc = ra / 2;
    const auto Ac = 1 / 5. * m_c * (rb * rb + rc * rc);
    const auto Bc = 1 / 5. * m_c * (ra * ra + rc * rc);
    const auto Cc = 1 / 5. * m_c * (ra * ra + rb * rb);

    const auto m_s = m_c / 223;
    const auto mu_s = m_s * G;
    const auto a_s = 20000e3;
    const auto e_s = 0.05;
    const auto inc_s = 55. * 2 * boost::math::constants::pi<double>() / 360;

    const auto ct = 5000.;
    const auto radius = 10e3 / 3;
    const auto dt = 10000.;

    const auto M_s = std::sqrt((mu_c + mu_s) / (a_s * a_s * a_s)) * hy::time;

    const auto x_so = a_s * (hy::cos(M_s) - e_s * (1. + hy::sin(M_s) * hy::sin(M_s)));
    const auto y_so = a_s * (hy::sin(M_s) + e_s * hy::sin(M_s) * hy::cos(M_s));

    const auto x_s = x_so;
    const auto y_s = y_so * std::cos(inc_s);
    const auto z_s = y_so * std::sin(inc_s);

    const auto [x, y, z] = hy::make_vars("x", "y", "z");

    const auto dps_m3 = hy::pow((x_s - x) * (x_s - x) + (y_s - y) * (y_s - y) + (z_s - z) * (z_s - z), -3. / 2);

    const auto dcs_m3 = hy::pow(x_s * x_s + y_s * y_s + z_s * z_s, -3. / 2);

    // # Perturbations on a ring particle due to Hi'iaka and
    // # the non-inertial reference frame.
    auto pert_x_s = mu_s * ((x_s - x) * dps_m3 - x_s * dcs_m3);
    auto pert_y_s = mu_s * ((y_s - y) * dps_m3 - y_s * dcs_m3);
    auto pert_z_s = mu_s * ((z_s - z) * dps_m3 - z_s * dcs_m3);

    const auto I = (Ac * x * x + Bc * y * y + Cc * z * z) / hy::sum_sq({x, y, z});
    const auto Vell = G / 2. * (Ac + Bc + Cc - 3. * I) * hy::pow(hy::sum_sq({x, y, z}), -3. / 2);

    pert_x_s += hy::diff(Vell, x);
    pert_y_s += hy::diff(Vell, y);
    pert_z_s += hy::diff(Vell, z);

    auto dynamics = dynamics::kepler(mu_c);

    dynamics[3].second += pert_x_s;
    dynamics[4].second += pert_y_s;
    dynamics[5].second += pert_z_s;

    sim s(state_data[0], state_data[1], state_data[2], state_data[3], state_data[4], state_data[5],
          std::vector(state_data[0].size(), radius), ct, kw::dyn = dynamics,
          kw::c_radius = std::vector<double>{ra, rb, rc}, kw::d_radius = a_s * 10);

    s.set_time(13401085490.242563);

    auto oc = s.step(dt);

    // NOTE: we just want to check that the step finished without error and without hanging.
    REQUIRE(oc != outcome::err_nf_state);
    REQUIRE(oc != outcome::time_limit);
}
