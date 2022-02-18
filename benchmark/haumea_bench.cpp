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

#include <oneapi/tbb/global_control.h>

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

    auto [a, e, i, om, Om, nu] = kep;

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
    namespace hy = heyoka;

    oneapi::tbb::global_control gc(oneapi::tbb::global_control::max_allowed_parallelism, 8);

    create_logger();

    // set_logger_level_trace();

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

    const auto ct = 5000. / 4;
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

    const auto nu = .3;

    // Perturbations on a ring particle due to Hi'iaka and
    // the non-inertial reference frame.
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

    // Create the initial conditions.
    const auto nparts = 10000ull;
    std::array<std::vector<double>, 6> state_data;

    std::uniform_real_distribution<double> a_dist(5000e3, 10000e3), e_dist(0., 1e-3),
        i_dist(0., 2. * boost::math::constants::pi<double>() / 360),
        ang_dist(0., 2 * boost::math::constants::pi<double>());

    for (auto i = 0ull; i < nparts; ++i) {
        const auto a = a_dist(rng);
        const auto e = e_dist(rng);
        const auto inc = i_dist(rng);
        const auto om = ang_dist(rng);
        const auto Om = ang_dist(rng);
        const auto nu = ang_dist(rng);

        auto [r, v] = kep_to_cart<double>({a, e, inc, om, Om, nu}, mu_c);

        state_data[0].push_back(r[0]);
        state_data[1].push_back(r[1]);
        state_data[2].push_back(r[2]);

        state_data[3].push_back(v[0]);
        state_data[4].push_back(v[1]);
        state_data[5].push_back(v[2]);
    }

    // Create the simulation.
    sim s(state_data[0], state_data[1], state_data[2], state_data[3], state_data[4], state_data[5],
          std::vector(state_data[0].size(), radius), ct, kw::dyn = dynamics,
          kw::c_radius = std::vector<double>{ra, rb, rc}, kw::d_radius = a_s * 10);

    while (true) {
        auto oc = s.step(dt);

        if (oc == outcome::collision) {
            // Fetch the indices of the collision.
            const auto [i, j] = std::get<0>(*s.get_interrupt_info());

            xt::xtensor_fixed<double, xt::xshape<3>> ri{s.get_x()[i], s.get_y()[i], s.get_z()[i]};
            xt::xtensor_fixed<double, xt::xshape<3>> rj{s.get_x()[j], s.get_y()[j], s.get_z()[j]};

            xt::xtensor_fixed<double, xt::xshape<3>> vi{s.get_vx()[i], s.get_vy()[i], s.get_vz()[i]};
            xt::xtensor_fixed<double, xt::xshape<3>> vj{s.get_vx()[j], s.get_vy()[j], s.get_vz()[j]};

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

            auto new_vx = s.get_vx();
            auto new_vy = s.get_vy();
            auto new_vz = s.get_vz();

            new_vx[i] = vi(0);
            new_vy[i] = vi(1);
            new_vz[i] = vi(2);

            new_vx[j] = vj(0);
            new_vy[j] = vj(1);
            new_vz[j] = vj(2);

            s.set_new_state(s.get_x(), s.get_y(), s.get_z(), std::move(new_vx), std::move(new_vy), std::move(new_vz),
                            s.get_sizes());
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
