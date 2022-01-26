// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cmath>
#include <initializer_list>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <heyoka/expression.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sum_sq.hpp>

#include <cascade/sim.hpp>

namespace cascade::dynamics
{

std::vector<std::pair<heyoka::expression, heyoka::expression>> kepler(double mu)
{
    namespace hy = heyoka;

    if (!std::isfinite(mu) || mu <= 0) {
        throw std::invalid_argument(fmt::format(
            "The mu parameter for Keplerian dynamics must be finite and positive, but it is {} instead", mu));
    }

    auto [x, y, z, vx, vy, vz] = hy::make_vars("x", "y", "z", "vx", "vy", "vz");

    return {hy::prime(x) = vx,
            hy::prime(y) = vy,
            hy::prime(z) = vz,
            hy::prime(vx) = -mu * x * hy::pow(hy::sum_sq({x, y, z}), -1.5),
            hy::prime(vy) = -mu * y * hy::pow(hy::sum_sq({x, y, z}), -1.5),
            hy::prime(vz) = -mu * z * hy::pow(hy::sum_sq({x, y, z}), -1.5)};
}

} // namespace cascade::dynamics
