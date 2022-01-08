// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/format.h>

#include <spdlog/stopwatch.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/math/sum_sq.hpp>
#include <heyoka/taylor.hpp>

#include <cascade/detail/logging_impl.hpp>
#include <cascade/detail/sim_data.hpp>
#include <cascade/sim.hpp>

#if defined(_MSC_VER) && !defined(__clang__)

// NOTE: MSVC has issues with the other "using"
// statement form.
using namespace fmt::literals;

#else

using fmt::literals::operator""_format;

#endif

namespace cascade
{

sim::sim()
    : sim(std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{},
          std::vector<double>{}, std::vector<double>{})
{
}

// TODO fill in copy/move ctor/assignment ops.
// sim::sim(sim &&) noexcept = default;

sim::~sim()
{
    std::unique_ptr<sim_data> tmp_ptr(m_data);
}

void sim::finalise_ctor()
{
    namespace hy = heyoka;

    auto *logger = detail::get_logger();

    // Check consistency of the particles' state vectors.
    const auto nparts = m_x.size();

    if (m_y.size() != nparts) {
        throw std::invalid_argument("Inconsistent number of particles detected: the number of x coordinates is {}, "
                                    "but the number of y coordinates is {}"_format(nparts, m_y.size()));
    }

    if (m_z.size() != nparts) {
        throw std::invalid_argument("Inconsistent number of particles detected: the number of x coordinates is {}, "
                                    "but the number of z coordinates is {}"_format(nparts, m_z.size()));
    }

    if (m_vx.size() != nparts) {
        throw std::invalid_argument("Inconsistent number of particles detected: the number of x coordinates is {}, "
                                    "but the number of x velocities is {}"_format(nparts, m_vx.size()));
    }

    if (m_vy.size() != nparts) {
        throw std::invalid_argument("Inconsistent number of particles detected: the number of x coordinates is {}, "
                                    "but the number of y velocities is {}"_format(nparts, m_vy.size()));
    }

    if (m_vz.size() != nparts) {
        throw std::invalid_argument("Inconsistent number of particles detected: the number of x coordinates is {}, "
                                    "but the number of z velocities is {}"_format(nparts, m_vz.size()));
    }

    std::optional<hy::taylor_adaptive<double>> s_ta;
    std::optional<hy::taylor_adaptive_batch<double>> b_ta;

    auto integrators_setup = [&s_ta, &b_ta]() {
        // Set up the dynamics.
        auto [x, y, z, vx, vy, vz, r] = hy::make_vars("x", "y", "z", "vx", "vy", "vz", "r");

        const auto dynamics = std::vector<std::pair<hy::expression, hy::expression>>{
            hy::prime(x) = vx,
            hy::prime(y) = vy,
            hy::prime(z) = vz,
            hy::prime(vx) = -x * hy::pow(hy::sum_sq({x, y, z}), -1.5),
            hy::prime(vy) = -y * hy::pow(hy::sum_sq({x, y, z}), -1.5),
            hy::prime(vz) = -z * hy::pow(hy::sum_sq({x, y, z}), -1.5),
            hy::prime(r) = hy::sum({x * vx, y * vy, z * vz}) / r};

        // TODO overflow checks.
        const std::uint32_t batch_size = hy::recommended_simd_size<double>();
        oneapi::tbb::parallel_invoke(
            [&]() { s_ta.emplace(dynamics, std::vector<double>(7u)); },
            [&]() { b_ta.emplace(dynamics, std::vector<double>(7u * batch_size), batch_size); });
    };

    // Helper to check that all values in a vector
    // are finite.
    auto finite_checker = [](const auto &v) {
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(v.begin(), v.end()), [](const auto &range) {
            for (const auto &val : range) {
                if (!std::isfinite(val)) {
                    throw std::domain_error("The non-finite value {} was detected in the particle states"_format(val));
                }
            }
        });
    };

    spdlog::stopwatch sw;

    oneapi::tbb::parallel_invoke(
        integrators_setup, [&finite_checker, this]() { finite_checker(m_x); },
        [&finite_checker, this]() { finite_checker(m_y); }, [&finite_checker, this]() { finite_checker(m_z); },
        [&finite_checker, this]() { finite_checker(m_vx); }, [&finite_checker, this]() { finite_checker(m_vy); },
        [&finite_checker, this]() { finite_checker(m_vz); },
        [this, nparts]() {
            // Compute the initial values of the radiuses.
            m_r.resize(nparts);

            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(size_type(0), nparts), [this](const auto &range) {
                for (auto i = range.begin(); i != range.end(); ++i) {
                    m_r[i] = std::sqrt(m_x[i] * m_x[i] + m_y[i] * m_y[i] + m_z[i] * m_z[i]);

                    if (!std::isfinite(m_r[i])) {
                        throw std::domain_error(
                            "The non-finite value {} was detected in the particle states"_format(m_r[i]));
                    }
                }
            });
        });

    logger->trace("Integrators setup time: {}s", sw);

    auto data_ptr = std::make_unique<sim_data>(std::move(*s_ta), std::move(*b_ta));
    m_data = data_ptr.release();
}

} // namespace cascade
