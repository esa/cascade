// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <fmt/format.h>

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

// Helper to compute the begin and end of a chunk within
// a superstep for a given collisional timestep.
std::array<double, 2> sim::sim_data::get_chunk_begin_end(unsigned chunk_idx, double ct) const
{
    assert(nchunks > 0u);
    assert(std::isfinite(delta_t) && delta_t > 0);
    assert(std::isfinite(ct) && ct > 0);

    auto cbegin = ct * chunk_idx;
    // NOTE: for the last chunk we force the ending
    // at delta_t.
    auto cend = (chunk_idx == nchunks - 1u) ? delta_t : (ct * (chunk_idx + 1u));

    if (!std::isfinite(cbegin) || !std::isfinite(cend) || !(cend > cbegin) || cbegin < 0 || cend > delta_t) {
        throw std::invalid_argument(fmt::format("Invalid chunk range [{}, {})", cbegin, cend));
    }

    return {cbegin, cend};
}

sim::sim()
    : sim(std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, std::vector<double>{},
          std::vector<double>{}, std::vector<double>{}, std::vector<double>{}, 1)
{
}

sim::sim(const sim &other)
    : m_x(other.m_x), m_y(other.m_y), m_z(other.m_z), m_vx(other.m_vx), m_vy(other.m_vy), m_vz(other.m_vz),
      m_sizes(other.m_sizes), m_ct(other.m_ct), m_int_info(other.m_int_info)
{
    // For m_data, we will be copying only:
    // - the integrator templates,
    // - the llvm state.
    auto data_ptr = std::make_unique<sim_data>(other.m_data->s_ta, other.m_data->b_ta, other.m_data->state);

    // Need to assign the JIT function pointers.
    data_ptr->pta = reinterpret_cast<decltype(data_ptr->pta)>(data_ptr->state.jit_lookup("poly_translate_a"));
    data_ptr->pssdiff3 = reinterpret_cast<decltype(data_ptr->pssdiff3)>(data_ptr->state.jit_lookup("poly_ssdiff3"));
    data_ptr->fex_check = reinterpret_cast<decltype(data_ptr->fex_check)>(data_ptr->state.jit_lookup("fex_check"));
    data_ptr->rtscc = reinterpret_cast<decltype(data_ptr->rtscc)>(data_ptr->state.jit_lookup("poly_rtscc"));
    // NOTE: this is implicitly added by llvm_add_poly_rtscc().
    data_ptr->pt1 = reinterpret_cast<decltype(data_ptr->pt1)>(data_ptr->state.jit_lookup("poly_translate_1"));

    // Assign the pointer.
    m_data = data_ptr.release();
}

// Move everything, then destroy the m_data pointer in other.
sim::sim(sim &&other) noexcept
    : m_x(std::move(other.m_x)), m_y(std::move(other.m_y)), m_z(std::move(other.m_z)), m_vx(std::move(other.m_vx)),
      m_vy(std::move(other.m_vy)), m_vz(std::move(other.m_vz)), m_sizes(std::move(other.m_sizes)),
      m_ct(std::move(other.m_ct)), m_data(other.m_data), m_int_info(std::move(other.m_int_info))
{
    other.m_data = nullptr;
}

sim &sim::operator=(const sim &other)
{
    if (this != &other) {
        *this = sim(other);
    }

    return *this;
}

sim &sim::operator=(sim &&other) noexcept
{
    // Assign everything, then destroy the m_data pointer in other.
    m_x = std::move(other.m_x);
    m_y = std::move(other.m_y);
    m_z = std::move(other.m_z);
    m_vx = std::move(other.m_vx);
    m_vy = std::move(other.m_vy);
    m_vz = std::move(other.m_vz);
    m_sizes = std::move(other.m_sizes);

    m_ct = std::move(other.m_ct);

    m_data = other.m_data;

    m_int_info = std::move(other.m_int_info);

    other.m_data = nullptr;

    return *this;
}

sim::~sim()
{
    // NOTE: well-defined if m_data is null.
    std::unique_ptr<sim_data> tmp_ptr(m_data);
}

double sim::get_ct() const
{
    return m_ct;
}

void sim::set_ct(double ct)
{
    if (!std::isfinite(ct) || ct <= 0) {
        throw std::invalid_argument(
            fmt::format("The collisional timestep must be finite and positive, but it is {} instead", ct));
    }

    m_ct = ct;
}

void sim::set_new_state_impl(std::array<std::vector<double>, 7> &new_state)
{
    // Check the new state.
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<decltype(new_state.size())>(0, new_state.size()),
        [new_nparts = new_state[0].size(), &new_state](const auto &range) {
            for (auto i = range.begin(); i != range.end(); ++i) {
                // Check size consistency.
                if (new_state[i].size() != new_nparts) {
                    throw std::invalid_argument(
                        "An invalid new state was specified: the number of particles is not the "
                        "same for all the state vectors");
                }

                // Check finiteness and, for sizes, non-negativity.
                oneapi::tbb::parallel_for(
                    oneapi::tbb::blocked_range(new_state[i].begin(), new_state[i].end()), [i](const auto &r2) {
                        for (const auto &val : r2) {
                            if (!std::isfinite(val)) {
                                throw std::invalid_argument(fmt::format(
                                    "The non-finite value {} was detected in the new particle states", val));
                            }

                            if (i == 6u && val < 0) {
                                throw std::invalid_argument(fmt::format(
                                    "The negative particle radius {} was detected in the new particle states", val));
                            }
                        }
                    });
            }
        });

    // Move it in.
    m_x = std::move(new_state[0]);
    m_y = std::move(new_state[1]);
    m_z = std::move(new_state[2]);
    m_vx = std::move(new_state[3]);
    m_vy = std::move(new_state[4]);
    m_vz = std::move(new_state[5]);
    m_sizes = std::move(new_state[6]);
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

    if (m_sizes.size() != nparts) {
        throw std::invalid_argument("Inconsistent number of particles detected: the number of x coordinates is {}, "
                                    "but the number of particle radiuses is {}"_format(nparts, m_sizes.size()));
    }

    if (!std::isfinite(m_ct) || m_ct <= 0) {
        throw std::invalid_argument(
            fmt::format("The collisional timestep must be finite and positive, but it is {} instead", m_ct));
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

        const std::uint32_t batch_size = hy::recommended_simd_size<double>();
        oneapi::tbb::parallel_invoke(
            [&]() { s_ta.emplace(dynamics, std::vector<double>(7u)); },
            [&]() {
                if (batch_size > std::numeric_limits<std::vector<double>::size_type>::max() / 7u) {
                    throw std::overflow_error(
                        "An overflow as detected during the construction of the batch integrator");
                }

                b_ta.emplace(dynamics, std::vector<double>(7u * batch_size), batch_size);
            });
    };

    // Helper to check that all values in a vector
    // are finite.
    auto finite_checker = [](const auto &v) {
        oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(v.begin(), v.end()), [](const auto &range) {
            for (const auto &val : range) {
                if (!std::isfinite(val)) {
                    throw std::invalid_argument(
                        "The non-finite value {} was detected in the particle states"_format(val));
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
        [this]() {
            // NOTE: for the particle sizes, we also check that no size is negative.
            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range(m_sizes.begin(), m_sizes.end()), [](const auto &range) {
                    for (const auto &val : range) {
                        if (!std::isfinite(val)) {
                            throw std::invalid_argument("A non-finite particle radius of {} was detected"_format(val));
                        }

                        if (val < 0) {
                            throw std::invalid_argument("A negative particle radius of {} was detected"_format(val));
                        }
                    }
                });
        });

    logger->trace("Integrators setup time: {}s", sw);

    auto data_ptr = std::make_unique<sim_data>(std::move(*s_ta), std::move(*b_ta));
    m_data = data_ptr.release();

    sw.reset();

    add_jit_functions();

    logger->trace("JIT functions setup time: {}s", sw);
}

double sim::get_time() const
{
    return static_cast<double>(m_data->time);
}

std::ostream &operator<<(std::ostream &os, const sim &s)
{
    os << "Total number of particles: " << s.get_nparts() << '\n';
    os << "Collisional timestep     : " << s.get_ct() << '\n';

    return os;
}

} // namespace cascade
