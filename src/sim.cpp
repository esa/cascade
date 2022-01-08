// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>

#include <fmt/ranges.h>

#include <spdlog/stopwatch.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/parallel_sort.h>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/math/sum_sq.hpp>
#include <heyoka/taylor.hpp>

#include <cascade/detail/logging_impl.hpp>
#include <cascade/detail/mortonND_LUT.h>
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

namespace detail
{

namespace
{

// Minimal interval class supporting a couple
// of elementary operations.
struct ival {
    double lower;
    double upper;

    ival() : ival(0) {}
    explicit ival(double val) : ival(val, val) {}
    explicit ival(double l, double u) : lower(l), upper(u) {}
};

// NOTE: see https://en.wikipedia.org/wiki/Interval_arithmetic.
ival operator+(ival a, ival b)
{
    return ival(a.lower + b.lower, a.upper + b.upper);
}

ival operator*(ival a, ival b)
{
    const auto tmp1 = a.lower * b.lower;
    const auto tmp2 = a.lower * b.upper;
    const auto tmp3 = a.upper * b.lower;
    const auto tmp4 = a.upper * b.upper;

    const auto l = std::min(std::min(tmp1, tmp2), std::min(tmp3, tmp4));
    const auto u = std::max(std::max(tmp1, tmp2), std::max(tmp3, tmp4));

    return ival(l, u);
}

// Data structures for storing the lower/upper bounds of a 4D AABB
// in atomic variables.
struct lb_atomic {
    static constexpr auto finf = std::numeric_limits<float>::infinity();

    // NOTE: both default construction
    // and copy construction init all
    // values to +inf.
    lb_atomic() = default;
    lb_atomic(const lb_atomic &) : lb_atomic() {}

    std::atomic<float> x = finf;
    std::atomic<float> y = finf;
    std::atomic<float> z = finf;
    std::atomic<float> r = finf;
};

struct ub_atomic {
    static constexpr auto finf = std::numeric_limits<float>::infinity();

    // NOTE: both default construction
    // and copy construction init all
    // values to -inf.
    ub_atomic() = default;
    ub_atomic(const ub_atomic &) : ub_atomic() {}

    std::atomic<float> x = -finf;
    std::atomic<float> y = -finf;
    std::atomic<float> z = -finf;
    std::atomic<float> r = -finf;
};

// Quantise a value x in [min, max) into one of 2**16
// discrete slots, numbered from 0 to 2**16 - 1.
// NOTE: before invoking this function we must ensure that:
// - all args are finite,
// - max > min,
// - max - min gives a finite result.
// We don't check via assertion that x is in [min, max), because
// conceivably in some corner cases FP computations necessary to
// calculate x outside this function could lead to a value slightly outside
// the allowed range. In such case, we will clamp the result.
std::uint64_t disc_single_coord(float x, float min, float max)
{
    assert(std::isfinite(min));
    assert(std::isfinite(max));
    assert(std::isfinite(x));
    assert(max > min);
    assert(std::isfinite(max - min));

    // Determine the interval size.
    const auto isize = max - min;

    // Translate and rescale x so that min becomes zero
    // and max becomes 1.
    auto rx = (x - min) / isize;

    // Ensure that rx is not negative.
    // NOTE: if rx is NaN, this will set rx to zero.
    rx = rx >= 0.f ? rx : 0.f;

    // Rescale by 2**16.
    rx *= std::uint64_t(1) << 16;

    // Cast back to integer.
    const auto retval = static_cast<std::uint64_t>(rx);

    // Make sure to clamp it before returning, in case
    // somehow FP arithmetic makes it spill outside
    // the bound.
    // TODO min usage?
    return std::min(retval, std::uint64_t((std::uint64_t(1) << 16) - 1u));
}

} // namespace

} // namespace detail

namespace hy = heyoka;

struct sim::sim_data {
    // The adaptive integrators.
    // NOTE: these are never used directly,
    // we just copy them as necessary to setup
    // the integrator caches below.
    hy::taylor_adaptive<double> s_ta;
    hy::taylor_adaptive_batch<double> b_ta;

    // The integrator caches.
    // NOTE: the integrators in the caches are those
    // actually used in numerical propagations.
    oneapi::tbb::concurrent_queue<std::unique_ptr<hy::taylor_adaptive<double>>> s_ta_cache;
    oneapi::tbb::concurrent_queue<std::unique_ptr<hy::taylor_adaptive_batch<double>>> b_ta_cache;

    // The time coordinate.
    hy::detail::dfloat<double> time;

    // Particle substep data to be filled in at each superstep.
    struct step_data {
        // Taylor coefficients for the position vector,
        // each vector contains data for multiple substeps.
        std::vector<double> tc_x, tc_y, tc_z, tc_r;
        // Time coordinates of the end of each substep.
        std::vector<hy::detail::dfloat<double>> tcoords;
    };
    std::vector<step_data> s_data;

    // Bounding box data and Morton codes for each particle.
    // NOTE: each vector contains the data for all chunks.
    std::vector<float> x_lb, y_lb, z_lb, r_lb;
    std::vector<float> x_ub, y_ub, z_ub, r_ub;
    std::vector<std::uint64_t> mcodes;

    // The atomic versions of the global bounding boxes for each chunk.
    std::vector<detail::lb_atomic> global_lb_atomic;
    std::vector<detail::ub_atomic> global_ub_atomic;

    // The non-atomic counterparts of the above.
    std::vector<std::array<float, 4>> global_lb;
    std::vector<std::array<float, 4>> global_ub;

    // The indices vectors for indirect sorting.
    std::vector<size_type> vidx;

    // Temporary buffers to apply the indirect sorting
    // to the AABB data and the Morton codes.
    std::vector<float> aabb_isort;
    std::vector<std::uint64_t> mcodes_isort;
};

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
        oneapi::tbb::parallel_for(tbb::blocked_range(v.begin(), v.end()), [](const auto &range) {
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

            oneapi::tbb::parallel_for(tbb::blocked_range(size_type(0), nparts), [this](const auto &range) {
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

void sim::propagate_for(double t)
{
    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    // Cache a few quantities.
    const auto batch_size = m_data->b_ta.get_batch_size();
    const auto nparts = get_nparts();
    const auto order = m_data->b_ta.get_order();
    // Number of regular batches.
    const auto n_batches = nparts / batch_size;
    // Scalar remainder.
    const auto s_rem = nparts % batch_size;
    // Do we have events in the numerical integration?
    const auto with_events = m_data->s_ta.with_events();
    // The time coordinate at the beginning of
    // the superstep.
    const auto init_time = m_data->time;

    // TODO fix.
    const auto delta_t = 0.46 * 8u;

    // TODO fix.
    // TODO enforce power of 2?
    const auto nchunks = 8u;
    const auto chunk_size = delta_t / nchunks;

    // Ensure the vectors in m_data are set up with the correct size.
    m_data->s_data.resize(boost::numeric_cast<decltype(m_data->s_data.size())>(nparts));
    // TODO overflow checks/numeric cast.
    m_data->x_lb.resize(nparts * nchunks);
    m_data->y_lb.resize(nparts * nchunks);
    m_data->z_lb.resize(nparts * nchunks);
    m_data->r_lb.resize(nparts * nchunks);
    m_data->x_ub.resize(nparts * nchunks);
    m_data->y_ub.resize(nparts * nchunks);
    m_data->z_ub.resize(nparts * nchunks);
    m_data->r_ub.resize(nparts * nchunks);
    m_data->mcodes.resize(nparts * nchunks);
    m_data->vidx.resize(nparts * nchunks);
    m_data->aabb_isort.resize(nparts * nchunks);
    m_data->mcodes_isort.resize(nparts * nchunks);

    // Setup the global atomic lb/ub for each chunk.
    // NOTE: clear() + resize() results in the default
    // construction of the atomic lb/ub objects.
    // TODO numeric cast.
    m_data->global_lb_atomic.clear();
    m_data->global_ub_atomic.clear();
    m_data->global_lb_atomic.resize(nchunks);
    m_data->global_ub_atomic.resize(nchunks);

    // TODO numeric casts.
    m_data->global_lb.resize(nchunks);
    m_data->global_ub.resize(nchunks);

    std::atomic<bool> int_error{false};

    constexpr auto finf = std::numeric_limits<float>::infinity();

    // Batch integration and computation of the AABBs for all particles.
    // TODO scalar remainder.
    auto batch_int_aabb = [&](const auto &range) {
        // Fetch an integrator from the cache, or create it.
        std::unique_ptr<hy::taylor_adaptive_batch<double>> ta_ptr;

        if (!m_data->b_ta_cache.try_pop(ta_ptr)) {
            logger->debug("Creating new batch integrator");

            ta_ptr = std::make_unique<hy::taylor_adaptive_batch<double>>(m_data->b_ta);
        }

        // Cache a few variables.
        auto &ta = *ta_ptr;
        auto *st_data = ta.get_state_data();
        auto &s_data = m_data->s_data;
        const auto &ta_tc = ta.get_tc();

        // The first step is the numerical integration and computation
        // of the bounding boxes for all particles in the range and for
        // all chunks.
        for (auto idx = range.begin(); idx != range.end(); ++idx) {
            // Particle indices corresponding to the current batch.
            const auto pidx_begin = idx * batch_size;
            const auto pidx_end = pidx_begin + batch_size;

            // Clear up the Taylor coefficients and the times
            // of the substeps.
            for (auto i = pidx_begin; i < pidx_end; ++i) {
                s_data[i].tc_x.clear();
                s_data[i].tc_y.clear();
                s_data[i].tc_z.clear();
                s_data[i].tc_r.clear();

                s_data[i].tcoords.clear();
            }

            // Reset cooldowns and set up the times.
            if (with_events) {
                ta.reset_cooldowns();
            }
            ta.set_dtime(init_time.hi, init_time.lo);

            // Copy over the state.
            // NOTE: would need to take care of synching up the
            // runtime parameters too.
            std::copy(m_x.data() + pidx_begin, m_x.data() + pidx_end, st_data);
            std::copy(m_y.data() + pidx_begin, m_y.data() + pidx_end, st_data + batch_size);
            std::copy(m_z.data() + pidx_begin, m_z.data() + pidx_end, st_data + 2u * batch_size);

            std::copy(m_vx.data() + pidx_begin, m_vx.data() + pidx_end, st_data + 3u * batch_size);
            std::copy(m_vy.data() + pidx_begin, m_vy.data() + pidx_end, st_data + 4u * batch_size);
            std::copy(m_vz.data() + pidx_begin, m_vz.data() + pidx_end, st_data + 5u * batch_size);
            std::copy(m_r.data() + pidx_begin, m_r.data() + pidx_end, st_data + 6u * batch_size);

            // Setup the callback.
            auto cb = [&](auto &) {
                for (std::uint32_t i = 0; i < batch_size; ++i) {
                    if (ta.get_last_h()[i] == 0.) {
                        // Ignore this batch index if the last
                        // timestep was zero.
                        continue;
                    }

                    // Record the time coordinate at the end of the step, relative
                    // to the initial time.
                    const auto time_f = hy::detail::dfloat<double>(ta.get_dtime().first[i], ta.get_dtime().second[i]);
                    s_data[pidx_begin + i].tcoords.push_back(time_f - init_time);
                    if (!isfinite(s_data[pidx_begin + i].tcoords.back())) {
                        return false;
                    }

                    // Copy over the Taylor coefficients.
                    // TODO resize + copy, instead of push back?
                    for (std::uint32_t o = 0; o <= order; ++o) {
                        s_data[pidx_begin + i].tc_x.push_back(ta_tc[o * batch_size + i]);
                        s_data[pidx_begin + i].tc_y.push_back(ta_tc[(order + 1u) * batch_size + o * batch_size + i]);
                        s_data[pidx_begin + i].tc_z.push_back(
                            ta_tc[2u * (order + 1u) * batch_size + o * batch_size + i]);
                        s_data[pidx_begin + i].tc_r.push_back(
                            ta_tc[6u * (order + 1u) * batch_size + o * batch_size + i]);
                    }
                }

                return true;
            };
            std::function<bool(hy::taylor_adaptive_batch<double> &)> cbf(std::cref(cb));

            // Integrate.
            ta.propagate_for(delta_t, hy::kw::write_tc = true, hy::kw::callback = cbf);

            // Check for errors.
            if (std::any_of(ta.get_propagate_res().begin(), ta.get_propagate_res().end(),
                            [](const auto &tup) { return std::get<0>(tup) != hy::taylor_outcome::time_limit; })) {
                // TODO distinguish various error codes?
                int_error.store(true, std::memory_order_relaxed);

                break;
            }

            // Compute the bounding boxes for each particle in the batch within each chunk,
            // using the Taylor coefficients which were recorded at each step
            // of the propagate_for().
            for (auto chunk_idx = 0u; chunk_idx < nchunks; ++chunk_idx) {
                // Compute the output pointers.
                const auto offset = nparts * chunk_idx;

                // TODO restrict pointers?
                auto x_lb_ptr = m_data->x_lb.data() + offset;
                auto y_lb_ptr = m_data->y_lb.data() + offset;
                auto z_lb_ptr = m_data->z_lb.data() + offset;
                auto r_lb_ptr = m_data->r_lb.data() + offset;

                auto x_ub_ptr = m_data->x_ub.data() + offset;
                auto y_ub_ptr = m_data->y_ub.data() + offset;
                auto z_ub_ptr = m_data->z_ub.data() + offset;
                auto r_ub_ptr = m_data->r_ub.data() + offset;

                // The time coordinate, relative to init_time, of
                // the chunk's begin/end.
                const auto chunk_begin = hy::detail::dfloat<double>(chunk_size * chunk_idx);
                const auto chunk_end = hy::detail::dfloat<double>(chunk_size * (chunk_idx + 1u));

                for (std::uint32_t i = 0; i < batch_size; ++i) {
                    // Setup the initial values for the bounding box
                    // of the current particle in the current chunk.
                    x_lb_ptr[pidx_begin + i] = finf;
                    y_lb_ptr[pidx_begin + i] = finf;
                    z_lb_ptr[pidx_begin + i] = finf;
                    r_lb_ptr[pidx_begin + i] = finf;

                    x_ub_ptr[pidx_begin + i] = -finf;
                    y_ub_ptr[pidx_begin + i] = -finf;
                    z_ub_ptr[pidx_begin + i] = -finf;
                    r_ub_ptr[pidx_begin + i] = -finf;

                    const auto &tcoords = s_data[pidx_begin + i].tcoords;
                    const auto tcoords_begin = tcoords.begin();
                    const auto tcoords_end = tcoords.end();

                    // We need to locate the substep range that fully includes
                    // the current chunk.
                    // First we locate the first substep whose end is strictly
                    // *greater* than the lower bound of the chunk.
                    auto ss_it_begin = std::upper_bound(tcoords_begin, tcoords_end, chunk_begin);
                    // Then, we locate the first substep whose end is *greater than or
                    // equal to* the end of the chunk.
                    auto ss_it_end = std::lower_bound(ss_it_begin, tcoords_end, chunk_end);
                    // Bump it up by one to define a half-open range.
                    // NOTE: don't bump it if it is already at the end.
                    // This could happen at the last chunk due to FP rounding.
                    ss_it_end += (ss_it_end != tcoords_end);

                    // Iterate over all substeps and update the bounding box
                    // for the current particle.
                    for (auto it = ss_it_begin; it != ss_it_end; ++it) {
                        // it points to the end of a substep which overlaps
                        // with the current chunk. The size of the polynomial evaluation
                        // interval is the size of the intersection between the substep and
                        // the chunk.

                        // Determine the initial time coordinate of the substep, relative
                        // to init_time. If it is tcoords_begin, ss_start will be zero, otherwise
                        // ss_start is given by the iterator preceding it.
                        const auto ss_start = (it == tcoords_begin) ? hy::detail::dfloat<double>(0) : *(it - 1);

                        // Determine lower/upper bounds of the evaluation interval,
                        // relative to init_time.
                        // TODO std::min/max here?
                        const auto ev_lb = std::max(chunk_begin, ss_start);
                        const auto ev_ub = std::min(chunk_end, *it);

                        // Create the actual evaluation interval, referring
                        // it to the beginning of the substep.
                        const auto h_int_lb = static_cast<double>(ev_lb - ss_start);
                        const auto h_int_ub = static_cast<double>(ev_ub - ss_start);

                        // Determine the index of the substep within the chunk.
                        // TODO: overflow check -> tcoords' size must fit in the
                        // iterator difference type.
                        const auto ss_idx = boost::numeric_cast<std::vector<double>::size_type>(it - tcoords_begin);

                        // Compute the pointers to the TCs for the current particle
                        // and substep.
                        const auto tc_ptr_x = s_data[pidx_begin + i].tc_x.data() + ss_idx * (order + 1u);
                        const auto tc_ptr_y = s_data[pidx_begin + i].tc_y.data() + ss_idx * (order + 1u);
                        const auto tc_ptr_z = s_data[pidx_begin + i].tc_z.data() + ss_idx * (order + 1u);
                        const auto tc_ptr_r = s_data[pidx_begin + i].tc_r.data() + ss_idx * (order + 1u);

                        // Run the polynomial evaluations using interval arithmetic.
                        // TODO jit for performance?
                        auto horner_eval = [order, h_int = detail::ival(h_int_lb, h_int_ub)](const double *ptr) {
                            auto acc = detail::ival(ptr[order]);
                            for (auto o = 1u; o <= order; ++o) {
                                acc = detail::ival(ptr[order - o]) + acc * h_int;
                            }

                            return acc;
                        };

                        // TODO: particle size/cross section should be accounted for here, by enlarging
                        // x/y/z/r_int as needed.
                        const auto x_int = horner_eval(tc_ptr_x);
                        const auto y_int = horner_eval(tc_ptr_y);
                        const auto z_int = horner_eval(tc_ptr_z);
                        const auto r_int = horner_eval(tc_ptr_r);

                        // Update the bounding box for the current particle.
                        // TODO: min/max usage?
                        // TODO: inf checking? Here or when updating the global AABB?
                        x_lb_ptr[pidx_begin + i] = std::min(x_lb_ptr[pidx_begin + i],
                                                            std::nextafter(static_cast<float>(x_int.lower), -finf));
                        y_lb_ptr[pidx_begin + i] = std::min(y_lb_ptr[pidx_begin + i],
                                                            std::nextafter(static_cast<float>(y_int.lower), -finf));
                        z_lb_ptr[pidx_begin + i] = std::min(z_lb_ptr[pidx_begin + i],
                                                            std::nextafter(static_cast<float>(z_int.lower), -finf));
                        r_lb_ptr[pidx_begin + i] = std::min(r_lb_ptr[pidx_begin + i],
                                                            std::nextafter(static_cast<float>(r_int.lower), -finf));

                        x_ub_ptr[pidx_begin + i]
                            = std::max(x_ub_ptr[pidx_begin + i], std::nextafter(static_cast<float>(x_int.upper), finf));
                        y_ub_ptr[pidx_begin + i]
                            = std::max(y_ub_ptr[pidx_begin + i], std::nextafter(static_cast<float>(y_int.upper), finf));
                        z_ub_ptr[pidx_begin + i]
                            = std::max(z_ub_ptr[pidx_begin + i], std::nextafter(static_cast<float>(z_int.upper), finf));
                        r_ub_ptr[pidx_begin + i]
                            = std::max(r_ub_ptr[pidx_begin + i], std::nextafter(static_cast<float>(r_int.upper), finf));
                    }
                }
            }
        }

        // We can now update the global AABB for each chunk.
        // NOTE: the reason we do it here, instead of the previous
        // loop, is to minimise contention on the updates to the
        // global AABBs.
        for (auto chunk_idx = 0u; chunk_idx < nchunks; ++chunk_idx) {
            // The global bounding box for the current chunk.
            auto &glb = m_data->global_lb_atomic[chunk_idx];
            auto &gub = m_data->global_ub_atomic[chunk_idx];

            // Chunk-specific global bounding box for the current particle range.
            std::array<float, 4> local_lb = {finf, finf, finf, finf};
            std::array<float, 4> local_ub = {-finf, -finf, -finf, -finf};

            const auto offset = nparts * chunk_idx;

            // TODO restrict pointers?
            auto x_lb_ptr = m_data->x_lb.data() + offset;
            auto y_lb_ptr = m_data->y_lb.data() + offset;
            auto z_lb_ptr = m_data->z_lb.data() + offset;
            auto r_lb_ptr = m_data->r_lb.data() + offset;

            auto x_ub_ptr = m_data->x_ub.data() + offset;
            auto y_ub_ptr = m_data->y_ub.data() + offset;
            auto z_ub_ptr = m_data->z_ub.data() + offset;
            auto r_ub_ptr = m_data->r_ub.data() + offset;

            for (auto idx = range.begin(); idx != range.end(); ++idx) {
                // Particle indices corresponding to the current batch.
                const auto pidx_begin = idx * batch_size;
                const auto pidx_end = pidx_begin + batch_size;

                for (std::uint32_t i = 0; i < batch_size; ++i) {
                    // Update the local AABB with the bounding box for the current particle.
                    // TODO: min/max usage?
                    local_lb[0] = std::min(local_lb[0], x_lb_ptr[pidx_begin + i]);
                    local_lb[1] = std::min(local_lb[1], y_lb_ptr[pidx_begin + i]);
                    local_lb[2] = std::min(local_lb[2], z_lb_ptr[pidx_begin + i]);
                    local_lb[3] = std::min(local_lb[3], r_lb_ptr[pidx_begin + i]);

                    local_ub[0] = std::max(local_ub[0], x_lb_ptr[pidx_begin + i]);
                    local_ub[1] = std::max(local_ub[1], y_lb_ptr[pidx_begin + i]);
                    local_ub[2] = std::max(local_ub[2], z_lb_ptr[pidx_begin + i]);
                    local_ub[3] = std::max(local_ub[3], r_lb_ptr[pidx_begin + i]);
                }
            }

            // Update the global AABB for the current chunk.
            auto lb_updater = [](auto &out, float val) {
                // Load the current value from the atomic.
                auto orig_val = out.load(std::memory_order_relaxed);
                float new_val;

                do {
                    // Compute the new value.
                    // TODO min usage?
                    new_val = std::min(val, orig_val);
                } while (!out.compare_exchange_weak(orig_val, new_val, std::memory_order_relaxed,
                                                    std::memory_order_relaxed));
            };

            auto ub_updater = [](auto &out, float val) {
                // Load the current value from the atomic.
                auto orig_val = out.load(std::memory_order_relaxed);
                float new_val;

                do {

                    // Compute the new value.
                    // TODO max usage?
                    new_val = std::max(val, orig_val);
                } while (!out.compare_exchange_weak(orig_val, new_val, std::memory_order_relaxed,
                                                    std::memory_order_relaxed));
            };

            lb_updater(glb.x, local_lb[0]);
            lb_updater(glb.y, local_lb[1]);
            lb_updater(glb.z, local_lb[2]);
            lb_updater(glb.r, local_lb[3]);

            ub_updater(gub.x, local_ub[0]);
            ub_updater(gub.y, local_ub[1]);
            ub_updater(gub.z, local_ub[2]);
            ub_updater(gub.r, local_ub[3]);
        }

        // Put the integrator (back) into the cache.
        m_data->b_ta_cache.push(std::move(ta_ptr));
    };

    // Do the integration concurrently with the initialisation of
    // the vector of indices for indirect sorting.
    oneapi::tbb::parallel_invoke(
        [&]() {
            auto &vidx = m_data->vidx;

            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(0u, nchunks), [&](const auto &range) {
                for (auto chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
                    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(0, nparts),
                                              [vidx_ptr = vidx.data() + nparts * chunk_idx](const auto &r2) {
                                                  for (auto i = r2.begin(); i != r2.end(); ++i) {
                                                      vidx_ptr[i] = i;
                                                  }
                                              });
                }
            });
        },
        [&]() { oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(0, n_batches), batch_int_aabb); });

    logger->trace("Propagation + AABB computation time: {}s", sw);

    if (int_error.load(std::memory_order_relaxed)) {
        // TODO
        throw;
    }

    // Computation of the Morton codes and sorting.
    sw.reset();

    constexpr auto morton_enc = mortonnd::MortonNDLutEncoder<4, 16, 8>();

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(0u, nchunks), [&](const auto &range) {
        for (auto chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
            auto &glb = m_data->global_lb[chunk_idx];
            auto &gub = m_data->global_ub[chunk_idx];

            // Load the atomic global AABB data into the non-atomic counterpart.
            const auto &glb_at = m_data->global_lb_atomic[chunk_idx];
            const auto &gub_at = m_data->global_ub_atomic[chunk_idx];

            glb[0] = glb_at.x.load(std::memory_order_relaxed);
            glb[1] = glb_at.y.load(std::memory_order_relaxed);
            glb[2] = glb_at.z.load(std::memory_order_relaxed);
            glb[3] = glb_at.r.load(std::memory_order_relaxed);

            // TODO:
            // - bump up UB in order to ensure it's always > lb, as requested by the spatial
            //   discretisation function;
            // - check finiteness and the disc_single_coord requirements, before adjusting;
            gub[0] = gub_at.x.load(std::memory_order_relaxed);
            gub[1] = gub_at.y.load(std::memory_order_relaxed);
            gub[2] = gub_at.z.load(std::memory_order_relaxed);
            gub[3] = gub_at.r.load(std::memory_order_relaxed);
            // TODO: run a second check here to verify the new upper bound is not +inf.

            // Computation of the Morton codes.
            const auto offset = nparts * chunk_idx;

            // TODO restrict pointers?
            auto x_lb_ptr = m_data->x_lb.data() + offset;
            auto y_lb_ptr = m_data->y_lb.data() + offset;
            auto z_lb_ptr = m_data->z_lb.data() + offset;
            auto r_lb_ptr = m_data->r_lb.data() + offset;

            auto x_ub_ptr = m_data->x_ub.data() + offset;
            auto y_ub_ptr = m_data->y_ub.data() + offset;
            auto z_ub_ptr = m_data->z_ub.data() + offset;
            auto r_ub_ptr = m_data->r_ub.data() + offset;

            auto mcodes_ptr = m_data->mcodes.data() + offset;

            auto vidx_ptr = m_data->vidx.data() + offset;

            auto aabb_isort_ptr = m_data->aabb_isort.data() + offset;
            auto mcodes_isort_ptr = m_data->mcodes_isort.data() + offset;

            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(0, nparts), [&](const auto &r2) {
                for (auto pidx = r2.begin(); pidx != r2.end(); ++pidx) {
                    // Compute the centre of the AABB.
                    const auto x_ctr = x_lb_ptr[pidx] / 2 + x_ub_ptr[pidx] / 2;
                    const auto y_ctr = y_lb_ptr[pidx] / 2 + y_ub_ptr[pidx] / 2;
                    const auto z_ctr = z_lb_ptr[pidx] / 2 + z_ub_ptr[pidx] / 2;
                    const auto r_ctr = r_lb_ptr[pidx] / 2 + r_ub_ptr[pidx] / 2;

                    const auto n0 = detail::disc_single_coord(x_ctr, glb[0], gub[0]);
                    const auto n1 = detail::disc_single_coord(y_ctr, glb[1], gub[1]);
                    const auto n2 = detail::disc_single_coord(z_ctr, glb[2], gub[2]);
                    const auto n3 = detail::disc_single_coord(r_ctr, glb[3], gub[3]);

                    mcodes_ptr[pidx] = morton_enc.Encode(n0, n1, n2, n3);
                }
            });

            // Indirect sorting of the indices for the current chunk
            // according to the Morton codes.
            oneapi::tbb::parallel_sort(vidx_ptr, vidx_ptr + nparts, [mcodes_ptr](auto idx1, auto idx2) {
                return mcodes_ptr[idx1] < mcodes_ptr[idx2];
            });

            // Helper to apply the indirect sorting defined in vidx to the data in src.
            // tmp is used as temporary buffer.
            auto isort_apply = [vidx_ptr, nparts](auto *tmp, auto *src) {
                oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(0, nparts), [&](const auto &range) {
                    for (auto i = range.begin(); i != range.end(); ++i) {
                        tmp[i] = src[vidx_ptr[i]];
                    }
                });

                oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(0, nparts), [&](const auto &range) {
                    for (auto i = range.begin(); i != range.end(); ++i) {
                        src[i] = tmp[i];
                    }
                });
            };

            // oneapi::tbb::parallel_invoke(
            //     [&]() {
            isort_apply(aabb_isort_ptr, x_lb_ptr);
            isort_apply(aabb_isort_ptr, y_lb_ptr);
            isort_apply(aabb_isort_ptr, z_lb_ptr);
            isort_apply(aabb_isort_ptr, r_lb_ptr);

            isort_apply(aabb_isort_ptr, x_ub_ptr);
            isort_apply(aabb_isort_ptr, y_ub_ptr);
            isort_apply(aabb_isort_ptr, z_ub_ptr);
            isort_apply(aabb_isort_ptr, r_ub_ptr);
            //     },
            //     [&]() {
            isort_apply(mcodes_isort_ptr, mcodes_ptr);
            //});
        }
    });

    logger->trace("Morton encoding and sorting time: {}s", sw);
}

} // namespace cascade
