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
#include <cstddef>
#include <utility>

#include <boost/numeric/conversion/cast.hpp>

#include <spdlog/stopwatch.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <heyoka/detail/dfloat.hpp>

#include <cascade/detail/logging_impl.hpp>
#include <cascade/detail/sim_data.hpp>
#include <cascade/sim.hpp>

namespace cascade
{

namespace detail
{

namespace
{

// Evaluate polynomial.
// Requires random-access iterator.
template <typename InputIt, typename T>
auto poly_eval(InputIt a, T x, std::uint32_t n)
{
    auto ret = a[n];

    for (std::uint32_t i = 1; i <= n; ++i) {
        ret = a[n - i] + ret * x;
    }

    return ret;
}

} // namespace

} // namespace detail

// Narrow phase collision detection: the trajectories
// of the particle pairs identified during broad
// phase collision detection are tested for intersection
// using polynomial root finding.
void sim::narrow_phase(double chunk_size)
{
    namespace hy = heyoka;

    assert(std::isfinite(chunk_size));
    assert(chunk_size > 0);

    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    // Cache a few bits.
    const auto nchunks = static_cast<unsigned>(m_data->global_lb.size());
    const auto order = m_data->s_ta.get_order();
    const auto &s_data = m_data->s_data;
    const auto pta = m_data->pta;
    const auto pssdiff3 = m_data->pssdiff3;
    const auto fex_check = m_data->fex_check;

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(0u, nchunks), [&](const auto &range) {
        for (auto chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
            // Fetch a reference to the chunk-specific broad
            // phase collision vector.
            const auto &bpc = m_data->bp_coll[chunk_idx];

            // Fetch references to the chunk-specific caches.
            auto &pcache = m_data->poly_caches[chunk_idx];

            // The time coordinate, relative to init_time, of
            // the chunk's begin/end.
            const auto chunk_begin = hy::detail::dfloat<double>(chunk_size * chunk_idx);
            const auto chunk_end = hy::detail::dfloat<double>(chunk_size * (chunk_idx + 1u));

            // Counter for the number of failed fast exclusion checks.
            std::atomic<std::size_t> n_ffex(0);

            // Iterate over all collisions.
            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(bpc.begin(), bpc.end()), [&](const auto &rn) {
                // Local version of n_ffex.
                std::size_t local_n_ffex = 0;

                // Try to fetch 6 polynomials from the cache.
                // TODO unique_ptr perhaps performs better here?
                std::array<std::vector<double>, 7> poly_temp;
                auto &[xi_temp, yi_temp, zi_temp, xj_temp, yj_temp, zj_temp, ss_diff] = poly_temp;

                if (!pcache.try_pop(poly_temp)) {
                    logger->debug("Creating new local polynomials for narrow phase collision detection");

                    xi_temp.resize(boost::numeric_cast<decltype(xi_temp.size())>(order + 1u));
                    yi_temp.resize(boost::numeric_cast<decltype(yi_temp.size())>(order + 1u));
                    zi_temp.resize(boost::numeric_cast<decltype(zi_temp.size())>(order + 1u));

                    xj_temp.resize(boost::numeric_cast<decltype(xj_temp.size())>(order + 1u));
                    yj_temp.resize(boost::numeric_cast<decltype(yj_temp.size())>(order + 1u));
                    zj_temp.resize(boost::numeric_cast<decltype(zj_temp.size())>(order + 1u));

                    ss_diff.resize(boost::numeric_cast<decltype(ss_diff.size())>(order + 1u));
                }

                for (const auto &pc : rn) {
                    const auto [pi, pj] = pc;

                    assert(pi != pj);

                    // Fetch a reference to the substep data
                    // for the two particles.
                    const auto &sd_i = s_data[pi];
                    const auto &sd_j = s_data[pj];

                    // Load the particle radiuses.
                    const auto p_rad_i = m_sizes[pi];
                    const auto p_rad_j = m_sizes[pj];

                    // Cache the range of end times of the substeps.
                    const auto tcoords_begin_i = sd_i.tcoords.begin();
                    const auto tcoords_end_i = sd_i.tcoords.end();

                    const auto tcoords_begin_j = sd_j.tcoords.begin();
                    const auto tcoords_end_j = sd_j.tcoords.end();

                    // Determine, for both particles, the range of substeps
                    // that fully includes the current chunk.
                    // NOTE: same code as in sim_propagate.cpp.
                    const auto ss_it_begin_i = std::upper_bound(tcoords_begin_i, tcoords_end_i, chunk_begin);
                    auto ss_it_end_i = std::lower_bound(ss_it_begin_i, tcoords_end_i, chunk_end);
                    ss_it_end_i += (ss_it_end_i != tcoords_end_i);

                    const auto ss_it_begin_j = std::upper_bound(tcoords_begin_j, tcoords_end_j, chunk_begin);
                    auto ss_it_end_j = std::lower_bound(ss_it_begin_j, tcoords_end_j, chunk_end);
                    ss_it_end_j += (ss_it_end_j != tcoords_end_j);

                    // Iterate until we get to the end of at least one range.
                    for (auto it_i = ss_it_begin_i, it_j = ss_it_begin_j; it_i != ss_it_end_i && it_j != ss_it_end_j;) {
                        // Neither at the end.

                        // Initial time coordinates of the substeps of i and j,
                        // relative to init_time.
                        const auto ss_start_i = (it_i == tcoords_begin_i) ? hy::detail::dfloat<double>(0) : *(it_i - 1);
                        const auto ss_start_j = (it_j == tcoords_begin_j) ? hy::detail::dfloat<double>(0) : *(it_j - 1);

                        // Determine the intersections of the two substeps
                        // with the current chunk.
                        // TODO std::min/max here?
                        const auto lb_i = std::max(chunk_begin, ss_start_i);
                        const auto ub_i = std::min(chunk_end, *it_i);
                        const auto lb_j = std::max(chunk_begin, ss_start_j);
                        const auto ub_j = std::min(chunk_end, *it_j);

                        // Determine the intersection between the two intervals
                        // we just computed. This will be the time range
                        // within which we need to do polynomial root finding.
                        // NOTE: at this stage lb/ub are still time coordinates wrt
                        // init_time.
                        // TODO std::min/max here?
                        const auto lb = std::max(lb_i, lb_j);
                        const auto ub = std::min(ub_i, ub_j);

                        // The Taylor polynomials for the two particles are time polynomials
                        // in which time is counted from the beginning of the substep. In order to
                        // create the polynomial representing the distance square, we need first to
                        // translate the polynomials of both particles so that they refer to a
                        // common time coordinate, the time elapsed from lb.

                        // Compute the translation amount for the two particles.
                        const auto delta_i = static_cast<double>(lb - ss_start_i);
                        const auto delta_j = static_cast<double>(lb - ss_start_j);

                        // Compute the time interval within which we will be performing root finding.
                        const auto rf_int = static_cast<double>(ub - lb);

                        // Do some checking before moving on.
                        if (!std::isfinite(delta_i) || !std::isfinite(delta_j) || !std::isfinite(rf_int) || delta_i < 0
                            || delta_j < 0 || rf_int < 0) {
                            // Bail out in case of errors.
                            logger->warn("During the narrow-phase collision detection of particles {} and {}, "
                                         "an invalid time interval for polynomial root finding was generated - the "
                                         "collision will be skipped",
                                         pi, pj);

                            break;
                        }

                        // Fetch pointers to the original Taylor polynomials for the two particles.
                        // NOTE: static_cast because overflow checking and numeric cast are already
                        // done in sim_propagate_for.
                        const auto ss_idx_i = static_cast<decltype(s_data[pi].tc_x.size())>(it_i - tcoords_begin_i);
                        const auto ss_idx_j = static_cast<decltype(s_data[pj].tc_x.size())>(it_j - tcoords_begin_j);

                        const auto *poly_xi = s_data[pi].tc_x.data() + ss_idx_i * (order + 1u);
                        const auto *poly_yi = s_data[pi].tc_y.data() + ss_idx_i * (order + 1u);
                        const auto *poly_zi = s_data[pi].tc_z.data() + ss_idx_i * (order + 1u);

                        const auto *poly_xj = s_data[pj].tc_x.data() + ss_idx_j * (order + 1u);
                        const auto *poly_yj = s_data[pj].tc_y.data() + ss_idx_j * (order + 1u);
                        const auto *poly_zj = s_data[pj].tc_z.data() + ss_idx_j * (order + 1u);

                        // Perform the translations, if needed.
                        // NOTE: perhaps we can write a dedicated function
                        // that does the translation for all 3 coordinates
                        // at once, for better performance?
                        if (delta_i != 0) {
                            poly_xi = pta(xi_temp.data(), poly_xi, delta_i);
                            poly_yi = pta(yi_temp.data(), poly_yi, delta_i);
                            poly_zi = pta(zi_temp.data(), poly_zi, delta_i);
                        }

                        if (delta_j != 0) {
                            poly_xj = pta(xj_temp.data(), poly_xj, delta_j);
                            poly_yj = pta(yj_temp.data(), poly_yj, delta_j);
                            poly_zj = pta(zj_temp.data(), poly_zj, delta_j);
                        }

                        // We can now construct the polynomial for the
                        // square of the distance.
                        pssdiff3(ss_diff.data(), poly_xi, poly_yi, poly_zi, poly_xj, poly_yj, poly_zj);

                        // Modify the constant term of the polynomial to account for
                        // particle sizes.
                        ss_diff[0] -= (p_rad_i + p_rad_j) * (p_rad_i + p_rad_j);

                        // Run the fast exclusion check.
                        std::uint32_t fex_check_res, back_flag = 0;
                        fex_check(ss_diff.data(), &rf_int, &back_flag, &fex_check_res);
                        if (!fex_check_res) {
                            ++local_n_ffex;
                        }

                        if (*it_i < *it_j) {
                            // The substep for particle i ends
                            // before the substep for particle j.
                            ++it_i;
                        } else if (*it_j < *it_i) {
                            // The substep for particle j ends
                            // before the substep for particle i.
                            ++it_j;
                        } else {
                            // Both substeps end at the same time.
                            // This happens at the last substeps of a chunk
                            // or in the very unlikely case in which both
                            // steps end exactly at the same time.
                            ++it_i;
                            ++it_j;
                        }
                    }
                }

                // Put the polynomials back into the cache.
                pcache.push(std::move(poly_temp));

                // Update n_ffex.
                n_ffex.fetch_add(local_n_ffex, std::memory_order::relaxed);
            });

            logger->debug("Number of failed fast exclusion checks for chunk {}: {} vs {} broad phase collisions",
                          chunk_idx, n_ffex.load(std::memory_order::relaxed), bpc.size());
        }
    });

    logger->trace("Narrow phase collision detection time: {}s", sw);
}

} // namespace cascade