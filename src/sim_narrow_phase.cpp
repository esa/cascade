// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>

#include <spdlog/stopwatch.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <heyoka/detail/dfloat.hpp>

#include <cascade/detail/logging_impl.hpp>
#include <cascade/detail/sim_data.hpp>
#include <cascade/sim.hpp>

namespace cascade
{

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

    const auto nchunks = static_cast<unsigned>(m_data->global_lb.size());

    const auto &s_data = m_data->s_data;

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(0u, nchunks), [&](const auto &range) {
        for (auto chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
            // Fetch a reference to the chunk-specific broad
            // phase collision vector.
            const auto &bpc = m_data->bp_coll[chunk_idx];

            // The time coordinate, relative to init_time, of
            // the chunk's begin/end.
            const auto chunk_begin = hy::detail::dfloat<double>(chunk_size * chunk_idx);
            const auto chunk_end = hy::detail::dfloat<double>(chunk_size * (chunk_idx + 1u));

            // Iterate over all collisions.
            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(bpc.begin(), bpc.end()), [&](const auto &rn) {
                for (const auto &pc : rn) {
                    const auto [pi, pj] = pc;

                    assert(pi != pj);

                    // Fetch a reference to the substep data
                    // for the two particles.
                    const auto &sd_i = s_data[pi];
                    const auto &sd_j = s_data[pj];

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

                        // Refer lb/ub to the beginning of the two substeps,
                        // and cast to double.
                        const auto h_int_lb_i = static_cast<double>(lb - ss_start_i);
                        const auto h_int_ub_i = static_cast<double>(ub - ss_start_i);
                        const auto h_int_lb_j = static_cast<double>(lb - ss_start_j);
                        const auto h_int_ub_j = static_cast<double>(ub - ss_start_j);

                        // Run checks on the results before moving forward, since we used
                        // FP arith and there could also be corner cases in which we end up
                        // with empty and/or invalid intervals.
                        auto lb_ub_check = [](double l, double u) {
                            return std::isfinite(l) && std::isfinite(u) && l >= 0 && u > l;
                        };

                        if (!lb_ub_check(h_int_lb_i, h_int_ub_i) || !lb_ub_check(h_int_lb_j, h_int_ub_j)) {
                            // Bail out in case of errors.
                            logger->warn("During the narrow-phase collision detection of particles {} and {}, "
                                         "an invalid time interval for polynomial root finding was generated - the "
                                         "collision will be skipped",
                                         pi, pj);

                            break;
                        }

                        // TODO poly translation, root finding.

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
            });
        }
    });

    logger->trace("Narrow phase collision detection time: {}s", sw);
}

} // namespace cascade