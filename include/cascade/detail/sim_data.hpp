// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CASCADE_DETAIL_SIM_DATA_HPP
#define CASCADE_DETAIL_SIM_DATA_HPP

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include <oneapi/tbb/concurrent_queue.h>

#include <heyoka/detail/dfloat.hpp>
#include <heyoka/taylor.hpp>

#include <cascade/sim.hpp>

namespace cascade
{

struct sim::sim_data {
    static constexpr auto finf = std::numeric_limits<float>::infinity();

    // Data structures for storing the lower/upper bounds of a 4D AABB
    // in atomic variables.
    struct lb_atomic {
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

    // The adaptive integrators.
    // NOTE: these are never used directly,
    // we just copy them as necessary to setup
    // the integrator caches below.
    heyoka::taylor_adaptive<double> s_ta;
    heyoka::taylor_adaptive_batch<double> b_ta;

    // The integrator caches.
    // NOTE: the integrators in the caches are those
    // actually used in numerical propagations.
    oneapi::tbb::concurrent_queue<std::unique_ptr<heyoka::taylor_adaptive<double>>> s_ta_cache;
    oneapi::tbb::concurrent_queue<std::unique_ptr<heyoka::taylor_adaptive_batch<double>>> b_ta_cache;

    // The time coordinate.
    heyoka::detail::dfloat<double> time;

    // Particle substep data to be filled in at each superstep.
    struct step_data {
        // Taylor coefficients for the position vector,
        // each vector contains data for multiple substeps.
        std::vector<double> tc_x, tc_y, tc_z, tc_r;
        // Time coordinates of the end of each substep.
        std::vector<heyoka::detail::dfloat<double>> tcoords;
    };
    std::vector<step_data> s_data;

    // Bounding box data and Morton codes for each particle.
    // NOTE: each vector contains the data for all chunks.
    std::vector<float> x_lb, y_lb, z_lb, r_lb;
    std::vector<float> x_ub, y_ub, z_ub, r_ub;
    std::vector<std::uint64_t> mcodes;

    // The atomic versions of the global bounding boxes for each chunk.
    std::vector<lb_atomic> global_lb_atomic;
    std::vector<ub_atomic> global_ub_atomic;

    // The non-atomic counterparts of the above.
    std::vector<std::array<float, 4>> global_lb;
    std::vector<std::array<float, 4>> global_ub;

    // The indices vectors for indirect sorting.
    std::vector<size_type> vidx;

    // Versions of AABBs and Morton codes sorted
    // according to vidx.
    std::vector<float> srt_x_lb, srt_y_lb, srt_z_lb, srt_r_lb;
    std::vector<float> srt_x_ub, srt_y_ub, srt_z_ub, srt_r_ub;
    std::vector<std::uint64_t> srt_mcodes;
};

} // namespace cascade

#endif
