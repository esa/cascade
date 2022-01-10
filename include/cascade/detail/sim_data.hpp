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
#include <cstdint>
#include <memory>
#include <vector>

#include <oneapi/tbb/concurrent_queue.h>

#include <heyoka/detail/dfloat.hpp>
#include <heyoka/taylor.hpp>

#include <cascade/sim.hpp>

namespace cascade
{

struct sim::sim_data {
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

    // The global bounding boxes, one for each chunk.
    std::vector<std::array<float, 4>> global_lb;
    std::vector<std::array<float, 4>> global_ub;

    // The indices vectors for indirect sorting.
    std::vector<size_type> vidx;

    // Versions of AABBs and Morton codes sorted
    // according to vidx.
    std::vector<float> srt_x_lb, srt_y_lb, srt_z_lb, srt_r_lb;
    std::vector<float> srt_x_ub, srt_y_ub, srt_z_ub, srt_r_ub;
    std::vector<std::uint64_t> srt_mcodes;

    // The BVH node struct.
    struct bvh_node {
        size_type begin, end;
        std::int64_t parent, left, right;
        std::array<float, 4> lb, ub;
        // NOTE: split_idx is used only during tree construction.
        int split_idx;
    };

    // The BVH trees, one for each chunk.
    using bvh_tree_t = std::vector<bvh_node>;
    std::vector<bvh_tree_t> bvh_trees;
    // Temporary buffer used in the construction of the BVH trees.
    template <typename T>
    struct uninit {
        T val;
    };
    std::vector<std::vector<uninit<bvh_tree_t::size_type>>> nc_buffer;
    std::vector<std::vector<uninit<bvh_tree_t::size_type>>> ps_buffer;
    std::vector<std::vector<uninit<size_type>>> nplc_buffer;
};

} // namespace cascade

#endif
