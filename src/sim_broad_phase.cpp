// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <atomic>
#include <chrono>
#include <utility>
#include <vector>

#include <spdlog/stopwatch.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <cascade/detail/logging_impl.hpp>
#include <cascade/detail/sim_data.hpp>
#include <cascade/sim.hpp>

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)

#define CASCADE_RESTRICT __restrict

#else

#define CASCADE_RESTRICT

#endif

namespace cascade
{

void sim::broad_phase()
{
    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    // Fetch the number of particles and chunks from m_data.
    const auto nparts = get_nparts();
    const auto nchunks = static_cast<unsigned>(m_data->global_lb.size());

    // Global counter for the total number of AABBs collisions
    // across all chunks.
    std::atomic<decltype(m_data->bp_coll[0].size())> tot_n_bp(0);

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(0u, nchunks), [&](const auto &range) {
        for (auto chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
            const auto offset = nparts * chunk_idx;

            const auto *CASCADE_RESTRICT x_lb_ptr = m_data->srt_x_lb.data() + offset;
            const auto *CASCADE_RESTRICT y_lb_ptr = m_data->srt_y_lb.data() + offset;
            const auto *CASCADE_RESTRICT z_lb_ptr = m_data->srt_z_lb.data() + offset;
            const auto *CASCADE_RESTRICT r_lb_ptr = m_data->srt_r_lb.data() + offset;

            const auto *CASCADE_RESTRICT x_ub_ptr = m_data->srt_x_ub.data() + offset;
            const auto *CASCADE_RESTRICT y_ub_ptr = m_data->srt_y_ub.data() + offset;
            const auto *CASCADE_RESTRICT z_ub_ptr = m_data->srt_z_ub.data() + offset;
            const auto *CASCADE_RESTRICT r_ub_ptr = m_data->srt_r_ub.data() + offset;

            // Fetch a reference to the tree.
            const auto &tree = m_data->bvh_trees[chunk_idx];

            // Fetch a reference to the AABB collision vector for the
            // current chunk and clear it out.
            auto &coll_vec = m_data->bp_coll[chunk_idx];
            coll_vec.clear();

            // Fetch a reference to the bp cache for the current chunk.
            auto &bp_cache = m_data->bp_caches[chunk_idx];

            // Fetch a reference to the stack cache for the current chunk.
            auto &st_cache = m_data->stack_caches[chunk_idx];

            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(0, nparts), [&](const auto &r2) {
                // Fetch a local collision vector from the cache.
                std::vector<std::pair<size_type, size_type>> local_bp;
                if (bp_cache.try_pop(local_bp)) {
                    // Clear it.
                    local_bp.clear();
                } else {
                    logger->debug("Creating new local BP");
                }

                // Fetch a local stack from the cache.
                // NOTE: this will be cleared at the beginning
                // of the traversal for each particle.
                std::vector<std::int32_t> stack;
                if (!st_cache.try_pop(stack)) {
                    logger->debug("Creating new local stack");
                }

                for (auto pidx = r2.begin(); pidx != r2.end(); ++pidx) {
                    // Reset the stack, and add the root node to it.
                    stack.clear();
                    stack.push_back(0);

                    // Cache the AABB of the current particle.
                    const auto x_lb = x_lb_ptr[pidx];
                    const auto y_lb = y_lb_ptr[pidx];
                    const auto z_lb = z_lb_ptr[pidx];
                    const auto r_lb = r_lb_ptr[pidx];

                    const auto x_ub = x_ub_ptr[pidx];
                    const auto y_ub = y_ub_ptr[pidx];
                    const auto z_ub = z_ub_ptr[pidx];
                    const auto r_ub = r_ub_ptr[pidx];

                    while (!stack.empty()) {
                        // Pop a node.
                        const auto cur_node_idx = stack.back();
                        stack.pop_back();

                        // Fetch the AABB of the node.
                        const auto &cur_node = tree[static_cast<std::uint32_t>(cur_node_idx)];
                        const auto &n_lb = cur_node.lb;
                        const auto &n_ub = cur_node.ub;

                        // Check for overlap with the AABB of the current particle.
                        const bool overlap
                            = (x_ub >= n_lb[0] && x_lb <= n_ub[0]) && (y_ub >= n_lb[1] && y_lb <= n_ub[1])
                              && (z_ub >= n_lb[2] && z_lb <= n_ub[2]) && (r_ub >= n_lb[3] && r_lb <= n_ub[3]);

                        if (overlap) {
                            if (cur_node.left == -1) {
                                // Leaf node: mark pidx as colliding with
                                // all particles in the node, except for itself.
                                for (auto i = cur_node.begin; i != cur_node.end; ++i) {
                                    if (pidx != i) {
                                        local_bp.emplace_back(pidx, i);
                                    }
                                }
                            } else {
                                // Internal node: add both children to the
                                // stack and iterate.
                                stack.push_back(cur_node.left);
                                stack.push_back(cur_node.right);
                            }
                        }
                    }
                }

                // Merge the local bp into the global one.
                coll_vec.grow_by(local_bp.begin(), local_bp.end());

                // Put local_bp and the stack (back) into the caches.
                bp_cache.push(std::move(local_bp));
                st_cache.push(std::move(stack));
            });

            // Update tot_n_bp with the data from the current chunk.
            tot_n_bp.fetch_add(coll_vec.size(), std::memory_order::relaxed);
        }
    });

    logger->trace("Broad phase collision detection time: {}s", sw);

    logger->debug("Average number of collisions per particle per chunk: {}",
                  static_cast<double>(tot_n_bp.load(std::memory_order::relaxed)) / static_cast<double>(nchunks)
                      / static_cast<double>(nparts));
}

} // namespace cascade