// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <atomic>
#include <cassert>
#include <initializer_list>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <cascade/detail/logging_impl.hpp>
#include <cascade/detail/sim_data.hpp>
#include <cascade/sim.hpp>

#if defined(__clang__) || defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"

#endif

#include "mdspan/mdspan"

#if defined(__clang__) || defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

namespace cascade
{

// Broad phase collision detection - i.e., collision
// detection between the AABBs of the particles' trajectories.
void sim::broad_phase_parallel()
{
    namespace stdex = std::experimental;

    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    // Fetch the number of particles and chunks from m_data.
    const auto nparts = get_nparts();
    const auto nchunks = m_data->nchunks;

    // Global counter for the total number of AABBs collisions
    // across all chunks.
    std::atomic<decltype(m_data->bp_coll[0].size())> tot_n_bp(0);

    // Views for accessing the sorted lbs/ubs data.
    using b_size_t = decltype(m_data->lbs.size());
    stdex::mdspan srt_lbs(std::as_const(m_data->srt_lbs).data(),
                          stdex::extents<b_size_t, stdex::dynamic_extent, stdex::dynamic_extent, 4u>(nchunks, nparts));
    stdex::mdspan srt_ubs(std::as_const(m_data->srt_ubs).data(),
                          stdex::extents<b_size_t, stdex::dynamic_extent, stdex::dynamic_extent, 4u>(nchunks, nparts));

    // View for accessing the indices vector.
    using idx_size_t = decltype(m_data->vidx.size());
    stdex::mdspan vidx(std::as_const(m_data->vidx).data(),
                       stdex::extents<idx_size_t, stdex::dynamic_extent, stdex::dynamic_extent>(nchunks, nparts));

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(0u, nchunks), [&](const auto &range) {
        for (auto chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
            // Fetch a reference to the tree.
            const auto &tree = m_data->bvh_trees[chunk_idx];

            // Fetch a reference to the AABB collision vector for the
            // current chunk and clear it out.
            auto &bp_cv = m_data->bp_coll[chunk_idx];
            bp_cv.clear();

            // Fetch a reference to the bp data cache for the current chunk.
            auto &bp_data_cache_ptr = m_data->bp_data_caches[chunk_idx];
            // NOTE: the pointer will require initialisation the first time
            // it is used.
            if (!bp_data_cache_ptr) {
                bp_data_cache_ptr
                    = std::make_unique<typename decltype(m_data->bp_data_caches)::value_type::element_type>();
            }
            auto &bp_data_cache = *bp_data_cache_ptr;

            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(0, nparts), [&](const auto &r2) {
                // Fetch local data for broad phase collision detection.
                std::unique_ptr<sim_data::bp_data> local_bp_data;
                if (bp_data_cache.try_pop(local_bp_data)) {
                    assert(local_bp_data);
                } else {
                    SPDLOG_LOGGER_DEBUG(logger, "Creating new local BP data");

                    local_bp_data = std::make_unique<sim_data::bp_data>();
                }

                // Cache and clear the local list of collisions.
                auto &local_bp = local_bp_data->bp;
                local_bp.clear();

                // Cache the stack.
                // NOTE: this will be cleared at the beginning
                // of the traversal for each particle.
                auto &stack = local_bp_data->stack;

                for (auto pidx = r2.begin(); pidx != r2.end(); ++pidx) {
                    // Reset the stack, and add the root node to it.
                    stack.clear();
                    stack.push_back(0);

                    // Cache the AABB of the current particle.
                    const auto x_lb = srt_lbs(chunk_idx, pidx, 0);
                    const auto y_lb = srt_lbs(chunk_idx, pidx, 1);
                    const auto z_lb = srt_lbs(chunk_idx, pidx, 2);
                    const auto r_lb = srt_lbs(chunk_idx, pidx, 3);

                    const auto x_ub = srt_ubs(chunk_idx, pidx, 0);
                    const auto y_ub = srt_ubs(chunk_idx, pidx, 1);
                    const auto z_ub = srt_ubs(chunk_idx, pidx, 2);
                    const auto r_ub = srt_ubs(chunk_idx, pidx, 3);

                    do {
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
                                // all particles in the node, unless either:
                                // - pidx is colliding with itself (pidx == i), or
                                // - pidx > i, in order to avoid counting twice
                                //   the collisions (pidx, i) and (i, pidx).
                                // NOTE: in case of a multi-particle leaf,
                                // the node's AABB is the composition of the AABBs
                                // of all particles in the node, and thus, in general,
                                // it is not strictly true tha pidx will overlap with
                                // *all* particles in the node. In other words, we will
                                // be detecting AABB overlaps which are not actually there.
                                // This is ok, as they will be filtered out in the
                                // next stages of collision detection.
                                // NOTE: we want to store the particle indices
                                // in the *original* order, not in the Morton order,
                                // hence the indirection via vidx_ptr.
                                for (auto i = cur_node.begin; i != cur_node.end; ++i) {
                                    if (vidx(chunk_idx, pidx) < vidx(chunk_idx, i)) {
                                        local_bp.emplace_back(vidx(chunk_idx, pidx), vidx(chunk_idx, i));
                                    }
                                }
                            } else {
                                // Internal node: add both children to the
                                // stack and iterate.
                                stack.push_back(cur_node.left);
                                stack.push_back(cur_node.right);
                            }
                        }
                    } while (!stack.empty());
                }

                // Atomically merge the local bp into the global one.
                bp_cv.grow_by(local_bp.begin(), local_bp.end());

                // Put the local data back into the cache.
                bp_data_cache.push(std::move(local_bp_data));
            });

            // Update tot_n_bp with the data from the current chunk.
            tot_n_bp.fetch_add(bp_cv.size(), std::memory_order::relaxed);
        }
    });

    logger->trace("Broad phase collision detection time: {}s", sw);

    logger->trace("Average number of AABB collisions per particle per chunk: {}",
                  static_cast<double>(tot_n_bp.load(std::memory_order::relaxed)) / static_cast<double>(nchunks)
                      / static_cast<double>(nparts));

#if !defined(NDEBUG)
    verify_broad_phase_parallel();
#endif
}

// Debug checks on the broad phase collision detection.
void sim::verify_broad_phase_parallel() const
{
    namespace stdex = std::experimental;

    const auto nparts = get_nparts();
    const auto nchunks = m_data->nchunks;

    // Don't run the check if there's too many particles.
    if (nparts > 10000u) {
        return;
    }

    // Views for accessing the lbs/ubs data.
    using b_size_t = decltype(m_data->lbs.size());
    stdex::mdspan lbs(std::as_const(m_data->lbs).data(),
                      stdex::extents<b_size_t, stdex::dynamic_extent, stdex::dynamic_extent, 4u>(nchunks, nparts));
    stdex::mdspan ubs(std::as_const(m_data->ubs).data(),
                      stdex::extents<b_size_t, stdex::dynamic_extent, stdex::dynamic_extent, 4u>(nchunks, nparts));

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(0u, nchunks), [&](const auto &range) {
        for (auto chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
            // Build a set version of the collision list
            // for fast lookup.
            std::set<std::pair<size_type, size_type>> coll_tree;
            for (const auto &p : m_data->bp_coll[chunk_idx]) {
                // Check that, for all collisions (i, j), i is always < j.
                assert(p.first < p.second);
                // Check that the collision pairs are unique.
                assert(coll_tree.emplace(p).second);
            }

            // A counter for the N**2 collision detection algorithm below.
            std::atomic<decltype(coll_tree.size())> coll_counter(0);

            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(0, nparts), [&](const auto &ri) {
                for (auto i = ri.begin(); i != ri.end(); ++i) {
                    const auto xi_lb = lbs(chunk_idx, i, 0);
                    const auto yi_lb = lbs(chunk_idx, i, 1);
                    const auto zi_lb = lbs(chunk_idx, i, 2);
                    const auto ri_lb = lbs(chunk_idx, i, 3);

                    const auto xi_ub = ubs(chunk_idx, i, 0);
                    const auto yi_ub = ubs(chunk_idx, i, 1);
                    const auto zi_ub = ubs(chunk_idx, i, 2);
                    const auto ri_ub = ubs(chunk_idx, i, 3);

                    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(i + 1u, nparts),
                                              [&](const auto &rj) {
                                                  decltype(coll_tree.size()) loc_ncoll = 0;

                                                  for (auto j = rj.begin(); j != rj.end(); ++j) {
                                                      const auto xj_lb = lbs(chunk_idx, j, 0);
                                                      const auto yj_lb = lbs(chunk_idx, j, 1);
                                                      const auto zj_lb = lbs(chunk_idx, j, 2);
                                                      const auto rj_lb = lbs(chunk_idx, j, 3);

                                                      const auto xj_ub = ubs(chunk_idx, j, 0);
                                                      const auto yj_ub = ubs(chunk_idx, j, 1);
                                                      const auto zj_ub = ubs(chunk_idx, j, 2);
                                                      const auto rj_ub = ubs(chunk_idx, j, 3);

                                                      const bool overlap = (xi_ub >= xj_lb && xi_lb <= xj_ub)
                                                                           && (yi_ub >= yj_lb && yi_lb <= yj_ub)
                                                                           && (zi_ub >= zj_lb && zi_lb <= zj_ub)
                                                                           && (ri_ub >= rj_lb && ri_lb <= rj_ub);

                                                      if (overlap) {
                                                          // Overlap detected in the simple algorithm:
                                                          // the collision must be present also
                                                          // in the tree code.
                                                          assert(coll_tree.find({i, j}) != coll_tree.end());
                                                      } else {
                                                          // NOTE: the contrary is not necessarily
                                                          // true: for multi-particle leaves, we
                                                          // may detect overlaps that do not actually exist.
                                                      }

                                                      loc_ncoll += overlap;
                                                  }

                                                  coll_counter.fetch_add(loc_ncoll, std::memory_order::relaxed);
                                              });
                }
            });

            // NOTE: in case of multi-particle leaves, we will have detected
            // non-existing AABBs overlaps. Thus, just require that the number
            // of collisions detected via the tree is at least as large
            // as the number of "true" collisions detected with the N**2 algorithm.
            assert(coll_tree.size() >= coll_counter.load());
        }
    });
}

} // namespace cascade
