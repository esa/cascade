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
#include <chrono>
#include <cstdint>
#include <limits>

#include <spdlog/stopwatch.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_scan.h>

#include <cascade/detail/logging_impl.hpp>
#include <cascade/detail/sim_data.hpp>
#include <cascade/sim.hpp>

namespace cascade
{

void sim::construct_bvh_trees()
{
    using bvh_tree_t = sim_data::bvh_tree_t;

    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    // Fetch the number of particles and chunks from m_data.
    const auto nparts = get_nparts();
    const auto nchunks = static_cast<unsigned>(m_data->global_lb.size());

    // Setup the initial values for the nodes' bounding boxes.
    constexpr auto finf = std::numeric_limits<float>::infinity();
    constexpr std::array<float, 4> default_lb = {finf, finf, finf, finf};
    constexpr std::array<float, 4> default_ub = {-finf, -finf, -finf, -finf};

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(0u, nchunks), [&](const auto &range) {
        for (auto chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
            const auto offset = nparts * chunk_idx;

            const auto mcodes_ptr = m_data->srt_mcodes.data() + offset;

            // Fetch a reference to the tree and clear it out.
            auto &tree = m_data->bvh_trees[chunk_idx];
            tree.clear();

            // Fetch references to the temp buffers and
            // clear them out.
            auto &nc_buf = m_data->nc_buffer[chunk_idx];
            auto &ps_buf = m_data->ps_buffer[chunk_idx];
            auto &nplc_buf = m_data->nplc_buffer[chunk_idx];
            nc_buf.clear();
            ps_buf.clear();
            nplc_buf.clear();

            // Insert the root node.
            tree.emplace_back(0, nparts, -1, -1, -1, default_lb, default_ub, 0);

            // The number of nodes at the current tree level.
            bvh_tree_t::size_type cur_n_nodes = 1;

            while (cur_n_nodes != 0u) {
                // Fetch the current tree size.
                const auto cur_tree_size = tree.size();

                // The node index range for the iteration at the
                // current level.
                const auto n_begin = cur_tree_size - cur_n_nodes;
                const auto n_end = cur_tree_size;

                // Number of nodes at the next level, inited
                // with the maximum possible value.
                // TODO overflow check?
                auto next_n_nodes = cur_n_nodes * 2u;

                // Prepare the temp buffers.
                // TODO numeric casts.
                nc_buf.resize(cur_n_nodes);
                ps_buf.resize(cur_n_nodes);
                nplc_buf.resize(cur_n_nodes);

                // For each node in a range, this function will:
                // - determine if the node is a leaf, and
                // - if it is *not* a leaf, how many particles
                //   are in the left child.
                auto node_split = [&](auto rbegin, auto rend) {
                    // Local accumulator for the number of leaf nodes
                    // detected in the range.
                    bvh_tree_t::size_type n_leaf_nodes = 0;

                    for (auto node_idx = rbegin; node_idx != rend; ++node_idx) {
                        auto &cur_node = tree[node_idx];

                        // Flag to signal that this is a leaf node.
                        bool is_leaf_node = false;

                        const std::uint64_t *split_ptr;

                        const auto mcodes_begin = mcodes_ptr + cur_node.begin;
                        const auto mcodes_end = mcodes_ptr + cur_node.end;

                        if (cur_node.end - cur_node.begin > 1u) {
                            // The node contains more than 1 particle.
                            // Figure out where the bit at index cur_node.split_idx flips from 0 to 1
                            // for the Morton codes in the range.
                            split_ptr = std::lower_bound(
                                mcodes_begin, mcodes_end, 1u,
                                [mask = std::uint64_t(1) << (63 - cur_node.split_idx)](
                                    std::uint64_t mcode, unsigned val) { return (mcode & mask) < val; });

                            while (split_ptr == mcodes_begin || split_ptr == mcodes_end) {
                                // There is no bit flip at the current index.
                                // We will try the next bit index.

                                if (cur_node.split_idx == 63) {
                                    // No more bit indices are available.
                                    // This will be a leaf node containing more than 1 particle.
                                    is_leaf_node = true;

                                    break;
                                }

                                // Bump up the bit index and look
                                // again for the bit flip.
                                ++cur_node.split_idx;
                                split_ptr = std::lower_bound(
                                    mcodes_begin, mcodes_end, 1u,
                                    [mask = std::uint64_t(1) << (63 - cur_node.split_idx)](
                                        std::uint64_t mcode, unsigned val) { return (mcode & mask) < val; });
                            }
                        } else {
                            // Node with a single particle, leaf.
                            is_leaf_node = true;
                        }

                        if (is_leaf_node) {
                            // A leaf node has no children.
                            nc_buf[node_idx - n_begin].val = 0;
                            nplc_buf[node_idx - n_begin].val = 0;

                            // Update the leaf nodes counter.
                            ++n_leaf_nodes;
                        } else {
                            // An internal node has 2 children.
                            nc_buf[node_idx - n_begin].val = 2;
                            // NOTE: if we are here, it means that is_leaf_node is false,
                            // which implies that split_ptr was written to at least once.
                            // TODO overflow check.
                            nplc_buf[node_idx - n_begin].val = split_ptr - mcodes_begin;
                        }
                    }

                    // Atomically decrease next_n_nodes by n_leaf_nodes * 2.
                    std::atomic_ref<bvh_tree_t::size_type> next_n_nodes_at(next_n_nodes);
                    next_n_nodes_at.fetch_sub(n_leaf_nodes * 2u, std::memory_order::relaxed);
                };

                // For each node in a range, and using the
                // data gathered in the temp buffers,
                // this function will compute:
                // - the pointers to the children, if any, and
                // - the children's initial properties.
                auto node_writer = [&](auto rbegin, auto rend) {
                    for (auto node_idx = rbegin; node_idx != rend; ++node_idx) {
                        auto &cur_node = tree[node_idx];

                        // Fetch the number of children.
                        const auto nc = nc_buf[node_idx - n_begin];

                        if (nc.val == 0u) {
                            // Leaf node.
                            // TODO AABB computation and update AABBs of ancestors, and ensure
                            // all node properties are set up in cur_node (only the children
                            // are missing here I think).
                        } else {
                            // Fetch the number of particles in the left child.
                            const auto lsize = nplc_buf[node_idx - n_begin];

                            // Compute the index in the tree into which the left child will
                            // be stored.
                            // NOTE: this computation is safe because we checked earlier
                            // that cur_tree_size + next_n_nodes can be computed safely.
                            const auto lc_idx = cur_tree_size + ps_buf[node_idx - n_begin].val - 2u;

                            // Assign the children indices for the current node.
                            // TODO numeric casts.
                            cur_node.left = lc_idx;
                            cur_node.right = lc_idx + 1u;

                            // Set up the children.
                            auto &lc = tree[lc_idx];
                            auto &rc = tree[lc_idx + 1u];

                            lc.begin = cur_node.begin;
                            lc.end = cur_node.begin + lsize.val;
                            lc.parent = node_idx;
                            lc.left = -1;
                            lc.right = -1;
                            lc.lb = default_lb;
                            lc.ub = default_ub;
                            lc.split_idx = cur_node.split_idx + 1;

                            rc.begin = cur_node.begin + lsize.val;
                            rc.end = cur_node.end;
                            rc.parent = node_idx;
                            rc.left = -1;
                            rc.right = -1;
                            rc.lb = default_lb;
                            rc.ub = default_ub;
                            rc.split_idx = cur_node.split_idx + 1;
                        }
                    }
                };

                // Step 1: determine, for each node in the range,
                // if the node is a leaf or not, and, if so, the number
                // of particles in the left child.
                oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(n_begin, n_end),
                                          [&](const auto &range) { node_split(range.begin(), range.end()); });

                // Step 2: prepare the tree for the new nodes.
                // NOTE: next_n_nodes was computed in the previous step.
                // TODO numeric cast, overflow check.
                tree.resize(cur_tree_size + next_n_nodes);

                // Step 3: prefix sum over the number of children for each
                // node in the range.
                oneapi::tbb::parallel_scan(
                    oneapi::tbb::blocked_range<decltype(nc_buf.size())>(0, nc_buf.size()),
                    sim_data::uninit<bvh_tree_t::size_type>{0},
                    [&](const auto &r, auto sum, bool is_final_scan) {
                        auto temp = sum;

                        for (auto i = r.begin(); i < r.end(); ++i) {
                            temp.val = temp.val + nc_buf[i].val;

                            if (is_final_scan) {
                                ps_buf[i].val = temp.val;
                            }
                        }

                        return temp;
                    },
                    [](auto left, auto right) {
                        return sim_data::uninit<bvh_tree_t::size_type>{left.val + right.val};
                    });

                // Step 4: finalise the nodes in the range with the children pointers,
                // and perform the initial setup of the children nodes.
                oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(n_begin, n_end),
                                          [&](const auto &range) { node_writer(range.begin(), range.end()); });

                // Assign the next value for cur_n_nodes.
                // If next_n_nodes is zero, this means that
                // all the nodes processed in this iteration
                // were leaves, and this signals the end of the
                // construction of the tree.
                cur_n_nodes = next_n_nodes;
            }
        }
    });

    logger->trace("BVH construction time: {}s", sw);
}

} // namespace cascade
