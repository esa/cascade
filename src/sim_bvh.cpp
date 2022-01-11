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

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER)

#define CASCADE_RESTRICT __restrict

#else

#define CASCADE_RESTRICT

#endif

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

            const auto *CASCADE_RESTRICT x_lb_ptr = m_data->srt_x_lb.data() + offset;
            const auto *CASCADE_RESTRICT y_lb_ptr = m_data->srt_y_lb.data() + offset;
            const auto *CASCADE_RESTRICT z_lb_ptr = m_data->srt_z_lb.data() + offset;
            const auto *CASCADE_RESTRICT r_lb_ptr = m_data->srt_r_lb.data() + offset;

            const auto *CASCADE_RESTRICT x_ub_ptr = m_data->srt_x_ub.data() + offset;
            const auto *CASCADE_RESTRICT y_ub_ptr = m_data->srt_y_ub.data() + offset;
            const auto *CASCADE_RESTRICT z_ub_ptr = m_data->srt_z_ub.data() + offset;
            const auto *CASCADE_RESTRICT r_ub_ptr = m_data->srt_r_ub.data() + offset;

            const auto *CASCADE_RESTRICT mcodes_ptr = m_data->srt_mcodes.data() + offset;

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
            // NOTE: nn_level is inited to zero, even if we already know it is 1.
            tree.emplace_back(0, nparts, -1, -1, -1, default_lb, default_ub, 0, 0);

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
                auto nn_next_level = cur_n_nodes * 2u;

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
                            nc_buf[node_idx - n_begin] = 0;
                            nplc_buf[node_idx - n_begin] = 0;

                            // Update the leaf nodes counter.
                            ++n_leaf_nodes;
                        } else {
                            // An internal node has 2 children.
                            nc_buf[node_idx - n_begin] = 2;
                            // NOTE: if we are here, it means that is_leaf_node is false,
                            // which implies that split_ptr was written to at least once.
                            // TODO overflow check.
                            nplc_buf[node_idx - n_begin] = split_ptr - mcodes_begin;
                        }
                    }

                    // Atomically decrease nn_next_level by n_leaf_nodes * 2.
                    std::atomic_ref<bvh_tree_t::size_type> nn_next_level_at(nn_next_level);
                    nn_next_level_at.fetch_sub(n_leaf_nodes * 2u, std::memory_order::relaxed);
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

                        // Set the nn_level member. This needs to be done
                        // regardless of whether the node is internal or a leaf.
                        // TODO numeric cast.
                        cur_node.nn_level = cur_n_nodes;

                        // NOTE: for a leaf node, the left/right indices are already set to -1:
                        // if cur_node is the root node, it was inited properly
                        // by the initial insertion in the tree, otherwise,
                        // when inserting new children nodes in the tree below, we ensure
                        // to prepare children nodes with left/right already set to -1.

                        if (nc != 0u) {
                            // Internal node.

                            // Fetch the number of particles in the left child.
                            const auto lsize = nplc_buf[node_idx - n_begin];

                            // Compute the index in the tree into which the left child will
                            // be stored.
                            // NOTE: this computation is safe because we checked earlier
                            // that cur_tree_size + nn_next_level can be computed safely.
                            const auto lc_idx = cur_tree_size + ps_buf[node_idx - n_begin] - 2u;

                            // Assign the children indices for the current node.
                            // TODO numeric casts.
                            cur_node.left = lc_idx;
                            cur_node.right = lc_idx + 1u;

                            // Set up the children's initial properties.
                            auto &lc = tree[lc_idx];
                            auto &rc = tree[lc_idx + 1u];

                            lc.begin = cur_node.begin;
                            lc.end = cur_node.begin + lsize;
                            lc.parent = node_idx;
                            lc.left = -1;
                            lc.right = -1;
                            lc.lb = default_lb;
                            lc.ub = default_ub;
                            lc.nn_level = 0;
                            lc.split_idx = cur_node.split_idx + 1;

                            rc.begin = cur_node.begin + lsize;
                            rc.end = cur_node.end;
                            rc.parent = node_idx;
                            rc.left = -1;
                            rc.right = -1;
                            rc.lb = default_lb;
                            rc.ub = default_ub;
                            rc.nn_level = 0;
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
                // NOTE: nn_next_level was computed in the previous step.
                // TODO numeric cast, overflow check.
                tree.resize(cur_tree_size + nn_next_level);

                // Step 3: prefix sum over the number of children for each
                // node in the range.
                oneapi::tbb::parallel_scan(
                    oneapi::tbb::blocked_range<decltype(nc_buf.size())>(0, nc_buf.size()), bvh_tree_t::size_type(0),
                    [&](const auto &r, auto sum, bool is_final_scan) {
                        auto temp = sum;

                        for (auto i = r.begin(); i < r.end(); ++i) {
                            temp = temp + nc_buf[i];

                            if (is_final_scan) {
                                ps_buf[i] = temp;
                            }
                        }

                        return temp;
                    },
                    [](auto left, auto right) { return left + right; });

                // Step 4: finalise the nodes in the range with the children pointers,
                // and perform the initial setup of the children nodes.
                oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(n_begin, n_end),
                                          [&](const auto &range) { node_writer(range.begin(), range.end()); });

                // Assign the next value for cur_n_nodes.
                // If nn_next_level is zero, this means that
                // all the nodes processed in this iteration
                // were leaves, and this signals the end of the
                // construction of the tree.
                cur_n_nodes = nn_next_level;
            }

            // Perform a backwards pass on the tree to compute the AABBs
            // of the internal nodes.

            // Node index range for the last level.
            auto n_begin = tree.size() - tree.back().nn_level;
            auto n_end = tree.size();

            while (true) {
                oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(n_begin, n_end), [&](const auto &range) {
                    for (auto node_idx = range.begin(); node_idx != range.end(); ++node_idx) {
                        auto &cur_node = tree[node_idx];

                        if (cur_node.left == -1) {
                            // Leaf node, compute the bounding box.
                            for (auto pidx = cur_node.begin; pidx != cur_node.end; ++pidx) {
                                // TODO min max usage?
                                cur_node.lb[0] = std::min(cur_node.lb[0], x_lb_ptr[pidx]);
                                cur_node.lb[1] = std::min(cur_node.lb[1], y_lb_ptr[pidx]);
                                cur_node.lb[2] = std::min(cur_node.lb[2], z_lb_ptr[pidx]);
                                cur_node.lb[3] = std::min(cur_node.lb[3], r_lb_ptr[pidx]);

                                cur_node.ub[0] = std::max(cur_node.ub[0], x_ub_ptr[pidx]);
                                cur_node.ub[1] = std::max(cur_node.ub[1], y_ub_ptr[pidx]);
                                cur_node.ub[2] = std::max(cur_node.ub[2], z_ub_ptr[pidx]);
                                cur_node.ub[3] = std::max(cur_node.ub[3], r_ub_ptr[pidx]);
                            }
                        } else {
                            // Internal node, compute its AABB from the children.
                            auto &lc = tree[static_cast<bvh_tree_t::size_type>(cur_node.left)];
                            auto &rc = tree[static_cast<bvh_tree_t::size_type>(cur_node.right)];

                            for (auto j = 0u; j < 4u; ++j) {
                                // TODO min/max usage?
                                cur_node.lb[j] = std::min(lc.lb[j], rc.lb[j]);
                                cur_node.ub[j] = std::max(lc.ub[j], rc.ub[j]);
                            }
                        }
                    }
                });

                if (n_begin == 0u) {
                    break;
                } else {
                    const auto new_n_end = n_begin;
                    n_begin -= tree[n_begin - 1u].nn_level;
                    n_end = new_n_end;
                }
            }
        }
    });

    logger->trace("BVH construction time: {}s", sw);
}

} // namespace cascade
