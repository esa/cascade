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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <type_traits>

#include <boost/numeric/conversion/cast.hpp>

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

namespace detail
{

namespace
{

#if !defined(NDEBUG) && (defined(__GNUC__) || defined(__clang__))

// Debug helper to compute the index of the first different
// bit between n1 and n2, starting from the MSB.
template <typename T>
int first_diff_bit(T n1, T n2)
{
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);

    const auto res_xor = n1 ^ n2;

    if (res_xor == 0u) {
        return std::numeric_limits<T>::digits;
    } else {
        if constexpr (std::is_same_v<T, unsigned>) {
            return __builtin_clz(res_xor);
        } else if constexpr (std::is_same_v<T, unsigned long>) {
            return __builtin_clzl(res_xor);
        } else if constexpr (std::is_same_v<T, unsigned long long>) {
            return __builtin_clzll(res_xor);
        } else {
            assert(false);
            throw;
        }
    }
}

#endif

} // namespace

} // namespace detail

// Construct the BVH tree for each chunk.
void sim::construct_bvh_trees()
{
    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    // Fetch the number of particles and chunks from m_data.
    const auto nparts = get_nparts();
    const auto nchunks = m_data->nchunks;

    // Initial values for the nodes' bounding boxes.
    constexpr auto finf = std::numeric_limits<float>::infinity();
    constexpr std::array<float, 4> default_lb = {finf, finf, finf, finf};
    constexpr std::array<float, 4> default_ub = {-finf, -finf, -finf, -finf};

    // Overflow check: we need to be able to represent all pointer differences
    // in the Morton codes vector.
    constexpr auto overflow_err_msg = "Overflow detected during the construction of a BVH tree";

    if (m_data->srt_mcodes.size()
        > static_cast<std::make_unsigned_t<std::ptrdiff_t>>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::overflow_error(overflow_err_msg);
    }

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
            tree.emplace_back(0, boost::numeric_cast<std::uint32_t>(nparts), -1, -1, -1, default_lb, default_ub, 0, 0);

            // The number of nodes at the current tree level.
            std::uint32_t cur_n_nodes = 1;

            // The total number of levels and nodes.
            std::uint32_t n_levels = 0, n_nodes = 0;

            while (cur_n_nodes != 0u) {
                // Fetch the current tree size.
                const auto cur_tree_size = tree.size();

                // The node index range for the iteration at the
                // current level.
                const auto n_begin = cur_tree_size - cur_n_nodes;
                const auto n_end = cur_tree_size;

                // Number of nodes at the next level, inited
                // with the maximum possible value.
                if (cur_n_nodes > std::numeric_limits<std::uint32_t>::max() / 2u) {
                    throw std::overflow_error(overflow_err_msg);
                }
                auto nn_next_level = cur_n_nodes * 2u;

                // Prepare the temp buffers.
                nc_buf.resize(boost::numeric_cast<decltype(nc_buf.size())>(cur_n_nodes));
                ps_buf.resize(boost::numeric_cast<decltype(ps_buf.size())>(cur_n_nodes));
                nplc_buf.resize(boost::numeric_cast<decltype(nplc_buf.size())>(cur_n_nodes));

                // For each node in a range, this function will:
                // - determine if the node is a leaf, and
                // - if it is *not* a leaf, how many particles
                //   are in the left child.
                auto node_split = [&](auto rbegin, auto rend) {
                    // Local accumulator for the number of leaf nodes
                    // detected in the range.
                    std::uint32_t n_leaf_nodes = 0;

                    // NOTE: this for loop can *probably* be written in a vectorised
                    // fashion, using the gather primitives as done in heyoka.
                    for (auto node_idx = rbegin; node_idx != rend; ++node_idx) {
                        assert(node_idx - n_begin < cur_n_nodes);

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
                            nplc_buf[node_idx - n_begin] = boost::numeric_cast<std::uint32_t>(split_ptr - mcodes_begin);
                        }
                    }

                    // Atomically decrease nn_next_level by n_leaf_nodes * 2.
                    std::atomic_ref nn_next_level_at(nn_next_level);
                    nn_next_level_at.fetch_sub(n_leaf_nodes * 2u, std::memory_order::relaxed);
                };

                // For each node in a range, and using the
                // data gathered in the temp buffers,
                // this function will compute:
                // - the pointers to the children, if any, and
                // - the children's initial properties.
                auto node_writer = [&](auto rbegin, auto rend) {
                    for (auto node_idx = rbegin; node_idx != rend; ++node_idx) {
                        assert(node_idx - n_begin < cur_n_nodes);

                        auto &cur_node = tree[node_idx];

                        // Fetch the number of children.
                        const auto nc = nc_buf[node_idx - n_begin];

                        // Set the nn_level member. This needs to be done
                        // regardless of whether the node is internal or a leaf.
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
                            assert(lc_idx >= cur_tree_size);
                            assert(lc_idx < tree.size());
                            assert(lc_idx + 1u > cur_tree_size);
                            assert(lc_idx + 1u < tree.size());

                            // Assign the children indices for the current node.
                            cur_node.left = boost::numeric_cast<decltype(cur_node.left)>(lc_idx);
                            cur_node.right = boost::numeric_cast<decltype(cur_node.right)>(lc_idx + 1u);

                            // Set up the children's initial properties.
                            auto &lc = tree[lc_idx];
                            auto &rc = tree[lc_idx + 1u];

                            lc.begin = cur_node.begin;
                            // NOTE: the computation is safe
                            // because we know we can represent nparts
                            // as a std::uint32_t.
                            lc.end = cur_node.begin + lsize;
                            lc.parent = boost::numeric_cast<std::int32_t>(node_idx);
                            lc.left = -1;
                            lc.right = -1;
                            lc.lb = default_lb;
                            lc.ub = default_ub;
                            lc.nn_level = 0;
                            lc.split_idx = cur_node.split_idx + 1;

                            rc.begin = cur_node.begin + lsize;
                            rc.end = cur_node.end;
                            rc.parent = boost::numeric_cast<std::int32_t>(node_idx);
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
                                          [&](const auto &rn) { node_split(rn.begin(), rn.end()); });

                // Step 2: prepare the tree for the new nodes.
                // NOTE: nn_next_level was computed in the previous step.
                if (nn_next_level > std::numeric_limits<decltype(cur_tree_size)>::max() - cur_tree_size) {
                    throw std::overflow_error(overflow_err_msg);
                }
                tree.resize(cur_tree_size + nn_next_level);

                // Step 3: prefix sum over the number of children for each
                // node in the range.
                oneapi::tbb::parallel_scan(
                    oneapi::tbb::blocked_range<decltype(nc_buf.size())>(0, nc_buf.size()), std::uint32_t(0),
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
                                          [&](const auto &rn) { node_writer(rn.begin(), rn.end()); });

                // Assign the next value for cur_n_nodes.
                // If nn_next_level is zero, this means that
                // all the nodes processed in this iteration
                // were leaves, and this signals the end of the
                // construction of the tree.
                cur_n_nodes = nn_next_level;

                ++n_levels;
                n_nodes += cur_n_nodes;
            }

            // Perform a backwards pass on the tree to compute the AABBs
            // of the internal nodes.

            // Node index range for the last level.
            auto n_begin = tree.size() - tree.back().nn_level;
            auto n_end = tree.size();

            while (true) {
                oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(n_begin, n_end), [&](const auto &rn) {
                    for (auto node_idx = rn.begin(); node_idx != rn.end(); ++node_idx) {
                        auto &cur_node = tree[node_idx];

                        if (cur_node.left == -1) {
                            // Leaf node, compute the bounding box.
                            for (auto pidx = cur_node.begin; pidx != cur_node.end; ++pidx) {
                                // NOTE: min/max is fine here, we already checked
                                // that all AABBs are finite.
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
                            auto &lc = tree[static_cast<decltype(tree.size())>(cur_node.left)];
                            auto &rc = tree[static_cast<decltype(tree.size())>(cur_node.right)];

                            for (auto j = 0u; j < 4u; ++j) {
                                // NOTE: min/max is fine here, we already checked
                                // that all AABBs are finite.
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

            SPDLOG_LOGGER_DEBUG(logger, "Tree levels/nodes for chunk {}: {}/{}", chunk_idx, n_levels, n_nodes);
        }
    });

    logger->trace("BVH construction time: {}s", sw);

#if !defined(NDEBUG)
    verify_bvh_trees();
#endif
}

void sim::verify_bvh_trees() const
{
    const auto nparts = get_nparts();
    const auto nchunks = m_data->nchunks;

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(0u, nchunks), [&](const auto &range) {
        for (auto chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
            const auto offset = nparts * chunk_idx;

            const auto *CASCADE_RESTRICT x_lb_ptr = m_data->srt_x_lb.data() + offset;
            const auto *CASCADE_RESTRICT y_lb_ptr = m_data->srt_y_lb.data() + offset;
            const auto *CASCADE_RESTRICT z_lb_ptr = m_data->srt_z_lb.data() + offset;
            const auto *CASCADE_RESTRICT r_lb_ptr = m_data->srt_r_lb.data() + offset;

            const auto *CASCADE_RESTRICT ux_lb_ptr = m_data->x_lb.data() + offset;
            const auto *CASCADE_RESTRICT uy_lb_ptr = m_data->y_lb.data() + offset;
            const auto *CASCADE_RESTRICT uz_lb_ptr = m_data->z_lb.data() + offset;
            const auto *CASCADE_RESTRICT ur_lb_ptr = m_data->r_lb.data() + offset;

            const auto *CASCADE_RESTRICT x_ub_ptr = m_data->srt_x_ub.data() + offset;
            const auto *CASCADE_RESTRICT y_ub_ptr = m_data->srt_y_ub.data() + offset;
            const auto *CASCADE_RESTRICT z_ub_ptr = m_data->srt_z_ub.data() + offset;
            const auto *CASCADE_RESTRICT r_ub_ptr = m_data->srt_r_ub.data() + offset;

            const auto *CASCADE_RESTRICT ux_ub_ptr = m_data->x_ub.data() + offset;
            const auto *CASCADE_RESTRICT uy_ub_ptr = m_data->y_ub.data() + offset;
            const auto *CASCADE_RESTRICT uz_ub_ptr = m_data->z_ub.data() + offset;
            const auto *CASCADE_RESTRICT ur_ub_ptr = m_data->r_ub.data() + offset;

            const auto *CASCADE_RESTRICT mcodes_ptr = m_data->srt_mcodes.data() + offset;
            const auto *CASCADE_RESTRICT umcodes_ptr = m_data->mcodes.data() + offset;

            const auto *CASCADE_RESTRICT vidx_ptr = m_data->vidx.data() + offset;

            const auto &bvh_tree = m_data->bvh_trees[chunk_idx];

            std::set<size_type> pset;

            for (decltype(bvh_tree.size()) i = 0; i < bvh_tree.size(); ++i) {
                const auto &cur_node = bvh_tree[i];

                // The node must contain 1 or more particles.
                assert(cur_node.end > cur_node.begin);

                // The node must have either 0 or 2 children.
                if (cur_node.left == -1) {
                    assert(cur_node.right == -1);
                } else {
                    assert(cur_node.left > 0);
                    assert(cur_node.right > 0);
                }

                if (cur_node.end - cur_node.begin == 1u) {
                    // A node with a single particle is a leaf and must have no children.
                    assert(cur_node.left == -1);
                    assert(cur_node.right == -1);

                    // Add the particle to the global particle set,
                    // ensuring the particle has not been added to pset yet.
                    assert(pset.find(boost::numeric_cast<size_type>(cur_node.begin)) == pset.end());
                    pset.insert(boost::numeric_cast<size_type>(cur_node.begin));
                } else if (cur_node.left == -1) {
                    // A leaf with multiple particles.
                    assert(cur_node.right == -1);

                    // All particles must have the same Morton code.
                    const auto mc = mcodes_ptr[cur_node.begin];

                    // Make also sure that all particles are accounted
                    // for in pset.
                    assert(pset.find(boost::numeric_cast<size_type>(cur_node.begin)) == pset.end());
                    pset.insert(boost::numeric_cast<size_type>(cur_node.begin));

                    for (auto j = cur_node.begin + 1u; j < cur_node.end; ++j) {
                        assert(mcodes_ptr[j] == mc);

                        assert(pset.find(boost::numeric_cast<size_type>(j)) == pset.end());
                        pset.insert(boost::numeric_cast<size_type>(j));
                    }
                }

                if (cur_node.left != -1) {
                    // A node with children.
                    assert(cur_node.left > 0);
                    assert(cur_node.right > 0);

                    const auto uleft = static_cast<std::uint32_t>(cur_node.left);
                    const auto uright = static_cast<std::uint32_t>(cur_node.right);

                    // The children indices must be greater than the current node's
                    // index and within the tree.
                    assert(uleft > i && uleft < bvh_tree.size());
                    assert(uright > i && uright < bvh_tree.size());

                    // Check that the ranges of the children are consistent with
                    // the range of the current node.
                    assert(bvh_tree[uleft].begin == cur_node.begin);
                    assert(bvh_tree[uleft].end < cur_node.end);
                    assert(bvh_tree[uright].begin == bvh_tree[uleft].end);
                    assert(bvh_tree[uright].end == cur_node.end);

#if defined(__GNUC__) || defined(__clang__)
                    // Check that a node with children was split correctly (i.e.,
                    // cur_node.split_idx corresponds to the index of the first
                    // different bit at the boundary between first and second child).
                    const auto split_idx = bvh_tree[uleft].end - 1u;
                    assert(detail::first_diff_bit(mcodes_ptr[split_idx], mcodes_ptr[split_idx + 1u])
                           == cur_node.split_idx);
                    assert(mcodes_ptr[split_idx] == umcodes_ptr[vidx_ptr[split_idx]]);
#endif
                }

                // Check the parent info.
                if (i == 0u) {
                    assert(cur_node.parent == -1);
                } else {
                    assert(cur_node.parent >= 0);

                    const auto upar = static_cast<std::uint32_t>(cur_node.parent);

                    assert(upar < i);
                    assert(cur_node.begin >= bvh_tree[upar].begin);
                    assert(cur_node.end <= bvh_tree[upar].end);
                    assert(cur_node.begin == bvh_tree[upar].begin || cur_node.end == bvh_tree[upar].end);
                }

                // nn_level must alway be nonzero.
                assert(cur_node.nn_level > 0u);

                // Check that the AABB of the node is correct.
                constexpr auto finf = std::numeric_limits<float>::infinity();
                std::array<float, 4> lb = {finf, finf, finf, finf};
                std::array<float, 4> ub = {-finf, -finf, -finf, -finf};

                for (auto j = cur_node.begin; j < cur_node.end; ++j) {
                    assert(x_lb_ptr[j] == ux_lb_ptr[vidx_ptr[j]]);
                    assert(y_lb_ptr[j] == uy_lb_ptr[vidx_ptr[j]]);
                    assert(z_lb_ptr[j] == uz_lb_ptr[vidx_ptr[j]]);
                    assert(r_lb_ptr[j] == ur_lb_ptr[vidx_ptr[j]]);

                    lb[0] = std::min(lb[0], x_lb_ptr[j]);
                    lb[1] = std::min(lb[1], y_lb_ptr[j]);
                    lb[2] = std::min(lb[2], z_lb_ptr[j]);
                    lb[3] = std::min(lb[3], r_lb_ptr[j]);

                    assert(x_ub_ptr[j] == ux_ub_ptr[vidx_ptr[j]]);
                    assert(y_ub_ptr[j] == uy_ub_ptr[vidx_ptr[j]]);
                    assert(z_ub_ptr[j] == uz_ub_ptr[vidx_ptr[j]]);
                    assert(r_ub_ptr[j] == ur_ub_ptr[vidx_ptr[j]]);

                    ub[0] = std::max(ub[0], x_ub_ptr[j]);
                    ub[1] = std::max(ub[1], y_ub_ptr[j]);
                    ub[2] = std::max(ub[2], z_ub_ptr[j]);
                    ub[3] = std::max(ub[3], r_ub_ptr[j]);
                }

                assert(lb == cur_node.lb);
                assert(ub == cur_node.ub);
            }
        }
    });
}

} // namespace cascade
