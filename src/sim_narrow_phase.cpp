// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/core.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>

#include <heyoka/detail/dfloat.hpp>
#include <heyoka/taylor.hpp>

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

void sim::narrow_phase_parallel()
{
    namespace hy = heyoka;
    using dfloat = hy::detail::dfloat<double>;
    namespace stdex = std::experimental;

    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    // Cache a few bits.
    const auto nchunks = m_data->nchunks;
    const auto &s_data = m_data->s_data;
    const auto batch_size = m_data->b_ta.get_batch_size();
    const auto order = m_data->s_ta.get_order();
    const auto nparts = get_nparts();
    const auto npars = get_npars();

    // Reset the collision vector.
    m_data->coll_vec.clear();

    // Fetch a view on the state vector and on the pars.
    stdex::mdspan sv(std::as_const(m_state)->data(), stdex::extents<size_type, stdex::dynamic_extent, 7u>(nparts));
    stdex::mdspan pv(std::as_const(m_pars)->data(), nparts, npars);

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(0u, nchunks), [&](const auto &range) {
        for (auto chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
            // Fetch a reference to the chunk-specific broad
            // phase collision vector.
            const auto &bpc = m_data->bp_coll[chunk_idx];

            // Fetch a reference to the c_ta_cache pointer for the current chunk.
            auto &c_ta_cache_ptr = m_data->c_ta_caches[chunk_idx];
            // NOTE: the pointer will require initialisation the first time
            // it is used.
            if (!c_ta_cache_ptr) {
                c_ta_cache_ptr = std::make_unique<typename decltype(m_data->c_ta_caches)::value_type::element_type>();
            }
            auto &c_ta_cache = *c_ta_cache_ptr;

            // The time coordinate, relative to the beginning of the superatep,
            // of the chunk's begin/end.
            const auto c_begin_end = m_data->get_chunk_begin_end(chunk_idx, m_ct);
            const auto c_begin = c_begin_end[0];
            const auto c_end = c_begin_end[1];
            const auto chunk_begin = dfloat(c_begin);

            // Iterate over all candidate collisions in groups of size batch_size.
            // If bpc.size() is not divided exactly by batch_size, the last group
            // will contain fewer than batch_size elements and will need to be
            // padded (see below).
            const auto bpc_size = bpc.size();
            const auto n_batches = bpc_size / batch_size + static_cast<unsigned>(bpc_size % batch_size != 0u);

            // NOTE: due to possible padding in the last group, we need to run an overflow check
            // as we might need to compute numbers larger than bpc.size() via the type
            // decltype(bpc.size()).
            using safe_bpc_size_t = boost::safe_numerics::safe<decltype(bpc.size())>;
            (void)(safe_bpc_size_t(n_batches) * batch_size);

            oneapi::tbb::parallel_for(
                oneapi::tbb::blocked_range<decltype(bpc.size())>(0, n_batches), [&](const auto &rn) {
                    // Try to fetch an integrator from the cache.
                    std::unique_ptr<heyoka::taylor_adaptive_batch<double>> ta_ptr;
                    if (c_ta_cache.try_pop(ta_ptr)) {
                        // Fetch successful.
                        assert(ta_ptr);
                    } else {
                        // Fetch failed, init a new integrator.
                        SPDLOG_LOGGER_DEBUG(logger,
                                            "Creating new local integator for narrow phase collision detection");

                        ta_ptr = std::make_unique<heyoka::taylor_adaptive_batch<double>>(m_data->c_ta);
                    }
                    auto &ta = *ta_ptr;

                    // Fetch views into the state/pars data for the integrator.
                    stdex::mdspan st(ta.get_state_data(),
                                     stdex::extents<std::uint32_t, 12u, stdex::dynamic_extent>(batch_size));
                    using safe_u32_t = boost::safe_numerics::safe<std::uint32_t>;
                    stdex::mdspan pt(ta.get_pars_data(), static_cast<std::uint32_t>(2u * safe_u32_t(npars) + 2u),
                                     batch_size);

                    for (auto ridx = rn.begin(); ridx != rn.end(); ++ridx) {
                        // Reset the cooldowns in ta.
                        // NOTE: there's always at least the collision event
                        // in this integrator, thus no need to check ta.with_events().
                        ta.reset_cooldowns();

                        // Setup the state/pars for the integrator.
                        for (std::uint32_t bidx = 0; bidx < batch_size; ++bidx) {
                            // Compute the current index into bpc.
                            auto bpc_idx = ridx * batch_size + bidx;

                            // NOTE: the last group might contain fewer
                            // than batch_size elements. In such a case,
                            // clamp bpc_idx to point to the last element
                            // in bpc. This means that, in the last group
                            // of candidate collisions, the last candidate
                            // might appear multiple times at the end of the group.
                            bpc_idx = (bpc_idx < bpc_size) ? bpc_idx : (bpc_size - 1u);

                            // Fetch the particles' indices.
                            const auto [pi, pj] = bpc[bpc_idx];

                            assert(pi != pj);

                            // Fetch a reference to the substep data
                            // for the two particles.
                            const auto &sd_i = s_data[pi];
                            const auto &sd_j = s_data[pj];

                            // Cache the range of end times of the substeps
                            // of the two particles.
                            const auto tcoords_begin_i = sd_i.tcoords.begin();
                            const auto tcoords_end_i = sd_i.tcoords.end();

                            const auto tcoords_begin_j = sd_j.tcoords.begin();
                            const auto tcoords_end_j = sd_j.tcoords.end();

                            // Determine, for both particles, the end of the first
                            // substep that ends after the beginning of the chunk
                            auto ss_it_i = std::upper_bound(tcoords_begin_i, tcoords_end_i, chunk_begin);
                            auto ss_it_j = std::upper_bound(tcoords_begin_j, tcoords_end_j, chunk_begin);

                            // NOTE: I am not 100% sure this is necessary, as there should be at least one
                            // substep that ends after the beginning of the chunk, but with this being FP,
                            // better safe than sorry.
                            ss_it_i -= (ss_it_i == tcoords_end_i);
                            ss_it_j -= (ss_it_j == tcoords_end_j);

                            // Determine the starting times of the substeps ending at ss_it_i/j.
                            // NOTE: these are times measured relative to the beginning of the
                            // superstep.
                            const auto start_i
                                = (ss_it_i == tcoords_begin_i) ? hy::detail::dfloat<double>(0) : *(ss_it_i - 1);
                            const auto start_j
                                = (ss_it_j == tcoords_begin_j) ? hy::detail::dfloat<double>(0) : *(ss_it_j - 1);

                            // The Taylor polynomials for the two particles are time polynomials
                            // in which time is counted from the beginning of the substep. In order
                            // to compute the state at the beginning of the chunk, we need to
                            // compute the time coordinate of the beginning of the chunk relative
                            // to the beginning of the substep.
                            const auto tm_i = static_cast<double>(chunk_begin - start_i);
                            const auto tm_j = static_cast<double>(chunk_begin - start_j);

                            // Fetch pointers to the original Taylor polynomials for the two particles
                            // for the substeps identified above. These are the substeps in which we use
                            // dense output to compute the state of the particles at the beginning of the
                            // chunk.
                            // NOTE: static_cast because:
                            // - we have verified during the propagation that we can safely compute
                            //   differences between iterators of tcoords vectors (see overflow checking in the
                            //   step() function), and
                            // - we know that there are multiple Taylor coefficients being recorded
                            //   for each time coordinate, thus the size type of the vector of Taylor
                            //   coefficients can certainly represent the size of the tcoords vectors.
                            const auto ss_idx_i
                                = static_cast<decltype(s_data[pi].tc_x.size())>(ss_it_i - tcoords_begin_i);
                            const auto ss_idx_j
                                = static_cast<decltype(s_data[pj].tc_x.size())>(ss_it_j - tcoords_begin_j);

                            const auto *poly_xi = s_data[pi].tc_x.data() + ss_idx_i * (order + 1u);
                            const auto *poly_yi = s_data[pi].tc_y.data() + ss_idx_i * (order + 1u);
                            const auto *poly_zi = s_data[pi].tc_z.data() + ss_idx_i * (order + 1u);
                            const auto *poly_vxi = s_data[pi].tc_vx.data() + ss_idx_i * (order + 1u);
                            const auto *poly_vyi = s_data[pi].tc_vy.data() + ss_idx_i * (order + 1u);
                            const auto *poly_vzi = s_data[pi].tc_vz.data() + ss_idx_i * (order + 1u);

                            const auto *poly_xj = s_data[pj].tc_x.data() + ss_idx_j * (order + 1u);
                            const auto *poly_yj = s_data[pj].tc_y.data() + ss_idx_j * (order + 1u);
                            const auto *poly_zj = s_data[pj].tc_z.data() + ss_idx_j * (order + 1u);
                            const auto *poly_vxj = s_data[pj].tc_vx.data() + ss_idx_j * (order + 1u);
                            const auto *poly_vyj = s_data[pj].tc_vy.data() + ss_idx_j * (order + 1u);
                            const auto *poly_vzj = s_data[pj].tc_vz.data() + ss_idx_j * (order + 1u);

                            // Determine the initial conditions for the two particles via
                            // dense output.
                            st(0, bidx) = detail::poly_eval(poly_xi, tm_i, order);
                            st(1, bidx) = detail::poly_eval(poly_yi, tm_i, order);
                            st(2, bidx) = detail::poly_eval(poly_zi, tm_i, order);
                            st(3, bidx) = detail::poly_eval(poly_vxi, tm_i, order);
                            st(4, bidx) = detail::poly_eval(poly_vyi, tm_i, order);
                            st(5, bidx) = detail::poly_eval(poly_vzi, tm_i, order);

                            st(6, bidx) = detail::poly_eval(poly_xj, tm_j, order);
                            st(7, bidx) = detail::poly_eval(poly_yj, tm_j, order);
                            st(8, bidx) = detail::poly_eval(poly_zj, tm_j, order);
                            st(9, bidx) = detail::poly_eval(poly_vxj, tm_j, order);
                            st(10, bidx) = detail::poly_eval(poly_vyj, tm_j, order);
                            st(11, bidx) = detail::poly_eval(poly_vzj, tm_j, order);

                            // Copy over the pars.
                            for (std::uint32_t par_idx = 0; par_idx < npars; ++par_idx) {
                                pt(par_idx, bidx) = pv(pi, par_idx);
                                pt(par_idx + npars + 1u, bidx) = pv(pj, par_idx);
                            }

                            // Load the particle radiuses.
                            // NOTE: for each particle, the radius is appended
                            // as an extra parameter on top of the parameters
                            // appearing in the dynamics.
                            pt(npars, bidx) = sv(pi, 6);
                            pt(2u * npars + 1u, bidx) = sv(pj, 6);
                        }

                        // Setup the time coordinate.
                        const auto tm_coord = m_data->time + chunk_begin;
                        ta.set_dtime(tm_coord.hi, tm_coord.lo);

                        // Run the integration.
                        ta.propagate_for(c_end - c_begin);

                        // Examine the outcomes.
                        for (std::uint32_t bidx = 0; bidx < batch_size; ++bidx) {
                            const auto oc = std::get<0>(ta.get_propagate_res()[bidx]);

                            if (oc != hy::taylor_outcome::time_limit) {
                                // NOTE: collision event index hard-coded
                                // to zero here.
                                if (oc == hy::taylor_outcome{-1}) {
                                    // Fetch the particles' indices.
                                    // NOTE: same code as above.
                                    auto bpc_idx = ridx * batch_size + bidx;
                                    bpc_idx = (bpc_idx < bpc_size) ? bpc_idx : (bpc_size - 1u);
                                    const auto [pi, pj] = bpc[bpc_idx];

                                    // Record the collision.
                                    // NOTE: we may end up recording the same collision multiple times,
                                    // if it happens to the repeated candidates at the end of the last
                                    // group. This is fine: repeated items in coll_vec do not
                                    // are inconsequential when we are looking at the earliest collision,
                                    // and the duplicate collisions will just be ignored.
                                    auto dtm = ta.get_dtime();
                                    auto tcoll = dfloat(dtm.first[bidx], dtm.second[bidx]);
                                    m_data->coll_vec.emplace_back(pi, pj, static_cast<double>(tcoll - m_data->time));
                                } else if (oc == hy::taylor_outcome::err_nf_state) {
                                    throw std::runtime_error(fmt::format(
                                        "An invalid outcome of {} was returned when integrating the trajectories of a "
                                        "pair of particles during narrow phase collision detection",
                                        oc));
                                } else {
                                    // NOTE: at this time the only possible outcome at this point
                                    // is success, as we don't have any events other than the collision
                                    // one and there is no cb involved. If we allow for generic user-defined
                                    // events, then we need to complicate the logic a bit here.
                                    assert(oc == hy::taylor_outcome::success);
                                }
                            }
                        }
                    }

                    // Put the integrator back into the cache.
                    c_ta_cache.push(std::move(ta_ptr));
                });
        }
    });

    logger->trace("Narrow phase collision detection time: {}s", sw);
    logger->trace("Total number of collisions detected: {}", m_data->coll_vec.size());
}

} // namespace cascade
