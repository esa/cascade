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
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>

#include <boost/numeric/conversion/cast.hpp>

#include <spdlog/stopwatch.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>
#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/parallel_sort.h>

#include <heyoka/detail/dfloat.hpp>
#include <heyoka/taylor.hpp>

#include <cascade/detail/atomic_utils.hpp>
#include <cascade/detail/logging_impl.hpp>
#include <cascade/detail/sim_data.hpp>
#include <cascade/sim.hpp>

#if defined(__clang__) || defined(__GNUC__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"

#endif

#include <cascade/detail/mortonND_LUT.h>

#if defined(__clang__) || defined(__GNUC__)

#pragma GCC diagnostic pop

#endif

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

// Minimal interval class supporting a couple
// of elementary operations.
struct ival {
    double lower;
    double upper;

    ival() : ival(0) {}
    explicit ival(double val) : ival(val, val) {}
    explicit ival(double l, double u) : lower(l), upper(u) {}
};

// NOTE: see https://en.wikipedia.org/wiki/Interval_arithmetic.
ival operator+(ival a, ival b)
{
    return ival(a.lower + b.lower, a.upper + b.upper);
}

ival operator*(ival a, ival b)
{
    const auto tmp1 = a.lower * b.lower;
    const auto tmp2 = a.lower * b.upper;
    const auto tmp3 = a.upper * b.lower;
    const auto tmp4 = a.upper * b.upper;

    const auto l = std::min(std::min(tmp1, tmp2), std::min(tmp3, tmp4));
    const auto u = std::max(std::max(tmp1, tmp2), std::max(tmp3, tmp4));

    return ival(l, u);
}

// Quantise a value x in [min, max) into one of 2**16
// discrete slots, numbered from 0 to 2**16 - 1.
// NOTE: before invoking this function we must ensure that:
// - all args are finite,
// - max > min,
// - max - min gives a finite result.
// We don't check via assertion that x is in [min, max), because
// conceivably in some corner cases FP computations necessary to
// calculate x outside this function could lead to a value slightly outside
// the allowed range. In such case, we will clamp the result.
std::uint64_t disc_single_coord(float x, float min, float max)
{
    assert(std::isfinite(min));
    assert(std::isfinite(max));
    assert(std::isfinite(x));
    assert(max > min);
    assert(std::isfinite(max - min));

    // Determine the interval size.
    const auto isize = max - min;

    // Translate and rescale x so that min becomes zero
    // and max becomes 1.
    auto rx = (x - min) / isize;

    // Ensure that rx is not negative.
    // NOTE: if rx is NaN, this will set rx to zero.
    rx = rx >= 0.f ? rx : 0.f;

    // Rescale by 2**16.
    rx *= std::uint64_t(1) << 16;

    // Cast back to integer.
    const auto retval = static_cast<std::uint64_t>(rx);

    // Make sure to clamp it before returning, in case
    // somehow FP arithmetic makes it spill outside
    // the bound.
    // TODO min usage?
    return std::min(retval, std::uint64_t((std::uint64_t(1) << 16) - 1u));
}

} // namespace

} // namespace detail

// Perform the Morton encoding of the centres of the AABBs of the particles
// and sort the AABB data according to the codes.
void sim::morton_encode_sort()
{
    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    // Fetch the number of particles and chunks from m_data.
    const auto nparts = get_nparts();
    const auto nchunks = static_cast<unsigned>(m_data->global_lb.size());

    constexpr auto morton_enc = mortonnd::MortonNDLutEncoder<4, 16, 8>();

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(0u, nchunks), [&](const auto &range) {
        for (auto chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
            // TODO:
            // - bump up UB in order to ensure it's always > lb, as requested by the spatial
            //   discretisation function;
            // - check finiteness and the disc_single_coord requirements, before adjusting;
            // TODO: run a second check here to verify the new upper bound is not +inf.
            auto &glb = m_data->global_lb[chunk_idx];
            auto &gub = m_data->global_ub[chunk_idx];

            // Computation of the Morton codes.
            const auto offset = nparts * chunk_idx;

            auto *CASCADE_RESTRICT x_lb_ptr = m_data->x_lb.data() + offset;
            auto *CASCADE_RESTRICT y_lb_ptr = m_data->y_lb.data() + offset;
            auto *CASCADE_RESTRICT z_lb_ptr = m_data->z_lb.data() + offset;
            auto *CASCADE_RESTRICT r_lb_ptr = m_data->r_lb.data() + offset;

            auto *CASCADE_RESTRICT x_ub_ptr = m_data->x_ub.data() + offset;
            auto *CASCADE_RESTRICT y_ub_ptr = m_data->y_ub.data() + offset;
            auto *CASCADE_RESTRICT z_ub_ptr = m_data->z_ub.data() + offset;
            auto *CASCADE_RESTRICT r_ub_ptr = m_data->r_ub.data() + offset;

            auto *CASCADE_RESTRICT mcodes_ptr = m_data->mcodes.data() + offset;

            auto *CASCADE_RESTRICT srt_x_lb_ptr = m_data->srt_x_lb.data() + offset;
            auto *CASCADE_RESTRICT srt_y_lb_ptr = m_data->srt_y_lb.data() + offset;
            auto *CASCADE_RESTRICT srt_z_lb_ptr = m_data->srt_z_lb.data() + offset;
            auto *CASCADE_RESTRICT srt_r_lb_ptr = m_data->srt_r_lb.data() + offset;

            auto *CASCADE_RESTRICT srt_x_ub_ptr = m_data->srt_x_ub.data() + offset;
            auto *CASCADE_RESTRICT srt_y_ub_ptr = m_data->srt_y_ub.data() + offset;
            auto *CASCADE_RESTRICT srt_z_ub_ptr = m_data->srt_z_ub.data() + offset;
            auto *CASCADE_RESTRICT srt_r_ub_ptr = m_data->srt_r_ub.data() + offset;

            auto *CASCADE_RESTRICT srt_mcodes_ptr = m_data->srt_mcodes.data() + offset;

            auto *CASCADE_RESTRICT vidx_ptr = m_data->vidx.data() + offset;

            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(0, nparts), [&](const auto &r2) {
                // NOTE: JIT optimisation opportunity here. Worth it?
                for (auto pidx = r2.begin(); pidx != r2.end(); ++pidx) {
                    // Compute the centre of the AABB.
                    const auto x_ctr = x_lb_ptr[pidx] / 2 + x_ub_ptr[pidx] / 2;
                    const auto y_ctr = y_lb_ptr[pidx] / 2 + y_ub_ptr[pidx] / 2;
                    const auto z_ctr = z_lb_ptr[pidx] / 2 + z_ub_ptr[pidx] / 2;
                    const auto r_ctr = r_lb_ptr[pidx] / 2 + r_ub_ptr[pidx] / 2;

                    const auto n0 = detail::disc_single_coord(x_ctr, glb[0], gub[0]);
                    const auto n1 = detail::disc_single_coord(y_ctr, glb[1], gub[1]);
                    const auto n2 = detail::disc_single_coord(z_ctr, glb[2], gub[2]);
                    const auto n3 = detail::disc_single_coord(r_ctr, glb[3], gub[3]);

                    mcodes_ptr[pidx] = morton_enc.Encode(n0, n1, n2, n3);
                }
            });

            // Indirect sorting of the indices for the current chunk
            // according to the Morton codes.
            oneapi::tbb::parallel_sort(vidx_ptr, vidx_ptr + nparts, [mcodes_ptr](auto idx1, auto idx2) {
                return mcodes_ptr[idx1] < mcodes_ptr[idx2];
            });

            // Helper to apply the indirect sorting defined in vidx to the data in src.
            // The sorted data will be written into out.
            auto isort_apply = [vidx_ptr, nparts](auto *out, const auto *src) {
                oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(0, nparts), [&](const auto &rn) {
                    for (auto i = rn.begin(); i != rn.end(); ++i) {
                        out[i] = src[vidx_ptr[i]];
                    }
                });
            };

            // NOTE: can do all of these in parallel in principle, but performance
            // is bottlenecked by RAM speed anyway. Perhaps revisit on machines
            // with larger core counts during performance tuning.
            isort_apply(srt_x_lb_ptr, x_lb_ptr);
            isort_apply(srt_y_lb_ptr, y_lb_ptr);
            isort_apply(srt_z_lb_ptr, z_lb_ptr);
            isort_apply(srt_r_lb_ptr, r_lb_ptr);

            isort_apply(srt_x_ub_ptr, x_ub_ptr);
            isort_apply(srt_y_ub_ptr, y_ub_ptr);
            isort_apply(srt_z_ub_ptr, z_ub_ptr);
            isort_apply(srt_r_ub_ptr, r_ub_ptr);

            isort_apply(srt_mcodes_ptr, mcodes_ptr);
        }
    });

    logger->trace("Morton encoding and sorting time: {}s", sw);
}

// TODO clarify behaviour in case of exceptions.
void sim::propagate_for(double t)
{
    namespace hy = heyoka;

    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    // Cache a few quantities.
    const auto batch_size = m_data->b_ta.get_batch_size();
    const auto nparts = get_nparts();
    const auto order = m_data->s_ta.get_order();
    // Number of regular batches.
    const auto n_batches = nparts / batch_size;
    // Do we have events in the numerical integration?
    const auto with_events = m_data->s_ta.with_events();
    // The time coordinate at the beginning of
    // the superstep.
    const auto init_time = m_data->time;

    // Superstep size and number of chunks.
    // TODO fix.
    const auto delta_t = 0.46 * 4u;
    // TODO enforce power of 2?
    const auto nchunks = 8u;

    const auto chunk_size = delta_t / nchunks;

    // Ensure the vectors in m_data are set up with the correct sizes.
    m_data->s_data.resize(boost::numeric_cast<decltype(m_data->s_data.size())>(nparts));
    // TODO overflow checks/numeric cast.
    m_data->x_lb.resize(nparts * nchunks);
    m_data->y_lb.resize(nparts * nchunks);
    m_data->z_lb.resize(nparts * nchunks);
    m_data->r_lb.resize(nparts * nchunks);
    m_data->x_ub.resize(nparts * nchunks);
    m_data->y_ub.resize(nparts * nchunks);
    m_data->z_ub.resize(nparts * nchunks);
    m_data->r_ub.resize(nparts * nchunks);
    m_data->mcodes.resize(nparts * nchunks);
    m_data->vidx.resize(nparts * nchunks);
    m_data->srt_x_lb.resize(nparts * nchunks);
    m_data->srt_y_lb.resize(nparts * nchunks);
    m_data->srt_z_lb.resize(nparts * nchunks);
    m_data->srt_r_lb.resize(nparts * nchunks);
    m_data->srt_x_ub.resize(nparts * nchunks);
    m_data->srt_y_ub.resize(nparts * nchunks);
    m_data->srt_z_ub.resize(nparts * nchunks);
    m_data->srt_r_ub.resize(nparts * nchunks);
    m_data->srt_mcodes.resize(nparts * nchunks);

    // The vectors containing the state at the
    // end of a superstep.
    m_data->final_x.resize(nparts);
    m_data->final_y.resize(nparts);
    m_data->final_z.resize(nparts);
    m_data->final_vx.resize(nparts);
    m_data->final_vy.resize(nparts);
    m_data->final_vz.resize(nparts);

    constexpr auto finf = std::numeric_limits<float>::infinity();

    // Setup the global lb/ub for each chunk.
    // TODO numeric casts.
    m_data->global_lb.resize(nchunks);
    m_data->global_ub.resize(nchunks);
    // NOTE: the global AABBs need to be set up with
    // initial values.
    std::ranges::fill(m_data->global_lb, std::array{finf, finf, finf, finf});
    std::ranges::fill(m_data->global_ub, std::array{-finf, -finf, -finf, -finf});

    // Setup the BVH data.
    // TODO numeric cast.
    m_data->bvh_trees.resize(nchunks);
    m_data->nc_buffer.resize(nchunks);
    m_data->ps_buffer.resize(nchunks);
    m_data->nplc_buffer.resize(nchunks);

    // Setup the broad phase collision detection data.
    // TODO numeric cast.
    m_data->bp_coll.resize(nchunks);
    m_data->bp_caches.resize(nchunks);
    m_data->stack_caches.resize(nchunks);

    // Setup the narrow phase collision detection data.
    // TODO numeric cast.
    m_data->np_caches.resize(nchunks);

    logger->trace("Initial buffer setup time: {}s", sw);

    logger->debug("Inferred superstep size: {}", infer_superstep());

    std::atomic<bool> int_error{false};

    // Numerical integration and computation of the AABBs in batch mode.
    auto batch_int_aabb = [&](const auto &range) {
        // Fetch an integrator from the cache, or create it.
        std::unique_ptr<hy::taylor_adaptive_batch<double>> ta_ptr;

        if (!m_data->b_ta_cache.try_pop(ta_ptr)) {
            logger->debug("Creating new batch integrator");

            ta_ptr = std::make_unique<hy::taylor_adaptive_batch<double>>(m_data->b_ta);
        }

        // Cache a few variables.
        auto &ta = *ta_ptr;
        auto *st_data = ta.get_state_data();
        auto &s_data = m_data->s_data;
        const auto &ta_tc = ta.get_tc();

        // The first step is the numerical integration for all particles
        // in range throughout the entire superstep.
        for (auto idx = range.begin(); idx != range.end(); ++idx) {
            // Particle indices corresponding to the current batch.
            const auto pidx_begin = idx * batch_size;
            const auto pidx_end = pidx_begin + batch_size;

            // Clear up the Taylor coefficients and the times
            // of the substeps for the current particle.
            for (auto i = pidx_begin; i < pidx_end; ++i) {
                s_data[i].tc_x.clear();
                s_data[i].tc_y.clear();
                s_data[i].tc_z.clear();
                s_data[i].tc_r.clear();

                s_data[i].tcoords.clear();
            }

            // Reset cooldowns and set up the times.
            if (with_events) {
                ta.reset_cooldowns();
            }
            ta.set_dtime(init_time.hi, init_time.lo);

            // Copy over the state.
            // NOTE: would need to take care of synching up the
            // runtime parameters too.
            std::copy(m_x.data() + pidx_begin, m_x.data() + pidx_end, st_data);
            std::copy(m_y.data() + pidx_begin, m_y.data() + pidx_end, st_data + batch_size);
            std::copy(m_z.data() + pidx_begin, m_z.data() + pidx_end, st_data + 2u * batch_size);

            std::copy(m_vx.data() + pidx_begin, m_vx.data() + pidx_end, st_data + 3u * batch_size);
            std::copy(m_vy.data() + pidx_begin, m_vy.data() + pidx_end, st_data + 4u * batch_size);
            std::copy(m_vz.data() + pidx_begin, m_vz.data() + pidx_end, st_data + 5u * batch_size);

            // NOTE: compute the radius on the fly from the x/y/z coords.
            for (std::uint32_t i = 0; i < batch_size; ++i) {
                st_data[6u * batch_size + i]
                    = std::sqrt(st_data[i] * st_data[i] + st_data[batch_size + i] * st_data[batch_size + i]
                                + st_data[batch_size * 2u + i] * st_data[batch_size * 2u + i]);
            }

            // Setup the propagate_for() callback.
            auto cb = [&](auto &) {
                for (std::uint32_t i = 0; i < batch_size; ++i) {
                    if (ta.get_last_h()[i] == 0.) {
                        // Ignore this batch index if the last
                        // timestep was zero.
                        continue;
                    }

                    // Record the time coordinate at the end of the step, relative
                    // to the initial time.
                    const auto time_f = hy::detail::dfloat<double>(ta.get_dtime().first[i], ta.get_dtime().second[i]);
                    s_data[pidx_begin + i].tcoords.push_back(time_f - init_time);
                    if (!isfinite(s_data[pidx_begin + i].tcoords.back())) {
                        return false;
                    }

                    // Copy over the Taylor coefficients.
                    // TODO resize + copy, instead of push back? In such
                    // a case, we should probably use the no init allocator
                    // for the tc vectors.
                    for (std::uint32_t o = 0; o <= order; ++o) {
                        s_data[pidx_begin + i].tc_x.push_back(ta_tc[o * batch_size + i]);
                        s_data[pidx_begin + i].tc_y.push_back(ta_tc[(order + 1u) * batch_size + o * batch_size + i]);
                        s_data[pidx_begin + i].tc_z.push_back(
                            ta_tc[2u * (order + 1u) * batch_size + o * batch_size + i]);
                        s_data[pidx_begin + i].tc_r.push_back(
                            ta_tc[6u * (order + 1u) * batch_size + o * batch_size + i]);
                    }
                }

                return true;
            };
            std::function<bool(hy::taylor_adaptive_batch<double> &)> cbf(std::cref(cb));

            // Integrate.
            ta.propagate_for(delta_t, hy::kw::write_tc = true, hy::kw::callback = cbf);

            // Check for errors.
            if (std::ranges::any_of(ta.get_propagate_res(), [](const auto &tup) {
                    return std::get<0>(tup) != hy::taylor_outcome::time_limit;
                })) {
                // TODO distinguish various error codes?
                int_error.store(true, std::memory_order_relaxed);

                // TODO return instead of break here?
                break;
            }

            // Fill in the state at the end of the superstep.
            std::copy(st_data, st_data + batch_size, m_data->final_x.data() + pidx_begin);
            std::copy(st_data + batch_size, st_data + 2u * batch_size, m_data->final_y.data() + pidx_begin);
            std::copy(st_data + 2u * batch_size, st_data + 3u * batch_size, m_data->final_z.data() + pidx_begin);
            std::copy(st_data + 3u * batch_size, st_data + 4u * batch_size, m_data->final_vx.data() + pidx_begin);
            std::copy(st_data + 4u * batch_size, st_data + 5u * batch_size, m_data->final_vy.data() + pidx_begin);
            std::copy(st_data + 5u * batch_size, st_data + 6u * batch_size, m_data->final_vz.data() + pidx_begin);
        }

        // We can now proceed, for each chunk, to:
        // - compute the bounding boxes of the trajectories of all particles
        //   in range,
        // - update the global bounding box.
        for (auto chunk_idx = 0u; chunk_idx < nchunks; ++chunk_idx) {
            // The global bounding box for the current chunk.
            auto &glb = m_data->global_lb[chunk_idx];
            auto &gub = m_data->global_ub[chunk_idx];

            // Chunk-specific bounding box for the current particle range.
            // This will eventually be used to update the global bounding box.
            auto local_lb = std::array{finf, finf, finf, finf};
            auto local_ub = std::array{-finf, -finf, -finf, -finf};

            const auto offset = nparts * chunk_idx;

            auto *CASCADE_RESTRICT x_lb_ptr = m_data->x_lb.data() + offset;
            auto *CASCADE_RESTRICT y_lb_ptr = m_data->y_lb.data() + offset;
            auto *CASCADE_RESTRICT z_lb_ptr = m_data->z_lb.data() + offset;
            auto *CASCADE_RESTRICT r_lb_ptr = m_data->r_lb.data() + offset;

            auto *CASCADE_RESTRICT x_ub_ptr = m_data->x_ub.data() + offset;
            auto *CASCADE_RESTRICT y_ub_ptr = m_data->y_ub.data() + offset;
            auto *CASCADE_RESTRICT z_ub_ptr = m_data->z_ub.data() + offset;
            auto *CASCADE_RESTRICT r_ub_ptr = m_data->r_ub.data() + offset;

            // The time coordinate, relative to init_time, of
            // the chunk's begin/end.
            const auto chunk_begin = hy::detail::dfloat<double>(chunk_size * chunk_idx);
            const auto chunk_end = hy::detail::dfloat<double>(chunk_size * (chunk_idx + 1u));

            for (auto idx = range.begin(); idx != range.end(); ++idx) {
                // Particle indices corresponding to the current batch.
                const auto pidx_begin = idx * batch_size;

                for (std::uint32_t i = 0; i < batch_size; ++i) {
                    // Setup the initial values for the bounding box
                    // of the current particle in the current chunk.
                    x_lb_ptr[pidx_begin + i] = finf;
                    y_lb_ptr[pidx_begin + i] = finf;
                    z_lb_ptr[pidx_begin + i] = finf;
                    r_lb_ptr[pidx_begin + i] = finf;

                    x_ub_ptr[pidx_begin + i] = -finf;
                    y_ub_ptr[pidx_begin + i] = -finf;
                    z_ub_ptr[pidx_begin + i] = -finf;
                    r_ub_ptr[pidx_begin + i] = -finf;

                    // Fetch the particle radius.
                    const auto p_radius = m_sizes[pidx_begin + i];

                    // Cache the range of end times of the substeps.
                    const auto &tcoords = s_data[pidx_begin + i].tcoords;
                    const auto tcoords_begin = tcoords.begin();
                    const auto tcoords_end = tcoords.end();

                    // We need to locate the substep range that fully includes
                    // the current chunk.
                    // First we locate the first substep whose end is strictly
                    // *greater* than the lower bound of the chunk.
                    const auto ss_it_begin = std::upper_bound(tcoords_begin, tcoords_end, chunk_begin);
                    // Then, we locate the first substep whose end is *greater than or
                    // equal to* the end of the chunk.
                    auto ss_it_end = std::lower_bound(ss_it_begin, tcoords_end, chunk_end);
                    // Bump it up by one to define a half-open range.
                    // NOTE: don't bump it if it is already at the end.
                    // This could happen at the last chunk due to FP rounding.
                    ss_it_end += (ss_it_end != tcoords_end);

                    // Iterate over all substeps and update the bounding box
                    // for the current particle.
                    for (auto it = ss_it_begin; it != ss_it_end; ++it) {
                        // it points to the end of a substep which overlaps
                        // with the current chunk. The size of the polynomial evaluation
                        // interval is the size of the intersection between the substep and
                        // the chunk.

                        // Determine the initial time coordinate of the substep, relative
                        // to init_time. If it is tcoords_begin, ss_start will be zero, otherwise
                        // ss_start is given by the iterator preceding it.
                        const auto ss_start = (it == tcoords_begin) ? hy::detail::dfloat<double>(0) : *(it - 1);

                        // Determine lower/upper bounds of the evaluation interval,
                        // relative to init_time.
                        // TODO std::min/max here?
                        const auto ev_lb = std::max(chunk_begin, ss_start);
                        const auto ev_ub = std::min(chunk_end, *it);

                        // Create the actual evaluation interval, referring
                        // it to the beginning of the substep.
                        const auto h_int_lb = static_cast<double>(ev_lb - ss_start);
                        const auto h_int_ub = static_cast<double>(ev_ub - ss_start);

                        // Determine the index of the substep within the chunk.
                        // TODO: overflow check -> tcoords' size must fit in the
                        // iterator difference type.
                        const auto ss_idx
                            = boost::numeric_cast<decltype(s_data[pidx_begin + i].tc_x.size())>(it - tcoords_begin);

                        // Compute the pointers to the TCs for the current particle
                        // and substep.
                        const auto tc_ptr_x = s_data[pidx_begin + i].tc_x.data() + ss_idx * (order + 1u);
                        const auto tc_ptr_y = s_data[pidx_begin + i].tc_y.data() + ss_idx * (order + 1u);
                        const auto tc_ptr_z = s_data[pidx_begin + i].tc_z.data() + ss_idx * (order + 1u);
                        const auto tc_ptr_r = s_data[pidx_begin + i].tc_r.data() + ss_idx * (order + 1u);

                        // Run the polynomial evaluations using interval arithmetic.
                        // TODO jit for performance? If so, we can do all 4 coordinates
                        // in a single JIT compiled function. Possibly also the update
                        // with the particle radius?
                        auto horner_eval = [order, h_int = detail::ival(h_int_lb, h_int_ub)](const double *ptr) {
                            auto acc = detail::ival(ptr[order]);
                            for (auto o = 1u; o <= order; ++o) {
                                acc = detail::ival(ptr[order - o]) + acc * h_int;
                            }

                            return acc;
                        };

                        auto x_int = horner_eval(tc_ptr_x);
                        auto y_int = horner_eval(tc_ptr_y);
                        auto z_int = horner_eval(tc_ptr_z);
                        auto r_int = horner_eval(tc_ptr_r);

                        // Adjust the intervals accounting for the particle radius.
                        x_int.lower -= p_radius;
                        x_int.upper += p_radius;

                        y_int.lower -= p_radius;
                        y_int.upper += p_radius;

                        z_int.lower -= p_radius;
                        z_int.upper += p_radius;

                        r_int.lower -= p_radius;
                        r_int.upper += p_radius;

                        // A couple of helpers to cast lower/upper bounds from double to float. After
                        // the cast, we will also move slightly the bounds to add a safety margin to account
                        // for possible truncation in the conversion.
                        // TODO: this looks like a good place for inf checking.
                        auto lb_make_float = [&](double lb) { return std::nextafter(static_cast<float>(lb), -finf); };
                        auto ub_make_float = [&](double ub) { return std::nextafter(static_cast<float>(ub), finf); };

                        // Update the bounding box for the current particle.
                        // TODO: min/max usage?
                        // TODO: inf checking? Here or when updating the global AABB?
                        x_lb_ptr[pidx_begin + i] = std::min(x_lb_ptr[pidx_begin + i], lb_make_float(x_int.lower));
                        y_lb_ptr[pidx_begin + i] = std::min(y_lb_ptr[pidx_begin + i], lb_make_float(y_int.lower));
                        z_lb_ptr[pidx_begin + i] = std::min(z_lb_ptr[pidx_begin + i], lb_make_float(z_int.lower));
                        r_lb_ptr[pidx_begin + i] = std::min(r_lb_ptr[pidx_begin + i], lb_make_float(r_int.lower));

                        x_ub_ptr[pidx_begin + i] = std::max(x_ub_ptr[pidx_begin + i], ub_make_float(x_int.upper));
                        y_ub_ptr[pidx_begin + i] = std::max(y_ub_ptr[pidx_begin + i], ub_make_float(y_int.upper));
                        z_ub_ptr[pidx_begin + i] = std::max(z_ub_ptr[pidx_begin + i], ub_make_float(z_int.upper));
                        r_ub_ptr[pidx_begin + i] = std::max(r_ub_ptr[pidx_begin + i], ub_make_float(r_int.upper));
                    }

                    // Update the local AABB with the bounding box for the current particle.
                    // TODO: min/max usage?
                    local_lb[0] = std::min(local_lb[0], x_lb_ptr[pidx_begin + i]);
                    local_lb[1] = std::min(local_lb[1], y_lb_ptr[pidx_begin + i]);
                    local_lb[2] = std::min(local_lb[2], z_lb_ptr[pidx_begin + i]);
                    local_lb[3] = std::min(local_lb[3], r_lb_ptr[pidx_begin + i]);

                    local_ub[0] = std::max(local_ub[0], x_lb_ptr[pidx_begin + i]);
                    local_ub[1] = std::max(local_ub[1], y_lb_ptr[pidx_begin + i]);
                    local_ub[2] = std::max(local_ub[2], z_lb_ptr[pidx_begin + i]);
                    local_ub[3] = std::max(local_ub[3], r_lb_ptr[pidx_begin + i]);
                }
            }

            // Atomically update the global AABB for the current chunk.
            detail::lb_atomic_update(glb[0], local_lb[0]);
            detail::lb_atomic_update(glb[1], local_lb[1]);
            detail::lb_atomic_update(glb[2], local_lb[2]);
            detail::lb_atomic_update(glb[3], local_lb[3]);

            detail::ub_atomic_update(gub[0], local_ub[0]);
            detail::ub_atomic_update(gub[1], local_ub[1]);
            detail::ub_atomic_update(gub[2], local_ub[2]);
            detail::ub_atomic_update(gub[3], local_ub[3]);
        }

        // Put the integrator (back) into the cache.
        m_data->b_ta_cache.push(std::move(ta_ptr));
    };

    // Numerical integration and computation of the AABBs for the scalar remainder.
    auto scalar_int_aabb = [&](const auto &range) {
        // Fetch an integrator from the cache, or create it.
        std::unique_ptr<hy::taylor_adaptive<double>> ta_ptr;

        if (!m_data->s_ta_cache.try_pop(ta_ptr)) {
            logger->debug("Creating new integrator");

            ta_ptr = std::make_unique<hy::taylor_adaptive<double>>(m_data->s_ta);
        }

        // Cache a few variables.
        auto &ta = *ta_ptr;
        auto *st_data = ta.get_state_data();
        auto &s_data = m_data->s_data;
        const auto &ta_tc = ta.get_tc();

        // The first step is the numerical integration for all particles
        // in range throughout the entire superstep.
        for (auto pidx = range.begin(); pidx != range.end(); ++pidx) {
            // Clear up the Taylor coefficients and the times
            // of the substeps for the current particle.
            s_data[pidx].tc_x.clear();
            s_data[pidx].tc_y.clear();
            s_data[pidx].tc_z.clear();
            s_data[pidx].tc_r.clear();

            s_data[pidx].tcoords.clear();

            // Reset cooldowns and set up the times.
            if (with_events) {
                ta.reset_cooldowns();
            }
            ta.set_dtime(init_time.hi, init_time.lo);

            // Copy over the state.
            // NOTE: would need to take care of synching up the
            // runtime parameters too.
            st_data[0] = m_x[pidx];
            st_data[1] = m_y[pidx];
            st_data[2] = m_z[pidx];
            st_data[3] = m_vx[pidx];
            st_data[4] = m_vy[pidx];
            st_data[5] = m_vz[pidx];

            // NOTE: compute the radius on the fly from the x/y/z coords.
            st_data[6] = std::sqrt(st_data[0] * st_data[0] + st_data[1] * st_data[1] + st_data[2] * st_data[2]);

            // Setup the propagate_for() callback.
            auto cb = [&](auto &) {
                // Record the time coordinate at the end of the step, relative
                // to the initial time.
                const auto time_f = hy::detail::dfloat<double>(ta.get_dtime().first, ta.get_dtime().second);
                s_data[pidx].tcoords.push_back(time_f - init_time);
                if (!isfinite(s_data[pidx].tcoords.back())) {
                    return false;
                }

                // Copy over the Taylor coefficients.
                // TODO resize + copy, instead of push back? In such
                // a case, we should probably use the no init allocator
                // for the tc vectors.
                for (std::uint32_t o = 0; o <= order; ++o) {
                    s_data[pidx].tc_x.push_back(ta_tc[o]);
                    s_data[pidx].tc_y.push_back(ta_tc[order + 1u + o]);
                    s_data[pidx].tc_z.push_back(ta_tc[2u * (order + 1u) + o]);
                    s_data[pidx].tc_r.push_back(ta_tc[6u * (order + 1u) + o]);
                }

                return true;
            };
            std::function<bool(hy::taylor_adaptive<double> &)> cbf(std::cref(cb));

            // Integrate.
            const auto oc = std::get<0>(ta.propagate_for(delta_t, hy::kw::write_tc = true, hy::kw::callback = cbf));

            // Check for errors.
            if (oc != hy::taylor_outcome::time_limit) {
                // TODO distinguish various error codes?
                int_error.store(true, std::memory_order_relaxed);

                // TODO return instead of break here?
                break;
            }

            // Fill in the state at the end of the superstep.
            m_data->final_x[pidx] = st_data[0];
            m_data->final_y[pidx] = st_data[1];
            m_data->final_z[pidx] = st_data[2];
            m_data->final_vx[pidx] = st_data[3];
            m_data->final_vy[pidx] = st_data[4];
            m_data->final_vz[pidx] = st_data[5];
        }

        // We can now proceed, for each chunk, to:
        // - compute the bounding boxes of the trajectories of all particles
        //   in range,
        // - update the global bounding box.
        for (auto chunk_idx = 0u; chunk_idx < nchunks; ++chunk_idx) {
            // The global bounding box for the current chunk.
            auto &glb = m_data->global_lb[chunk_idx];
            auto &gub = m_data->global_ub[chunk_idx];

            // Chunk-specific bounding box for the current particle range.
            // This will eventually be used to update the global bounding box.
            auto local_lb = std::array{finf, finf, finf, finf};
            auto local_ub = std::array{-finf, -finf, -finf, -finf};

            const auto offset = nparts * chunk_idx;

            auto *CASCADE_RESTRICT x_lb_ptr = m_data->x_lb.data() + offset;
            auto *CASCADE_RESTRICT y_lb_ptr = m_data->y_lb.data() + offset;
            auto *CASCADE_RESTRICT z_lb_ptr = m_data->z_lb.data() + offset;
            auto *CASCADE_RESTRICT r_lb_ptr = m_data->r_lb.data() + offset;

            auto *CASCADE_RESTRICT x_ub_ptr = m_data->x_ub.data() + offset;
            auto *CASCADE_RESTRICT y_ub_ptr = m_data->y_ub.data() + offset;
            auto *CASCADE_RESTRICT z_ub_ptr = m_data->z_ub.data() + offset;
            auto *CASCADE_RESTRICT r_ub_ptr = m_data->r_ub.data() + offset;

            // The time coordinate, relative to init_time, of
            // the chunk's begin/end.
            const auto chunk_begin = hy::detail::dfloat<double>(chunk_size * chunk_idx);
            const auto chunk_end = hy::detail::dfloat<double>(chunk_size * (chunk_idx + 1u));

            for (auto pidx = range.begin(); pidx != range.end(); ++pidx) {
                // Setup the initial values for the bounding box
                // of the current particle in the current chunk.
                x_lb_ptr[pidx] = finf;
                y_lb_ptr[pidx] = finf;
                z_lb_ptr[pidx] = finf;
                r_lb_ptr[pidx] = finf;

                x_ub_ptr[pidx] = -finf;
                y_ub_ptr[pidx] = -finf;
                z_ub_ptr[pidx] = -finf;
                r_ub_ptr[pidx] = -finf;

                // Fetch the particle radius.
                const auto p_radius = m_sizes[pidx];

                // Cache the range of end times of the substeps.
                const auto &tcoords = s_data[pidx].tcoords;
                const auto tcoords_begin = tcoords.begin();
                const auto tcoords_end = tcoords.end();

                // We need to locate the substep range that fully includes
                // the current chunk.
                // First we locate the first substep whose end is strictly
                // *greater* than the lower bound of the chunk.
                const auto ss_it_begin = std::upper_bound(tcoords_begin, tcoords_end, chunk_begin);
                // Then, we locate the first substep whose end is *greater than or
                // equal to* the end of the chunk.
                auto ss_it_end = std::lower_bound(ss_it_begin, tcoords_end, chunk_end);
                // Bump it up by one to define a half-open range.
                // NOTE: don't bump it if it is already at the end.
                // This could happen at the last chunk due to FP rounding.
                ss_it_end += (ss_it_end != tcoords_end);

                // Iterate over all substeps and update the bounding box
                // for the current particle.
                for (auto it = ss_it_begin; it != ss_it_end; ++it) {
                    // it points to the end of a substep which overlaps
                    // with the current chunk. The size of the polynomial evaluation
                    // interval is the size of the intersection between the substep and
                    // the chunk.

                    // Determine the initial time coordinate of the substep, relative
                    // to init_time. If it is tcoords_begin, ss_start will be zero, otherwise
                    // ss_start is given by the iterator preceding it.
                    const auto ss_start = (it == tcoords_begin) ? hy::detail::dfloat<double>(0) : *(it - 1);

                    // Determine lower/upper bounds of the evaluation interval,
                    // relative to init_time.
                    // TODO std::min/max here?
                    const auto ev_lb = std::max(chunk_begin, ss_start);
                    const auto ev_ub = std::min(chunk_end, *it);

                    // Create the actual evaluation interval, referring
                    // it to the beginning of the substep.
                    const auto h_int_lb = static_cast<double>(ev_lb - ss_start);
                    const auto h_int_ub = static_cast<double>(ev_ub - ss_start);

                    // Determine the index of the substep within the chunk.
                    // TODO: overflow check -> tcoords' size must fit in the
                    // iterator difference type.
                    const auto ss_idx = boost::numeric_cast<decltype(s_data[pidx].tc_x.size())>(it - tcoords_begin);

                    // Compute the pointers to the TCs for the current particle
                    // and substep.
                    const auto tc_ptr_x = s_data[pidx].tc_x.data() + ss_idx * (order + 1u);
                    const auto tc_ptr_y = s_data[pidx].tc_y.data() + ss_idx * (order + 1u);
                    const auto tc_ptr_z = s_data[pidx].tc_z.data() + ss_idx * (order + 1u);
                    const auto tc_ptr_r = s_data[pidx].tc_r.data() + ss_idx * (order + 1u);

                    // Run the polynomial evaluations using interval arithmetic.
                    // TODO jit for performance? If so, we can do all 4 coordinates
                    // in a single JIT compiled function. Possibly also the update
                    // with the particle radius?
                    auto horner_eval = [order, h_int = detail::ival(h_int_lb, h_int_ub)](const double *ptr) {
                        auto acc = detail::ival(ptr[order]);
                        for (auto o = 1u; o <= order; ++o) {
                            acc = detail::ival(ptr[order - o]) + acc * h_int;
                        }

                        return acc;
                    };

                    auto x_int = horner_eval(tc_ptr_x);
                    auto y_int = horner_eval(tc_ptr_y);
                    auto z_int = horner_eval(tc_ptr_z);
                    auto r_int = horner_eval(tc_ptr_r);

                    // Adjust the intervals accounting for the particle radius.
                    x_int.lower -= p_radius;
                    x_int.upper += p_radius;

                    y_int.lower -= p_radius;
                    y_int.upper += p_radius;

                    z_int.lower -= p_radius;
                    z_int.upper += p_radius;

                    r_int.lower -= p_radius;
                    r_int.upper += p_radius;

                    // A couple of helpers to cast lower/upper bounds from double to float. After
                    // the cast, we will also move slightly the bounds to add a safety margin to account
                    // for possible truncation in the conversion.
                    // TODO: this looks like a good place for inf checking.
                    auto lb_make_float = [&](double lb) { return std::nextafter(static_cast<float>(lb), -finf); };
                    auto ub_make_float = [&](double ub) { return std::nextafter(static_cast<float>(ub), finf); };

                    // Update the bounding box for the current particle.
                    // TODO: min/max usage?
                    // TODO: inf checking? Here or when updating the global AABB?
                    x_lb_ptr[pidx] = std::min(x_lb_ptr[pidx], lb_make_float(x_int.lower));
                    y_lb_ptr[pidx] = std::min(y_lb_ptr[pidx], lb_make_float(y_int.lower));
                    z_lb_ptr[pidx] = std::min(z_lb_ptr[pidx], lb_make_float(z_int.lower));
                    r_lb_ptr[pidx] = std::min(r_lb_ptr[pidx], lb_make_float(r_int.lower));

                    x_ub_ptr[pidx] = std::max(x_ub_ptr[pidx], ub_make_float(x_int.upper));
                    y_ub_ptr[pidx] = std::max(y_ub_ptr[pidx], ub_make_float(y_int.upper));
                    z_ub_ptr[pidx] = std::max(z_ub_ptr[pidx], ub_make_float(z_int.upper));
                    r_ub_ptr[pidx] = std::max(r_ub_ptr[pidx], ub_make_float(r_int.upper));
                }

                // Update the local AABB with the bounding box for the current particle.
                // TODO: min/max usage?
                local_lb[0] = std::min(local_lb[0], x_lb_ptr[pidx]);
                local_lb[1] = std::min(local_lb[1], y_lb_ptr[pidx]);
                local_lb[2] = std::min(local_lb[2], z_lb_ptr[pidx]);
                local_lb[3] = std::min(local_lb[3], r_lb_ptr[pidx]);

                local_ub[0] = std::max(local_ub[0], x_lb_ptr[pidx]);
                local_ub[1] = std::max(local_ub[1], y_lb_ptr[pidx]);
                local_ub[2] = std::max(local_ub[2], z_lb_ptr[pidx]);
                local_ub[3] = std::max(local_ub[3], r_lb_ptr[pidx]);
            }

            // Atomically update the global AABB for the current chunk.
            detail::lb_atomic_update(glb[0], local_lb[0]);
            detail::lb_atomic_update(glb[1], local_lb[1]);
            detail::lb_atomic_update(glb[2], local_lb[2]);
            detail::lb_atomic_update(glb[3], local_lb[3]);

            detail::ub_atomic_update(gub[0], local_ub[0]);
            detail::ub_atomic_update(gub[1], local_ub[1]);
            detail::ub_atomic_update(gub[2], local_ub[2]);
            detail::ub_atomic_update(gub[3], local_ub[3]);
        }

        // Put the integrator (back) into the cache.
        m_data->s_ta_cache.push(std::move(ta_ptr));
    };

    // Do the integration concurrently with the initialisation of
    // the vector of indices for indirect sorting.
    oneapi::tbb::parallel_invoke(
        [&]() {
            auto &vidx = m_data->vidx;

            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(0u, nchunks), [&](const auto &range) {
                for (auto chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
                    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(0, nparts),
                                              [vidx_ptr = vidx.data() + nparts * chunk_idx](const auto &r2) {
                                                  for (auto i = r2.begin(); i != r2.end(); ++i) {
                                                      vidx_ptr[i] = i;
                                                  }
                                              });
                }
            });
        },
        [&]() { oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(0, n_batches), batch_int_aabb); },
        [&]() {
            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(n_batches * batch_size, nparts),
                                      scalar_int_aabb);
        });

    logger->trace("Propagation + AABB computation time: {}s", sw);

    if (int_error.load(std::memory_order_relaxed)) {
        // TODO
        throw;
    }

    // Computation of the Morton codes and sorting.
    morton_encode_sort();

    // Construction of the BVH trees.
    construct_bvh_trees();

    // Broad phase collision detection.
    broad_phase();

    // Record the original size of the global
    // collision vector before running narrow phase.
    const auto orig_cv_size = m_data->coll_vec.size();

    // Narrow phase collision detection.
    narrow_phase(chunk_size);

    // Sort the detected collisions by time.
    // NOTE: this can of course become a parallel sort if needed.
    std::sort(m_data->coll_vec.data() + orig_cv_size, m_data->coll_vec.data() + m_data->coll_vec.size(),
              [](const auto &tup1, const auto &tup2) { return std::get<2>(tup1) < std::get<2>(tup2); });

    logger->debug("Total number of collisions detected: {}", m_data->coll_vec.size() - orig_cv_size);

    logger->trace("Total propagation time: {}s", sw);
}

// Helper for the automatic determination
// of the superstep size.
double sim::infer_superstep()
{
    namespace hy = heyoka;

    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    // Cache a few quantities.
    const auto batch_size = m_data->b_ta.get_batch_size();
    const auto nparts = get_nparts();
    const auto n_batches = nparts / batch_size;
    const auto with_events = m_data->s_ta.with_events();
    const auto init_time = m_data->time;

    // For the superstep determination, we won't
    // iterate over all particles, but (roughly)
    // every 'stride' particles.
    constexpr auto stride = 10u;

    // Overflow check: we will need to iterate
    // possibly up to index nparts + stride.
    if (stride > std::numeric_limits<size_type>::max() - nparts) {
        throw std::overflow_error("Overflow detected during the automatic determination of the superstep size");
    }

    // Helper to perform the element-wise addition
    // of pairs.
    auto pair_plus = [](const auto &p1, const auto &p2) -> std::pair<double, size_type> {
        return {p1.first + p2.first, p1.second + p2.second};
    };

    // Global variables to compute the mean
    // dynamical timestep.
    double acc = 0;
    size_type n_part_acc = 0;

    // NOTE: as usual, run in parallel the batch and scalar computations.
    oneapi::tbb::parallel_invoke(
        [&]() {
            const auto batch_res = oneapi::tbb::parallel_deterministic_reduce(
                oneapi::tbb::blocked_range<size_type>(0, n_batches, 100), std::pair{0., size_type(0)},
                [&](const auto &range, auto partial_sum) {
                    // Fetch an integrator from the cache, or create it.
                    std::unique_ptr<hy::taylor_adaptive_batch<double>> ta_ptr;

                    if (!m_data->b_ta_cache.try_pop(ta_ptr)) {
                        logger->debug("Creating new batch integrator");

                        ta_ptr = std::make_unique<hy::taylor_adaptive_batch<double>>(m_data->b_ta);
                    }

                    // Cache a few variables.
                    auto &ta = *ta_ptr;
                    auto *st_data = ta.get_state_data();

                    for (auto idx = range.begin(); idx < range.end(); idx += stride) {
                        // Particle indices corresponding to the current batch.
                        const auto pidx_begin = idx * batch_size;
                        const auto pidx_end = pidx_begin + batch_size;

                        // Reset cooldowns and set up the times.
                        if (with_events) {
                            ta.reset_cooldowns();
                        }
                        ta.set_dtime(init_time.hi, init_time.lo);

                        // Copy over the state.
                        // NOTE: would need to take care of synching up the
                        // runtime parameters too.
                        std::copy(m_x.data() + pidx_begin, m_x.data() + pidx_end, st_data);
                        std::copy(m_y.data() + pidx_begin, m_y.data() + pidx_end, st_data + batch_size);
                        std::copy(m_z.data() + pidx_begin, m_z.data() + pidx_end, st_data + 2u * batch_size);

                        std::copy(m_vx.data() + pidx_begin, m_vx.data() + pidx_end, st_data + 3u * batch_size);
                        std::copy(m_vy.data() + pidx_begin, m_vy.data() + pidx_end, st_data + 4u * batch_size);
                        std::copy(m_vz.data() + pidx_begin, m_vz.data() + pidx_end, st_data + 5u * batch_size);

                        // NOTE: compute the radius on the fly from the x/y/z coords.
                        for (std::uint32_t i = 0; i < batch_size; ++i) {
                            st_data[6u * batch_size + i]
                                = std::sqrt(st_data[i] * st_data[i] + st_data[batch_size + i] * st_data[batch_size + i]
                                            + st_data[batch_size * 2u + i] * st_data[batch_size * 2u + i]);
                        }

                        // Integrate a single step.
                        ta.step();

                        // Check for errors and accumulate into partial sum.
                        for (const auto &[oc, h] : ta.get_step_res()) {
                            if (oc != hy::taylor_outcome::success) {
                                // TODO here we should distinguish the following cases:
                                // - nf_error (throw?),
                                // - stopped by event -> ignore for timestep determination
                                //   purposes.
                                throw;
                            }

                            partial_sum.first += h;
                            ++partial_sum.second;
                        }
                    }

                    // Put the integrator (back) into the cache.
                    m_data->b_ta_cache.push(std::move(ta_ptr));

                    return partial_sum;
                },
                pair_plus);

            // Update the global values.
            {
                std::atomic_ref acc_at(acc);
                acc_at.fetch_add(batch_res.first, std::memory_order::relaxed);
            }

            {
                std::atomic_ref n_part_acc_at(n_part_acc);
                n_part_acc_at.fetch_add(batch_res.second, std::memory_order::relaxed);
            }
        },
        [&]() {
            const auto scal_res = oneapi::tbb::parallel_deterministic_reduce(
                oneapi::tbb::blocked_range<size_type>(n_batches * batch_size, nparts, 100), std::pair{0., size_type(0)},
                [&](const auto &range, auto partial_sum) {
                    // Fetch an integrator from the cache, or create it.
                    std::unique_ptr<hy::taylor_adaptive<double>> ta_ptr;

                    if (!m_data->s_ta_cache.try_pop(ta_ptr)) {
                        logger->debug("Creating new integrator");

                        ta_ptr = std::make_unique<hy::taylor_adaptive<double>>(m_data->s_ta);
                    }

                    // Cache a few variables.
                    auto &ta = *ta_ptr;
                    auto *st_data = ta.get_state_data();

                    for (auto pidx = range.begin(); pidx < range.end(); pidx += stride) {
                        // Reset cooldowns and set up the times.
                        if (with_events) {
                            ta.reset_cooldowns();
                        }
                        ta.set_dtime(init_time.hi, init_time.lo);

                        // Copy over the state.
                        // NOTE: would need to take care of synching up the
                        // runtime parameters too.
                        st_data[0] = m_x[pidx];
                        st_data[1] = m_y[pidx];
                        st_data[2] = m_z[pidx];
                        st_data[3] = m_vx[pidx];
                        st_data[4] = m_vy[pidx];
                        st_data[5] = m_vz[pidx];

                        // NOTE: compute the radius on the fly from the x/y/z coords.
                        st_data[6]
                            = std::sqrt(st_data[0] * st_data[0] + st_data[1] * st_data[1] + st_data[2] * st_data[2]);

                        // Integrate for a single step
                        const auto [oc, h] = ta.step();

                        // Check for errors and accumulate into partial sum.
                        if (oc != hy::taylor_outcome::success) {
                            // TODO here we should distinguish the following cases:
                            // - nf_error (throw?),
                            // - stopped by event -> ignore for timestep determination
                            //   purposes.
                            throw;
                        }

                        partial_sum.first += h;
                        ++partial_sum.second;
                    }

                    // Put the integrator (back) into the cache.
                    m_data->s_ta_cache.push(std::move(ta_ptr));

                    return partial_sum;
                },
                pair_plus);

            // Update the global values.
            {
                std::atomic_ref acc_at(acc);
                acc_at.fetch_add(scal_res.first, std::memory_order::relaxed);
            }

            {
                std::atomic_ref n_part_acc_at(n_part_acc);
                n_part_acc_at.fetch_add(scal_res.second, std::memory_order::relaxed);
            }
        });

    // NOTE: this can happen only if the simulation has zero particles.
    if (n_part_acc == 0u) {
        throw std::invalid_argument(
            "Cannot automatically determine the superstep size if there are no particles in the simulation");
    }

    // Compute the final result: average step size multiplied by
    // a small constant.
    const auto res = acc / static_cast<double>(n_part_acc) * 3;

    if (!std::isfinite(res)) {
        throw std::invalid_argument("The automatic determination of the superstep size yielded a non-finite value");
    }

    if (res == 0) {
        throw std::invalid_argument("The automatic determination of the superstep size yielded a value of zero");
    }

    logger->trace("Timestep deduction time: {}s", sw);
    logger->debug("Number of particles considered for timestep deduction: {}", n_part_acc);

    return res;
}

} // namespace cascade
