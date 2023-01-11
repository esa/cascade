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
#include <cmath>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <tuple>
#include <utility>

#include <boost/numeric/conversion/cast.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>

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

// Minimal interval class supporting a couple
// of elementary operations.
// NOTE: like in heyoka, the implementation of interval arithmetic
// could be improved in 2 areas:
// - accounting for floating-point truncation to yield results
//   which are truly mathematically exact,
// - ensuring that min/max propagate nans, instead of potentially
//   ignoring them.
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
    rx *= static_cast<std::uint64_t>(1) << 16;

    // Cast back to integer.
    const auto retval = static_cast<std::uint64_t>(rx);

    // Make sure to clamp it before returning, in case
    // somehow FP arithmetic makes it spill outside
    // the bound.
    // NOTE: std::min is safe with integral types.
    return std::min(retval, static_cast<std::uint64_t>((static_cast<std::uint64_t>(1) << 16) - 1u));
}

} // namespace

} // namespace detail

// Setup the scalar integrator ta to integrate the trajectory for the particle
// at index pidx. Time and state are taken from the current values in the
// global sim data.
template <typename T>
void sim::init_scalar_ta(T &ta, size_type pidx) const
{
    namespace stdex = std::experimental;

    const auto nparts = get_nparts();
    const auto npars = get_npars();

    assert(pidx < nparts);

    // Create views on the data.
    auto *st_data = ta.get_state_data();
    stdex::mdspan sv(m_state->data(), stdex::extents<size_type, stdex::dynamic_extent, 7u>(nparts));
    stdex::mdspan pv(m_pars->data(), nparts, npars);

    // Reset cooldowns and set up the times.
    if (ta.with_events()) {
        ta.reset_cooldowns();
    }
    ta.set_dtime(m_data->time.hi, m_data->time.lo);

    // Copy over the state.
    for (auto j = 0u; j < 6u; ++j) {
        st_data[j] = sv(pidx, j);
    }

    // NOTE: compute the radius on the fly from the x/y/z coords.
    st_data[6] = std::sqrt(st_data[0] * st_data[0] + st_data[1] * st_data[1] + st_data[2] * st_data[2]);

    // Copy over the parameters.
    auto pars_data = ta.get_pars_data();
    for (std::uint32_t i = 0; i < npars; ++i) {
        pars_data[i] = pv(pidx, i);
    }
}

// Setup the batch integrator ta to integrate the trajectory for the particles
// in the index range [pidx_begin, pidx_end). Time and states are taken from
// the current values in the global sim data.
template <typename T>
void sim::init_batch_ta(T &ta, size_type pidx_begin, size_type pidx_end) const
{
    namespace stdex = std::experimental;

    const auto batch_size = ta.get_batch_size();
    const auto nparts = get_nparts();
    const auto npars = get_npars();

    assert(pidx_end > pidx_begin);
    assert(pidx_end - pidx_begin == batch_size);
    assert(pidx_end <= nparts);

    // Create views on the data.
    stdex::mdspan sv(m_state->data(), stdex::extents<size_type, stdex::dynamic_extent, 7u>(nparts));
    stdex::mdspan pv(m_pars->data(), nparts, npars);

    stdex::mdspan st(ta.get_state_data(), stdex::extents<std::uint32_t, 7u, stdex::dynamic_extent>(batch_size));
    stdex::mdspan pt(ta.get_pars_data(), npars, batch_size);

    // Reset cooldowns and set up the times.
    if (ta.with_events()) {
        ta.reset_cooldowns();
    }
    ta.set_dtime(m_data->time.hi, m_data->time.lo);

    // Copy over the state and params.
    for (std::uint32_t i = 0; i < batch_size; ++i) {
        for (auto j = 0u; j < 6u; ++j) {
            st(j, i) = sv(pidx_begin + i, j);
        }

        // NOTE: compute the radius on the fly from the x/y/z coords.
        st(6, i) = std::sqrt(st(0, i) * st(0, i) + st(1, i) * st(1, i) + st(2, i) * st(2, i));

        for (std::uint32_t j = 0; j < npars; ++j) {
            pt(j, i) = pv(pidx_begin + i, j);
        }
    }
}

// Compute the AABB of the trajectory of the particle at index pidx within a chunk.
// chunk_idx is the chunk index, chunk_begin/end the time range of the chunk.
template <typename T>
void sim::compute_particle_aabb(unsigned chunk_idx, const T &chunk_begin, const T &chunk_end, size_type pidx)
{
    namespace hy = heyoka;
    using dfloat = hy::detail::dfloat<double>;
    namespace stdex = std::experimental;

    const auto nparts = get_nparts();

    // Fetch a view on the state vector in order to
    // access the particles' sizes.
    stdex::mdspan sv(std::as_const(m_state)->data(), stdex::extents<size_type, stdex::dynamic_extent, 7u>(nparts));

    // Fetch pointers to the AABB data for the current chunk.
    const auto offset = nparts * chunk_idx;

    auto *CASCADE_RESTRICT x_lb_ptr = m_data->x_lb.data() + offset;
    auto *CASCADE_RESTRICT y_lb_ptr = m_data->y_lb.data() + offset;
    auto *CASCADE_RESTRICT z_lb_ptr = m_data->z_lb.data() + offset;
    auto *CASCADE_RESTRICT r_lb_ptr = m_data->r_lb.data() + offset;

    auto *CASCADE_RESTRICT x_ub_ptr = m_data->x_ub.data() + offset;
    auto *CASCADE_RESTRICT y_ub_ptr = m_data->y_ub.data() + offset;
    auto *CASCADE_RESTRICT z_ub_ptr = m_data->z_ub.data() + offset;
    auto *CASCADE_RESTRICT r_ub_ptr = m_data->r_ub.data() + offset;

    // Setup the initial values for the bounding box.
    constexpr auto finf = std::numeric_limits<float>::infinity();

    x_lb_ptr[pidx] = finf;
    y_lb_ptr[pidx] = finf;
    z_lb_ptr[pidx] = finf;
    r_lb_ptr[pidx] = finf;

    x_ub_ptr[pidx] = -finf;
    y_ub_ptr[pidx] = -finf;
    z_ub_ptr[pidx] = -finf;
    r_ub_ptr[pidx] = -finf;

    // Fetch the particle radius.
    const auto p_radius = sv(pidx, 6);

    // Cache a few quantities.
    const auto order = m_data->s_ta.get_order();
    const auto &s_data = m_data->s_data;
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
    // NOTE: instead of this, perhaps we can just iterate below until
    // t_coords_end or until the first substep whose end is *greater than or
    // equal to* the end of the chunk, whichever comes first.
    auto ss_it_end = std::lower_bound(ss_it_begin, tcoords_end, chunk_end);
    // Bump it up by one to define a half-open range.
    // NOTE: don't bump it if it is already at the end.
    // This could happen at the last chunk due to FP rounding,
    // or if the integration for this particle was interrupted
    // early due to a stopping terminal event, or if tcoords is empty.
    ss_it_end += (ss_it_end != tcoords_end);

    // Iterate over all substeps and update the bounding box
    // for the current particle.
    // NOTE: if the particle has no steps covering the current chunk,
    // then this loop will never be entered, and the AABB for the particle
    // in the current chunk will remain inited with infinities. This is
    // fine as, after integration + AABB computation, we will redefine the
    // superstep size and number of chunks to avoid the current chunk.
    for (auto it = ss_it_begin; it != ss_it_end; ++it) {
        // it points to the end of a substep which overlaps
        // with the current chunk. The size of the polynomial evaluation
        // interval is the size of the intersection between the substep and
        // the chunk.

        // Determine the initial time coordinate of the substep, relative
        // to init_time. If it is tcoords_begin, ss_start will be zero, otherwise
        // ss_start is given by the iterator preceding it.
        const auto ss_start = (it == tcoords_begin) ? dfloat(0) : *(it - 1);

        // Determine lower/upper bounds of the evaluation interval,
        // relative to init_time.
        // NOTE: min/max is fine here: values in tcoords are always checked
        // for finiteness, chunk_begin/end are also checked in
        // get_chunk_begin_end().
        const auto ev_lb = std::max(chunk_begin, ss_start);
        const auto ev_ub = std::min(chunk_end, *it);

        // Create the actual evaluation interval, referring
        // it to the beginning of the substep.
        const auto h_int_lb = static_cast<double>(ev_lb - ss_start);
        const auto h_int_ub = static_cast<double>(ev_ub - ss_start);

        // Determine the index of the substep within the chunk.
        // NOTE: we checked at the end of the numerical integration
        // that the size of tcoords can be represented by its iterator
        // type's difference. Thus, the computation it - tcoords_begin is safe.
        const auto ss_idx = boost::numeric_cast<decltype(s_data[pidx].tc_x.size())>(it - tcoords_begin);

        // Compute the pointers to the TCs for the current particle
        // and substep.
        const auto tc_ptr_x = s_data[pidx].tc_x.data() + ss_idx * (order + 1u);
        const auto tc_ptr_y = s_data[pidx].tc_y.data() + ss_idx * (order + 1u);
        const auto tc_ptr_z = s_data[pidx].tc_z.data() + ss_idx * (order + 1u);
        const auto tc_ptr_r = s_data[pidx].tc_r.data() + ss_idx * (order + 1u);

        // Run the polynomial evaluations using interval arithmetic.
        // NOTE: jit for performance? If so, we can do all 4 coordinates
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
        auto lb_make_float = [&](double lb) {
            auto ret = std::nextafter(static_cast<float>(lb), -finf);

            if (!std::isfinite(ret)) {
                throw std::invalid_argument(fmt::format("The computation of the bounding box for the particle at index "
                                                        "{} produced the non-finite lower bound {}",
                                                        pidx, ret));
            }

            return ret;
        };
        auto ub_make_float = [&](double ub) {
            auto ret = std::nextafter(static_cast<float>(ub), finf);

            if (!std::isfinite(ret)) {
                throw std::invalid_argument(fmt::format("The computation of the bounding box for the particle at index "
                                                        "{} produced the non-finite upper bound {}",
                                                        pidx, ret));
            }

            return ret;
        };

        // Update the bounding box for the current particle.
        // NOTE: min/max is fine: the make_float() helpers check for finiteness,
        // and the other operand is never NaN.
        x_lb_ptr[pidx] = std::min(x_lb_ptr[pidx], lb_make_float(x_int.lower));
        y_lb_ptr[pidx] = std::min(y_lb_ptr[pidx], lb_make_float(y_int.lower));
        z_lb_ptr[pidx] = std::min(z_lb_ptr[pidx], lb_make_float(z_int.lower));
        r_lb_ptr[pidx] = std::min(r_lb_ptr[pidx], lb_make_float(r_int.lower));

        x_ub_ptr[pidx] = std::max(x_ub_ptr[pidx], ub_make_float(x_int.upper));
        y_ub_ptr[pidx] = std::max(y_ub_ptr[pidx], ub_make_float(y_int.upper));
        z_ub_ptr[pidx] = std::max(z_ub_ptr[pidx], ub_make_float(z_int.upper));
        r_ub_ptr[pidx] = std::max(r_ub_ptr[pidx], ub_make_float(r_int.upper));
    }
}

// Perform the Morton encoding of the centres of the AABBs of the particles
// and sort the AABB data according to the codes.
void sim::morton_encode_sort_parallel()
{
    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    // Fetch the number of particles and chunks from m_data.
    const auto nparts = get_nparts();
    const auto nchunks = m_data->nchunks;

    constexpr auto morton_enc = mortonnd::MortonNDLutEncoder<4, 16, 8>();

    constexpr auto finf = std::numeric_limits<float>::infinity();

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(0u, nchunks), [&](const auto &range) {
        for (auto chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
            // Fetch the global AABB for this chunk.
            auto &glb = m_data->global_lb[chunk_idx];
            auto &gub = m_data->global_ub[chunk_idx];

            // Bump up the upper bounds to make absolutely sure that ub > lb, as required
            // by the spatial discretisation function.
            for (auto i = 0u; i < 4u; ++i) {
                gub[i] = std::nextafter(gub[i], finf);

                // Check that the interval size is finite.
                // This also ensures that ub/lb are finite.
                if (!std::isfinite(gub[i] - glb[i])) {
                    throw std::invalid_argument(
                        "A global bounding box with non-finite boundaries and/or size was generated");
                }
            }

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

            assert(std::is_sorted(srt_mcodes_ptr, srt_mcodes_ptr + nparts));
        }
    });

    logger->trace("Morton encoding and sorting time: {}s", sw);
}

// NOTE: exception-wise: no user-visible data is altered
// until the end of the function, at which point the new
// sim data is set up in a noexcept manner.
outcome sim::step(double dt)
{
    namespace hy = heyoka;
    using dfloat = hy::detail::dfloat<double>;
    namespace stdex = std::experimental;

    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    logger->trace("---- STEP BEGIN ---");

    if (get_nparts() == 0u) {
        throw std::invalid_argument("Cannot integrate a simulation with no particles");
    }

    if (!std::isfinite(dt)) {
        throw std::invalid_argument("The superstep size must be finite");
    }

    // Setup the superstep size.
    m_data->delta_t = dt <= 0 ? infer_superstep() : dt;

    // Setup the number of chunks.
    m_data->nchunks = boost::numeric_cast<unsigned>(std::ceil(m_data->delta_t / m_ct));
    if (m_data->nchunks == 0u) {
        throw std::invalid_argument(
            "The number of chunks cannot be zero (this likely indicates that a zero supertstep size was specified)");
    }
    logger->trace("Number of chunks: {}", m_data->nchunks);

    // Cache a few quantities.
    const auto delta_t = m_data->delta_t;
    const auto nchunks = m_data->nchunks;
    const auto batch_size = m_data->b_ta.get_batch_size();
    const auto nparts = get_nparts();
    const auto order = m_data->s_ta.get_order();
    // Number of regular batches.
    const auto n_batches = nparts / batch_size;
    // The time coordinate at the beginning of
    // the superstep.
    const auto init_time = m_data->time;

    // Prepare the s_data buffers with the correct sizes.

    // NOTE: this is a helper that resizes vec to new_size
    // only if vec is currently smaller than new_size.
    // The idea here is that we don't want to downsize the
    // vector if not necessary.
    auto resize_if_needed = [](auto new_size, auto &...vecs) {
        auto apply = [new_size](auto &vec) {
            if (vec.size() < new_size) {
                vec.resize(boost::numeric_cast<decltype(vec.size())>(new_size));
            }
        };

        (apply(vecs), ...);
    };

    // Many of the buffers need to be of size nparts * nchunks. Do
    // an overflow check.
    if (nparts > std::numeric_limits<size_type>::max() / nchunks) {
        throw std::overflow_error("An overflow condition was detected in the step() function");
    }
    const auto npnc = nparts * nchunks;

    // The substep data.
    resize_if_needed(nparts, m_data->s_data);

    // The AABBs data.
    resize_if_needed(npnc, m_data->x_lb, m_data->y_lb, m_data->z_lb, m_data->r_lb);
    resize_if_needed(npnc, m_data->x_ub, m_data->y_ub, m_data->z_ub, m_data->r_ub);

    // Morton encoding/ordering.
    resize_if_needed(npnc, m_data->mcodes, m_data->vidx);

    // Morton-sorted AABBs data.
    resize_if_needed(npnc, m_data->srt_x_lb, m_data->srt_y_lb, m_data->srt_z_lb, m_data->srt_r_lb);
    resize_if_needed(npnc, m_data->srt_x_ub, m_data->srt_y_ub, m_data->srt_z_ub, m_data->srt_r_ub);
    resize_if_needed(npnc, m_data->srt_mcodes);

    // Final state vector.
    // NOTE: contrary to m_state, this does not contain the particle sizes,
    // hence the number of columns is 6 and not 7.
    m_data->final_state.resize(nparts * 6u);
    stdex::mdspan fsv(m_data->final_state.data(), stdex::extents<size_type, stdex::dynamic_extent, 6u>(nparts));

    // Global AABBs data.
    resize_if_needed(nchunks, m_data->global_lb, m_data->global_ub);
    // NOTE: the global AABBs need to be set up with
    // initial values.
    constexpr auto finf = std::numeric_limits<float>::infinity();
    std::ranges::fill(m_data->global_lb, std::array{finf, finf, finf, finf});
    std::ranges::fill(m_data->global_ub, std::array{-finf, -finf, -finf, -finf});

    // BVH data.
    resize_if_needed(nchunks, m_data->bvh_trees, m_data->nc_buffer, m_data->ps_buffer, m_data->nplc_buffer);

    // Broad phase data.
    resize_if_needed(nchunks, m_data->bp_coll, m_data->bp_data_caches);

    // Narrow phase data.
    resize_if_needed(nchunks, m_data->np_caches);

    // Stopping terminal events and err_nf_state vectors.
    m_data->ste_vec.clear();
    m_data->err_nf_state_vec.clear();

    // Numerical integration and computation of the AABBs in batch mode.
    auto batch_int_aabb = [&](const auto &range) {
        // Fetch batch data from the cache, or create it.
        std::unique_ptr<sim_data::batch_data> bdata_ptr;

        if (m_data->b_ta_cache.try_pop(bdata_ptr)) {
            assert(bdata_ptr);
            assert(bdata_ptr->pfor_ts.size() == batch_size);
        } else {
            SPDLOG_LOGGER_DEBUG(logger, "Creating new batch data");

            bdata_ptr = std::make_unique<sim_data::batch_data>(m_data->b_ta);
            bdata_ptr->pfor_ts.resize(boost::numeric_cast<decltype(bdata_ptr->pfor_ts.size())>(batch_size));
        }

        // Cache a few variables.
        auto &ta = bdata_ptr->ta;
        auto &pfor_ts = bdata_ptr->pfor_ts;
        stdex::mdspan st(std::as_const(ta).get_state_data(),
                         stdex::extents<std::uint32_t, 7u, stdex::dynamic_extent>(batch_size));
        auto &s_data = m_data->s_data;
        const auto &ta_tc = ta.get_tc();

        // The first step is the numerical integration for all particles
        // in range throughout the entire superstep.
        // NOTE: the idea here is that if a particle in a batch
        // stops early due to a stopping terminal event, then
        // we will still keep on integrating the other particles.
        for (auto idx = range.begin(); idx != range.end(); ++idx) {
            // Particle indices corresponding to the current batch.
            const auto pidx_begin = idx * batch_size;
            const auto pidx_end = pidx_begin + batch_size;

            // Clear up the Taylor coefficients and the times
            // of the substeps for the current particle, and fill in
            // the pfor_ts vector.
            for (auto i = pidx_begin; i < pidx_end; ++i) {
                s_data[i].tc_x.clear();
                s_data[i].tc_y.clear();
                s_data[i].tc_z.clear();
                s_data[i].tc_vx.clear();
                s_data[i].tc_vy.clear();
                s_data[i].tc_vz.clear();
                s_data[i].tc_r.clear();

                s_data[i].tcoords.clear();

                pfor_ts[i - pidx_begin] = delta_t;
            }

            // Setup the integrator.
            init_batch_ta(ta, pidx_begin, pidx_end);

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
                    const auto time_f = dfloat(ta.get_dtime().first[i], ta.get_dtime().second[i]);
                    s_data[pidx_begin + i].tcoords.push_back(time_f - init_time);
                    if (!isfinite(s_data[pidx_begin + i].tcoords.back())) {
                        throw std::invalid_argument(fmt::format("A non-finite time coordinate was generated during the "
                                                                "numerical integration of the particle at index {}",
                                                                pidx_begin + i));
                    }

                    // Copy over the Taylor coefficients.
                    // NOTE: resize + copy, instead of push back? In such
                    // a case, we should probably use the no init allocator
                    // for the tc vectors.
                    for (std::uint32_t o = 0; o <= order; ++o) {
                        s_data[pidx_begin + i].tc_x.push_back(ta_tc[o * batch_size + i]);
                        s_data[pidx_begin + i].tc_y.push_back(ta_tc[(order + 1u) * batch_size + o * batch_size + i]);
                        s_data[pidx_begin + i].tc_z.push_back(
                            ta_tc[2u * (order + 1u) * batch_size + o * batch_size + i]);
                        s_data[pidx_begin + i].tc_vx.push_back(
                            ta_tc[3u * (order + 1u) * batch_size + o * batch_size + i]);
                        s_data[pidx_begin + i].tc_vy.push_back(
                            ta_tc[4u * (order + 1u) * batch_size + o * batch_size + i]);
                        s_data[pidx_begin + i].tc_vz.push_back(
                            ta_tc[5u * (order + 1u) * batch_size + o * batch_size + i]);
                        s_data[pidx_begin + i].tc_r.push_back(
                            ta_tc[6u * (order + 1u) * batch_size + o * batch_size + i]);
                    }
                }

                return true;
            };
            std::function<bool(hy::taylor_adaptive_batch<double> &)> cbf(std::cref(cb));

            while (true) {
                // Integrate.
                ta.propagate_for(pfor_ts, hy::kw::write_tc = true, hy::kw::callback = cbf);

                // Let's do a first check on the outcomes to determine if everything went
                // to time_limit or a non-finite state was generated.
                std::uint32_t n_tlimit = 0, n_err_nf_state = 0;
                for (std::uint32_t i = 0; i < batch_size; ++i) {
                    const auto oc = std::get<0>(ta.get_propagate_res()[i]);

                    if (oc == hy::taylor_outcome::err_nf_state) {
                        // Non-finite state detected.
                        ++n_err_nf_state;

                        // Record in err_nf_state_vec the particle index and the time coordinate
                        // of the last successful step for the particle (relative to the beginning
                        // of the superstep).
                        // NOTE: the propagate callback is NOT executed if a non-finite state
                        // was detected, thus tcoords contains data only up to the last successful step.
                        const auto &tcoords = s_data[pidx_begin + i].tcoords;
                        const auto last_t = tcoords.empty() ? 0. : static_cast<double>(tcoords.back());
                        m_data->err_nf_state_vec.emplace_back(pidx_begin + i, last_t);
                    } else {
                        n_tlimit += (oc == hy::taylor_outcome::time_limit);
                    }
                }

                if (n_err_nf_state != 0u) {
                    // If any non-finite state in the batch was detected, just exit,
                    // as there is no point in doing anything else.
                    // NOTE: no need to run the overflow check on tcoords as we won't
                    // be computing any AABB for this particle, nor we will be proceeding
                    // past the integration + AABB computation phase.
                    return;
                }

                if (n_tlimit == batch_size) {
                    // The happy path, just break out of the endless loop.
                    break;
                }

                // NOTE: at this point, we know that the propagate_for() exited before
                // reaching the final time for at least 1 batch element. This means
                // that at least 1 stopping terminal event occurred.
                // We need to iterate again on the propagate results and perform different
                // actions depending on the outcomes.

#if !defined(NDEBUG)
                // Keep track of the number of stopping terminal events detected
                // for debug purposes.
                std::uint32_t n_ste = 0;
#endif

                for (std::uint32_t i = 0; i < batch_size; ++i) {
                    const auto oc = std::get<0>(ta.get_propagate_res()[i]);

                    // Get the time coordinate for the current batch element in double-length format.
                    const auto cur_t = dfloat(ta.get_dtime().first[i], ta.get_dtime().second[i]);

                    if (oc < hy::taylor_outcome{0} && oc > hy::taylor_outcome::success) {
                        // Stopping terminal event: set the integration time in
                        // pfor_ts to zero, and record the event.
                        // NOTE: setting pfor_ts to zero means that the next iteration
                        // this batch element will return an outcome of time_limit.
                        pfor_ts[i] = 0;
                        m_data->ste_vec.emplace_back(pidx_begin + i,
                                                     // Store the trigger time wrt
                                                     // the beginning of the superstep.
                                                     static_cast<double>(cur_t - init_time),
                                                     // Compute the event index.
                                                     static_cast<std::uint32_t>(-static_cast<std::int64_t>(oc) - 1));

#if !defined(NDEBUG)
                        ++n_ste;
#endif
                    } else {
                        // For all the other possible outcomes, we will set pfor_ts
                        // to the remaining time for the batch element (which could be zero).
                        const auto rem_time = init_time + delta_t - cur_t;

                        if (!isfinite(rem_time)) {
                            throw std::invalid_argument(
                                fmt::format("A non-finite time was generated during the integration of particle {}",
                                            pidx_begin + i));
                        }

                        // NOTE: not sure if rem_time can be negative due to floating-point
                        // rounding in corner cases, so let's stay on the safe side.
                        pfor_ts[i] = std::max(0., static_cast<double>(rem_time));
                    }
                }

                assert(n_ste > 0u);
            }

            // Overflow checks on tcoords: tcoords' size must fit in the
            // iterator difference type. This is relied upon when computing
            // the index of a substep within a chunk.
            for (std::uint32_t i = 0; i < batch_size; ++i) {
                const auto &tcoords = s_data[pidx_begin + i].tcoords;

                using it_diff_t = std::iter_difference_t<decltype(tcoords.begin())>;
                using it_udiff_t = std::make_unsigned_t<it_diff_t>;
                if (tcoords.size() > static_cast<it_udiff_t>(std::numeric_limits<it_diff_t>::max())) {
                    throw std::overflow_error(
                        fmt::format("Overflow detected during the numerical integration of the particle at index {}",
                                    pidx_begin + i));
                }
            }

            // Fill in the state at the end of the superstep.
            // NOTE: for those particles whose integration was interrupted early due
            // to stopping terminal events, the final_state vector will contain the
            // state at the interruption time.
            for (std::uint32_t i = 0; i < batch_size; ++i) {
                for (auto j = 0u; j < 6u; ++j) {
                    fsv(pidx_begin + i, j) = st(j, i);
                }
            }
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
            const auto [chunk_begin, chunk_end] = m_data->get_chunk_begin_end(chunk_idx, m_ct);

            for (auto idx = range.begin(); idx != range.end(); ++idx) {
                // Particle indices corresponding to the current batch.
                const auto pidx_begin = idx * batch_size;

                for (std::uint32_t i = 0; i < batch_size; ++i) {
                    // Compute the AABB for the current particle.
                    compute_particle_aabb(chunk_idx, dfloat(chunk_begin), dfloat(chunk_end), pidx_begin + i);

                    // Update the local AABB with the bounding box for the current particle.
                    // NOTE: min/max usage is safe, because compute_particle_aabb()
                    // ensures that the bounding boxes are finite.
                    local_lb[0] = std::min(local_lb[0], x_lb_ptr[pidx_begin + i]);
                    local_lb[1] = std::min(local_lb[1], y_lb_ptr[pidx_begin + i]);
                    local_lb[2] = std::min(local_lb[2], z_lb_ptr[pidx_begin + i]);
                    local_lb[3] = std::min(local_lb[3], r_lb_ptr[pidx_begin + i]);

                    local_ub[0] = std::max(local_ub[0], x_ub_ptr[pidx_begin + i]);
                    local_ub[1] = std::max(local_ub[1], y_ub_ptr[pidx_begin + i]);
                    local_ub[2] = std::max(local_ub[2], z_ub_ptr[pidx_begin + i]);
                    local_ub[3] = std::max(local_ub[3], r_ub_ptr[pidx_begin + i]);
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

        // Put the integrator data (back) into the caches.
        m_data->b_ta_cache.push(std::move(bdata_ptr));
    };

    // Numerical integration and computation of the AABBs for the scalar remainder.
    auto scalar_int_aabb = [&](const auto &range) {
        // Fetch an integrator from the cache, or create it.
        std::unique_ptr<hy::taylor_adaptive<double>> ta_ptr;

        if (m_data->s_ta_cache.try_pop(ta_ptr)) {
            assert(ta_ptr);
        } else {
            SPDLOG_LOGGER_DEBUG(logger, "Creating new integrator");

            ta_ptr = std::make_unique<hy::taylor_adaptive<double>>(m_data->s_ta);
        }

        // Cache a few variables.
        auto &ta = *ta_ptr;
        const auto *st_data = ta.get_state_data();
        auto &s_data = m_data->s_data;
        const auto &ta_tc = ta.get_tc();

        // The first step is the numerical integration for all particles
        // in range throughout the entire superstep.
        for (auto pidx = range.begin(); pidx != range.end(); ++pidx) {
            auto &tcoords = s_data[pidx].tcoords;

            // Clear up the Taylor coefficients and the times
            // of the substeps for the current particle.
            s_data[pidx].tc_x.clear();
            s_data[pidx].tc_y.clear();
            s_data[pidx].tc_z.clear();
            s_data[pidx].tc_vx.clear();
            s_data[pidx].tc_vy.clear();
            s_data[pidx].tc_vz.clear();
            s_data[pidx].tc_r.clear();

            tcoords.clear();

            // Setup the integrator.
            init_scalar_ta(ta, pidx);

            // Setup the propagate_for() callback.
            auto cb = [&](auto &) {
                // NOTE: ignore if a zero timestep was taken.
                if (ta.get_last_h() == 0.) {
                    return true;
                }

                // Record the time coordinate at the end of the step, relative
                // to the initial time.
                const auto time_f = dfloat(ta.get_dtime().first, ta.get_dtime().second);
                tcoords.push_back(time_f - init_time);
                if (!isfinite(tcoords.back())) {
                    throw std::invalid_argument(fmt::format("A non-finite time coordinate was generated during the "
                                                            "numerical integration of the particle at index {}",
                                                            pidx));
                }

                // Copy over the Taylor coefficients.
                // NOTE: resize + copy, instead of push back? In such
                // a case, we should probably use the no init allocator
                // for the tc vectors.
                for (std::uint32_t o = 0; o <= order; ++o) {
                    s_data[pidx].tc_x.push_back(ta_tc[o]);
                    s_data[pidx].tc_y.push_back(ta_tc[order + 1u + o]);
                    s_data[pidx].tc_z.push_back(ta_tc[2u * (order + 1u) + o]);
                    s_data[pidx].tc_vx.push_back(ta_tc[3u * (order + 1u) + o]);
                    s_data[pidx].tc_vy.push_back(ta_tc[4u * (order + 1u) + o]);
                    s_data[pidx].tc_vz.push_back(ta_tc[5u * (order + 1u) + o]);
                    s_data[pidx].tc_r.push_back(ta_tc[6u * (order + 1u) + o]);
                }

                return true;
            };
            std::function<bool(hy::taylor_adaptive<double> &)> cbf(std::cref(cb));

            // Integrate.
            const auto oc = std::get<0>(ta.propagate_for(delta_t, hy::kw::write_tc = true, hy::kw::callback = cbf));

            // Check for errors.
            if (oc == hy::taylor_outcome::err_nf_state) {
                // The particle generated a non-finite state.
                // Record in err_nf_state_vec the particle index and the time coordinate
                // of the last successful step for the particle (relative to the beginning
                // of the superstep).
                const auto last_t = tcoords.empty() ? 0. : static_cast<double>(tcoords.back());
                m_data->err_nf_state_vec.emplace_back(pidx, last_t);

                // Just exit, as there is no point in doing anything else.
                return;
            } else if (oc != hy::taylor_outcome::time_limit) {
                // Stopping terminal event detected, record it.
                assert(oc < hy::taylor_outcome{0} && oc > hy::taylor_outcome::success);

                // Get the time coordinate for the current batch element in double-length format.
                const auto cur_t = dfloat(ta.get_dtime().first, ta.get_dtime().second);

                m_data->ste_vec.emplace_back(pidx,
                                             // Store the trigger time wrt
                                             // the beginning of the superstep.
                                             static_cast<double>(cur_t - init_time),
                                             // Compute the event index.
                                             static_cast<std::uint32_t>(-static_cast<std::int64_t>(oc) - 1));
            }

            // Overflow check on tcoords: tcoords' size must fit in the
            // iterator difference type. This is relied upon when computing
            // the index of a substep within a chunk.
            using it_diff_t = std::iter_difference_t<decltype(tcoords.begin())>;
            using it_udiff_t = std::make_unsigned_t<it_diff_t>;
            if (tcoords.size() > static_cast<it_udiff_t>(std::numeric_limits<it_diff_t>::max())) {
                throw std::overflow_error(fmt::format(
                    "Overflow detected during the numerical integration of the particle at index {}", pidx));
            }

            // Fill in the state at the end of the superstep.
            // NOTE: if the integration was was interrupted early due
            // to a stopping terminal event, the final_state vector will contain the
            // state at the interruption time.
            for (auto j = 0u; j < 6u; ++j) {
                fsv(pidx, j) = st_data[j];
            }
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
            const auto [chunk_begin, chunk_end] = m_data->get_chunk_begin_end(chunk_idx, m_ct);

            for (auto pidx = range.begin(); pidx != range.end(); ++pidx) {
                // Compute the AABB for the current particle.
                compute_particle_aabb(chunk_idx, dfloat(chunk_begin), dfloat(chunk_end), pidx);

                // Update the local AABB with the bounding box for the current particle.
                // NOTE: min/max usage is safe, because compute_particle_aabb()
                // ensures that the bounding boxes are finite.
                local_lb[0] = std::min(local_lb[0], x_lb_ptr[pidx]);
                local_lb[1] = std::min(local_lb[1], y_lb_ptr[pidx]);
                local_lb[2] = std::min(local_lb[2], z_lb_ptr[pidx]);
                local_lb[3] = std::min(local_lb[3], r_lb_ptr[pidx]);

                local_ub[0] = std::max(local_ub[0], x_ub_ptr[pidx]);
                local_ub[1] = std::max(local_ub[1], y_ub_ptr[pidx]);
                local_ub[2] = std::max(local_ub[2], z_ub_ptr[pidx]);
                local_ub[3] = std::max(local_ub[3], r_ub_ptr[pidx]);
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

    // Check if the dynamical propagation generated
    // non-finite values.
    if (!m_data->err_nf_state_vec.empty()) {
        // Fetch the earliest element in err_nf_state_vec.
        const auto nf_it = std::ranges::min_element(m_data->err_nf_state_vec, [](const auto &tup1, const auto &tup2) {
            return std::get<1>(tup1) < std::get<1>(tup2);
        });

        // Setup the interrupt info.
        m_int_info.emplace(*nf_it);

        logger->debug("The step function was interrupted due to particle {} generating a non-finite state during the "
                      "dynamical propagation",
                      std::get<0>(*nf_it));

        logger->trace("Total propagation time: {}s", sw);

        logger->trace("---- STEP END ---");

        return outcome::err_nf_state;
    }

    // Check if we ran into stopping terminal events.
    auto ste_it = m_data->ste_vec.end();
    if (!m_data->ste_vec.empty()) {
        // Fetch the earliest element in ste_vec.
        ste_it = std::ranges::min_element(
            m_data->ste_vec, [](const auto &tup1, const auto &tup2) { return std::get<1>(tup1) < std::get<1>(tup2); });

        // The earliest stopping terminal event redefines the superstep
        // size and, by extension, the number of chunks.
        // NOTE: use std::min() for FP paranoia.
        m_data->delta_t = std::min(std::get<1>(*ste_it), m_data->delta_t);

        // Setup the number of chunks.
        const auto new_nchunks = boost::numeric_cast<unsigned>(std::ceil(m_data->delta_t / m_ct));
        if (new_nchunks == 0u) {
            throw std::invalid_argument("The recomputed number of chunks after the triggering of a stopping terminal "
                                        "event cannot be zero (this likely indicates that the simulation was restarted "
                                        "without resolving a stopping terminal event)");
        }
        assert(new_nchunks <= m_data->nchunks);
        m_data->nchunks = new_nchunks;

        SPDLOG_LOGGER_DEBUG(logger, "Number of chunks adjusted after stopping terminal event: {}", m_data->nchunks);
    }

#if !defined(NDEBUG)
    verify_global_aabbs();
#endif

    // Computation of the Morton codes and sorting.
    morton_encode_sort_parallel();

    // Construction of the BVH trees.
    construct_bvh_trees_parallel();

    // Broad phase collision detection.
    broad_phase_parallel();

    // Narrow phase collision detection.
    narrow_phase_parallel();

    // Data to determine and setup the outcome of the step.
    outcome oc = outcome::success;

    // Pointer to the time of the earliest event
    // causing the interruption of the simulation.
    // If there is no such event, this will remain null.
    const double *interrupt_time = nullptr;

    // Iterator to the collision vector, initially set to end().
    // It will be set up to point to the earliest collision
    // within the superstep, if any.
    auto coll_it = m_data->coll_vec.end();

    // Check for particle-particle collisions.
    if (!m_data->coll_vec.empty()) {
        // Fetch the earliest collision.
        coll_it = std::ranges::min_element(
            m_data->coll_vec, [](const auto &tup1, const auto &tup2) { return std::get<2>(tup1) < std::get<2>(tup2); });

        // Set the interrupt time.
        interrupt_time = &std::get<2>(*coll_it);

        // Set the outcome.
        oc = outcome::collision;
    }

    // Check for stopping terminal events.
    if (!m_data->ste_vec.empty()) {
        // NOTE: ste_it was already set up.
        assert(ste_it != m_data->ste_vec.end());

        // Fetch the corresponding time.
        const double *new_itime_ptr = &std::get<1>(*ste_it);

        // Update interrupt_time and oc, if needed.
        if (interrupt_time == nullptr || *new_itime_ptr < *interrupt_time) {
            interrupt_time = new_itime_ptr;

            // Check which type of terminal event we ran into.
            // NOTE: will have to modify this if we add support
            // for arbitrary terminal events.
            if (with_exit_event() && exit_event_idx() == std::get<2>(*ste_it)) {
                oc = outcome::exit;
            } else {
                assert(with_reentry_event());
                assert(reentry_event_idx() == std::get<2>(*ste_it));

                oc = outcome::reentry;
            }
        }
    }

    if (oc == outcome::success) {
        // No interruptions. We just need to:
        // - update the time variable,
        // - copy in final_state, which was
        //   filled in during the numerical integration
        //   of the particles,
        // - reset the interrupt info.

        assert(!interrupt_time);

        // NOTE: it is *important* that everything in this
        // block is noexcept.

        // Update the time coordinate.
        // NOTE: the original delta_t is fine,
        // since we did not detect any stopping terminal
        // event that would have redefined delta_t.
        m_data->time += delta_t;

        // Copy the final state.
        copy_from_final_state();

        // Reset the interrupt data.
        m_int_info.reset();
    } else {
        // Some event interrupted the integration.
        // We need to:
        // - propagate the state of all particles up to the
        //   first interruption,
        // - update the time coordinate,
        // - copy in final_state, which was filled
        //   in by dense_propagate(),
        // - set up the interrupt info.

        assert(interrupt_time);

        // Propagate the state of all particles up to the interrupt
        // time using dense output, writing the updated state
        // into final_state.
        // NOTE: interruption times are all reported with respect
        // to the time coordinate at the beginning of the superstep,
        // as expected by dense_propagate().
        dense_propagate(*interrupt_time);

        // NOTE: noexcept until the end of the block.

        // Update the time coordinate.
        m_data->time += *interrupt_time;

        // Copy the updated state.
        copy_from_final_state();

        // Setup the interrupt data.
        switch (oc) {
            case outcome::collision:
                assert(coll_it != m_data->coll_vec.end());
                m_int_info.emplace(std::array{std::get<0>(*coll_it), std::get<1>(*coll_it)});
                break;
            case outcome::exit:
                assert(ste_it != m_data->ste_vec.end());
                m_int_info.emplace(std::get<0>(*ste_it));
                break;
            default:
                assert(oc == outcome::reentry);
                assert(ste_it != m_data->ste_vec.end());
                m_int_info.emplace(std::get<0>(*ste_it));
        }
    }

    logger->trace("Total propagation time: {}s", sw);

    logger->trace("---- STEP END ---");

    return oc;
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
                    // Fetch batch data from the cache, or create it.
                    std::unique_ptr<sim_data::batch_data> bdata_ptr;

                    if (m_data->b_ta_cache.try_pop(bdata_ptr)) {
                        assert(bdata_ptr);
                        assert(bdata_ptr->pfor_ts.size() == batch_size);
                    } else {
                        SPDLOG_LOGGER_DEBUG(logger, "Creating new batch data");

                        bdata_ptr = std::make_unique<sim_data::batch_data>(m_data->b_ta);
                        bdata_ptr->pfor_ts.resize(boost::numeric_cast<decltype(bdata_ptr->pfor_ts.size())>(batch_size));
                    }

                    // Cache a few variables.
                    auto &ta = bdata_ptr->ta;

                    for (auto idx = range.begin(); idx < range.end(); idx += stride) {
                        // Particle indices corresponding to the current batch.
                        const auto pidx_begin = idx * batch_size;
                        const auto pidx_end = pidx_begin + batch_size;

                        // Setup the integrator.
                        init_batch_ta(ta, pidx_begin, pidx_end);

                        // Integrate a single step.
                        ta.step();

                        // Accumulate into partial_sum.
                        for (const auto &[oc, h] : ta.get_step_res()) {
                            // NOTE: ignore batch elements which did not end with success
                            // or whose timestep is not finite.
                            if (oc != hy::taylor_outcome::success || !std::isfinite(h)) {
                                continue;
                            }

                            partial_sum.first += h;
                            ++partial_sum.second;
                        }
                    }

                    // Put the integrator data (back) into the caches.
                    m_data->b_ta_cache.push(std::move(bdata_ptr));

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

                    if (m_data->s_ta_cache.try_pop(ta_ptr)) {
                        assert(ta_ptr);
                    } else {
                        SPDLOG_LOGGER_DEBUG(logger, "Creating new integrator");

                        ta_ptr = std::make_unique<hy::taylor_adaptive<double>>(m_data->s_ta);
                    }

                    // Cache a few variables.
                    auto &ta = *ta_ptr;

                    for (auto pidx = range.begin(); pidx < range.end(); pidx += stride) {
                        // Setup the integrator.
                        init_scalar_ta(ta, pidx);

                        // Integrate for a single step
                        const auto [oc, h] = ta.step();

                        // Accumulate into partial_sum.
                        if (oc != hy::taylor_outcome::success || !std::isfinite(h)) {
                            // NOTE: ignore particles which did not end with success
                            // or whose timestep is not finite.
                            continue;
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

    // NOTE: this can happen only if the simulation has zero particles, or if
    // no particle considered in the timestep determination ended with a success
    // outcome.
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
    logger->trace("Inferred superstep size: {}", res);
    SPDLOG_LOGGER_DEBUG(logger, "Number of particles considered for timestep deduction: {}", n_part_acc);

    return res;
}

// Helper to verify the global AABB computed for each chunk.
void sim::verify_global_aabbs() const
{
    constexpr auto finf = std::numeric_limits<float>::infinity();

    const auto nparts = get_nparts();
    const auto nchunks = m_data->nchunks;

    for (auto chunk_idx = 0u; chunk_idx < nchunks; ++chunk_idx) {
        std::array lb = {finf, finf, finf, finf};
        std::array ub = {-finf, -finf, -finf, -finf};

        const auto offset = nparts * chunk_idx;

        const auto *CASCADE_RESTRICT x_lb_ptr = m_data->x_lb.data() + offset;
        const auto *CASCADE_RESTRICT y_lb_ptr = m_data->y_lb.data() + offset;
        const auto *CASCADE_RESTRICT z_lb_ptr = m_data->z_lb.data() + offset;
        const auto *CASCADE_RESTRICT r_lb_ptr = m_data->r_lb.data() + offset;

        const auto *CASCADE_RESTRICT x_ub_ptr = m_data->x_ub.data() + offset;
        const auto *CASCADE_RESTRICT y_ub_ptr = m_data->y_ub.data() + offset;
        const auto *CASCADE_RESTRICT z_ub_ptr = m_data->z_ub.data() + offset;
        const auto *CASCADE_RESTRICT r_ub_ptr = m_data->r_ub.data() + offset;

        for (size_type i = 0; i < nparts; ++i) {
            lb[0] = std::min(lb[0], x_lb_ptr[i]);
            lb[1] = std::min(lb[1], y_lb_ptr[i]);
            lb[2] = std::min(lb[2], z_lb_ptr[i]);
            lb[3] = std::min(lb[3], r_lb_ptr[i]);

            ub[0] = std::max(ub[0], x_ub_ptr[i]);
            ub[1] = std::max(ub[1], y_ub_ptr[i]);
            ub[2] = std::max(ub[2], z_ub_ptr[i]);
            ub[3] = std::max(ub[3], r_ub_ptr[i]);
        }

        assert(lb == m_data->global_lb[chunk_idx]);
        assert(ub == m_data->global_ub[chunk_idx]);
    }
}

// Propagate the state of all particles up to t using dense output,
// writing the updated state into final_state.
// t is a time coordinate relative to the beginning of the current
// superstep.
void sim::dense_propagate(double t)
{
    namespace stdex = std::experimental;

    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    const auto order = m_data->s_ta.get_order();
    const auto nparts = get_nparts();

    // Fetch a view for writing into final_state.
    assert(m_data->final_state.size() == nparts * 6u);
    stdex::mdspan fsv(m_data->final_state.data(), stdex::extents<size_type, stdex::dynamic_extent, 6u>(nparts));

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(0, nparts), [&](const auto &range) {
        using dfloat = heyoka::detail::dfloat<double>;

        const auto &s_data = m_data->s_data;
        const dfloat dt(t);

        for (auto pidx = range.begin(); pidx != range.end(); ++pidx) {
            const auto &tcoords = s_data[pidx].tcoords;
            const auto tcoords_begin = tcoords.begin();
            const auto tcoords_end = tcoords.end();

            if (tcoords_begin == tcoords_end) {
                // NOTE: this should never happen because
                // this would mean that some particle took only steps
                // of zero size during the current superstep. I.e., either:
                // - a zero superstep has been taken, but this is prevented
                //   by checks, or
                // - a stopping terminal event triggered exactly at the beginning
                //   of the superstep.
                // In the latter case, because the stopping terminal event triggers
                // immediately, the superstep gets redefined to zero, which tiggers
                // the zero chunks exception.
                // However, this being FP arithmetics, I feel more safe
                // leaving the cheap runtime check here (rather than putting an assertion).
                throw std::invalid_argument(
                    fmt::format("The computation of dense_propagate() for particle {} could not be performed because "
                                "no timesteps have been taken for this particle",
                                pidx));
            }

            // Locate the first substep whose end is *greater than or
            // equal to* t.
            auto it = std::lower_bound(tcoords_begin, tcoords_end, dt);
            // NOTE: ss_it could be at the end due to FP rounding,
            // roll it back by 1 if necessary.
            it -= (it == tcoords_end);

            // Determine the initial time coordinate of the substep, relative
            // to the beginning of the superstep. If it is tcoords_begin,
            // ss_start will be zero, otherwise
            // ss_start is given by the iterator preceding it.
            const auto ss_start = (it == tcoords_begin) ? dfloat(0) : *(it - 1);

            // Determine the evaluation time for the Taylor polynomials.
            const auto eval_tm = static_cast<double>(t - ss_start);

            // Determine the index of the substep within the chunk.
            // NOTE: static cast because overflow detection has been
            // done already in earlier steps.
            const auto ss_idx = static_cast<decltype(s_data[pidx].tc_x.size())>(it - tcoords_begin);

            // Compute the pointers to the TCs for the current particle
            // and substep.
            const auto tc_ptr_x = s_data[pidx].tc_x.data() + ss_idx * (order + 1u);
            const auto tc_ptr_y = s_data[pidx].tc_y.data() + ss_idx * (order + 1u);
            const auto tc_ptr_z = s_data[pidx].tc_z.data() + ss_idx * (order + 1u);
            const auto tc_ptr_vx = s_data[pidx].tc_vx.data() + ss_idx * (order + 1u);
            const auto tc_ptr_vy = s_data[pidx].tc_vy.data() + ss_idx * (order + 1u);
            const auto tc_ptr_vz = s_data[pidx].tc_vz.data() + ss_idx * (order + 1u);

            // Run the polynomial evaluations.
            // NOTE: jit for performance? If so, we can do all variables
            // in a single JIT compiled function.
            auto horner_eval = [order, eval_tm](const double *ptr) {
                auto acc = ptr[order];
                for (auto o = 1u; o <= order; ++o) {
                    acc = ptr[order - o] + acc * eval_tm;
                }

                return acc;
            };

            const auto fx = horner_eval(tc_ptr_x);
            const auto fy = horner_eval(tc_ptr_y);
            const auto fz = horner_eval(tc_ptr_z);
            const auto fvx = horner_eval(tc_ptr_vx);
            const auto fvy = horner_eval(tc_ptr_vy);
            const auto fvz = horner_eval(tc_ptr_vz);

            // Write the state of the particle at t
            // into final_state.
            fsv(pidx, 0) = fx;
            fsv(pidx, 1) = fy;
            fsv(pidx, 2) = fz;
            fsv(pidx, 3) = fvx;
            fsv(pidx, 4) = fvy;
            fsv(pidx, 5) = fvz;
        }
    });

    logger->trace("Dense propagation time: {}s", sw);
}

template <typename T>
outcome sim::propagate_until_impl(const T &final_t, double dt)
{
    assert(isfinite(final_t) && final_t > m_data->time);

    while (true) {
        // Store the original time coord.
        const auto orig_t = m_data->time;

        // Take a step.
        const auto cur_oc = step(dt);

        if (cur_oc == outcome::success) {
            // Successful step with no interruption.
            // Check the time.
            if (m_data->time >= final_t) {
                // We are at or past the final time.

                // Propagate the state of the system up
                // to the final time, writing the new state
                // into final_state.
                dense_propagate(static_cast<double>(final_t - orig_t));

                // NOTE: everything noexcept from now on.

                // Update the time coordinate.
                m_data->time = final_t;

                // Copy in the updated state.
                copy_from_final_state();

                // NOTE: m_int_info has already been reset by the
                // step() function.

                return outcome::time_limit;
            }
        } else if (cur_oc != outcome::err_nf_state) {
            // Successful step with interruption.
            assert(cur_oc > outcome::success);

            if (m_data->time > final_t) {
                // If the interruption happened *after*
                // final_t, we need to:
                // - roll back the state to final_t,
                // - update the time,
                // - update the interrupt info.

                // Propagate the state of the system up
                // to the final time, writing the new state
                // into final_state.
                dense_propagate(static_cast<double>(final_t - orig_t));

                // NOTE: everything noexcept from now on.

                // Update the time coordinate.
                m_data->time = final_t;

                // Copy in the updated state.
                copy_from_final_state();

                // Reset the interrupt data, which
                // was set up at the end of the step() function.
                m_int_info.reset();

                return outcome::time_limit;
            } else {
                // Otherwise, we can just return cur_oc.
                // NOTE: this also includes the case
                // m_data->time == final_t: that is, collision
                // has priority wrt final time.
                return cur_oc;
            }
        } else {
            // Non-finite state detected: no need to set up
            // anything as the state of the simulation has not
            // changed since the last successful step.
            return cur_oc;
        }
    }
}

outcome sim::propagate_until(double t, double dt)
{
    using dfloat = heyoka::detail::dfloat<double>;

    if (!std::isfinite(t) || dfloat(t) < m_data->time) {
        throw std::invalid_argument(
            fmt::format("The final time passed to the propagate_until() function must be finite and not less than "
                        "the current simulation time, but a value of {} was provided instead",
                        t));
    }

    if (dfloat(t) == m_data->time) {
        // Already at the final time, don't do anything.
        return outcome::time_limit;
    }

    return propagate_until_impl(dfloat(t), dt);
}

// Helper to copy the global state vector from
// m_data->final_state to m_state.
// NOTE: mark as noexcept as we use this functions
// in chunks of code which are not supposed to throw
// for exception safety.
void sim::copy_from_final_state() noexcept
{
    namespace stdex = std::experimental;

    assert(m_data->final_state.size() == get_nparts() * 6u);

    const auto nparts = get_nparts();
    stdex::mdspan fsv(std::as_const(m_data->final_state).data(),
                      stdex::extents<size_type, stdex::dynamic_extent, 6u>(nparts));
    stdex::mdspan sv(m_state->data(), stdex::extents<size_type, stdex::dynamic_extent, 7u>(nparts));

    // NOTE: we need to quantify if the parallelisation
    // is worth it here.
    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(0, nparts), [&](const auto &range) {
        for (auto pidx = range.begin(); pidx != range.end(); ++pidx) {
            for (auto j = 0u; j < 6u; ++j) {
                sv(pidx, j) = fsv(pidx, j);
            }
        }
    });
}

} // namespace cascade
