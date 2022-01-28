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
    rx *= std::uint64_t(1) << 16;

    // Cast back to integer.
    const auto retval = static_cast<std::uint64_t>(rx);

    // Make sure to clamp it before returning, in case
    // somehow FP arithmetic makes it spill outside
    // the bound.
    // NOTE: std::min is safe with integral types.
    return std::min(retval, std::uint64_t((std::uint64_t(1) << 16) - 1u));
}

} // namespace

} // namespace detail

// Setup the scalar integrator ta to integrate the trajectory for the particle
// at index pidx. Time and state are taken from the current values in the
// global sim data.
template <typename T>
void sim::init_scalar_ta(T &ta, size_type pidx) const
{
    assert(pidx < m_x.size());

    auto *st_data = ta.get_state_data();

    // Reset cooldowns and set up the times.
    if (ta.with_events()) {
        ta.reset_cooldowns();
    }
    ta.set_dtime(m_data->time.hi, m_data->time.lo);

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
}

// Setup the batch integrator ta to integrate the trajectory for the particles
// in the index range [pidx_begin, pidx_end). Time and states are taken from
// the current values in the global sim data.
template <typename T>
void sim::init_batch_ta(T &ta, size_type pidx_begin, size_type pidx_end) const
{
    const auto batch_size = ta.get_batch_size();

    assert(pidx_end > pidx_begin);
    assert(pidx_end - pidx_begin == batch_size);
    assert(pidx_end <= m_x.size());

    auto *st_data = ta.get_state_data();

    // Reset cooldowns and set up the times.
    if (ta.with_events()) {
        ta.reset_cooldowns();
    }
    ta.set_dtime(m_data->time.hi, m_data->time.lo);

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
}

// Compute the AABB of the trajectory of the particle at index pidx within a chunk.
// chunk_idx is the chunk index, chunk_begin/end the time range of the chunk.
template <typename T>
void sim::compute_particle_aabb(unsigned chunk_idx, const T &chunk_begin, const T &chunk_end, size_type pidx)
{
    namespace hy = heyoka;

    // Fetch pointers to the AABB data for the current chunk.
    const auto offset = get_nparts() * chunk_idx;

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
    const auto p_radius = m_sizes[pidx];

    // Cache a few quantities.
    const auto order = m_data->s_ta.get_order();
    const auto &s_data = m_data->s_data;
    const auto &tcoords = s_data[pidx].tcoords;
    const auto tcoords_begin = tcoords.begin();
    const auto tcoords_end = tcoords.end();

    assert(tcoords_begin != tcoords_end);

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
void sim::morton_encode_sort()
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
        throw std::invalid_argument("The number of chunks cannot be zero");
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

    // Final state vectors.
    // NOTE: these need to match *exactly* nparts,
    // as their data will be swapped in as the new state
    // at the end of a step.
    m_data->final_x.resize(nparts);
    m_data->final_y.resize(nparts);
    m_data->final_z.resize(nparts);
    m_data->final_vx.resize(nparts);
    m_data->final_vy.resize(nparts);
    m_data->final_vz.resize(nparts);

    // Global AABBs data.
    resize_if_needed(nchunks, m_data->global_lb, m_data->global_ub);
    // NOTE: the global AABBs need to be set up with
    // initial values.
    constexpr auto finf = std::numeric_limits<float>::infinity();
    std::ranges::fill(m_data->global_lb, std::array{finf, finf, finf, finf});
    std::ranges::fill(m_data->global_ub, std::array{-finf, -finf, -finf, -finf});

    // BVH data.
    resize_if_needed(nchunks, m_data->bvh_trees, m_data->nc_buffer, m_data->ps_buffer, m_data->nplc_buffer);

    // Broad-phase data.
    resize_if_needed(nchunks, m_data->bp_coll, m_data->bp_caches, m_data->stack_caches);

    // Narrow phase data.
    resize_if_needed(nchunks, m_data->np_caches);

    // Numerical integration and computation of the AABBs in batch mode.
    auto batch_int_aabb = [&](const auto &range) {
        // Fetch an integrator from the cache, or create it.
        std::unique_ptr<hy::taylor_adaptive_batch<double>> ta_ptr;

        if (!m_data->b_ta_cache.try_pop(ta_ptr)) {
            SPDLOG_LOGGER_DEBUG(logger, "Creating new batch integrator");

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
                s_data[i].tc_vx.clear();
                s_data[i].tc_vy.clear();
                s_data[i].tc_vz.clear();
                s_data[i].tc_r.clear();

                s_data[i].tcoords.clear();
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
                    const auto time_f = hy::detail::dfloat<double>(ta.get_dtime().first[i], ta.get_dtime().second[i]);
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

            // Integrate.
            ta.propagate_for(delta_t, hy::kw::write_tc = true, hy::kw::callback = cbf);

            // Check for errors.
            for (std::uint32_t i = 0; i < batch_size; ++i) {
                if (std::get<0>(ta.get_propagate_res()[i]) != hy::taylor_outcome::time_limit) {
                    throw std::invalid_argument(fmt::format(
                        "The numerical integration of the particle at index {} returned an error", pidx_begin + i));
                }

                const auto &tcoords = s_data[pidx_begin + i].tcoords;

                // NOTE: tcoords can never be empty because that would mean that
                // we took a superstep of zero size, which is prevented by the checks
                // at the beginning of the step() function.
                assert(!tcoords.empty());

                // Overflow check on tcoords: tcoords' size must fit in the
                // iterator difference type. This is relied upon when computing
                // the index of a substep within a chunk.
                using it_diff_t = std::iter_difference_t<decltype(tcoords.begin())>;
                using it_udiff_t = std::make_unsigned_t<it_diff_t>;
                if (tcoords.size() > static_cast<it_udiff_t>(std::numeric_limits<it_diff_t>::max())) {
                    throw std::overflow_error(
                        fmt::format("Overflow detected during the numerical integration of the particle at index {}",
                                    pidx_begin + i));
                }
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

        // Put the integrator (back) into the cache.
        m_data->b_ta_cache.push(std::move(ta_ptr));
    };

    // Numerical integration and computation of the AABBs for the scalar remainder.
    auto scalar_int_aabb = [&](const auto &range) {
        // Fetch an integrator from the cache, or create it.
        std::unique_ptr<hy::taylor_adaptive<double>> ta_ptr;

        if (!m_data->s_ta_cache.try_pop(ta_ptr)) {
            SPDLOG_LOGGER_DEBUG(logger, "Creating new integrator");

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
            s_data[pidx].tc_vx.clear();
            s_data[pidx].tc_vy.clear();
            s_data[pidx].tc_vz.clear();
            s_data[pidx].tc_r.clear();

            s_data[pidx].tcoords.clear();

            // Setup the integrator.
            init_scalar_ta(ta, pidx);

            // Setup the propagate_for() callback.
            auto cb = [&](auto &) {
                // Record the time coordinate at the end of the step, relative
                // to the initial time.
                const auto time_f = hy::detail::dfloat<double>(ta.get_dtime().first, ta.get_dtime().second);
                s_data[pidx].tcoords.push_back(time_f - init_time);
                if (!isfinite(s_data[pidx].tcoords.back())) {
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
            if (oc != hy::taylor_outcome::time_limit) {
                throw std::invalid_argument(
                    fmt::format("The numerical integration of the particle at index {} returned an error", pidx));
            }

            const auto &tcoords = s_data[pidx].tcoords;

            // NOTE: tcoords can never be empty because that would mean that
            // we took a superstep of zero size, which is prevented by the checks
            // at the beginning of the step() function.
            assert(!tcoords.empty());

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

#if !defined(NDEBUG)
    verify_global_aabbs();
#endif

    // Computation of the Morton codes and sorting.
    morton_encode_sort();

    // Construction of the BVH trees.
    construct_bvh_trees();

    // Broad phase collision detection.
    broad_phase();

    // Narrow phase collision detection.
    narrow_phase();

    outcome oc{};

    if (m_data->coll_vec.empty()) {
        // NOTE: it is *important* that everything in this
        // block is noexcept.

        // Update the time coordinate.
        m_data->time += delta_t;

        // Swap in the updated state.
        m_x.swap(m_data->final_x);
        m_y.swap(m_data->final_y);
        m_z.swap(m_data->final_z);
        m_vx.swap(m_data->final_vx);
        m_vy.swap(m_data->final_vy);
        m_vz.swap(m_data->final_vz);

        // Reset the interrupt data.
        m_int_info.reset();

        // Set the exit flag.
        oc = outcome::success;
    } else {
        // Fetch the earliest collision.
        const auto coll_it = std::ranges::min_element(
            m_data->coll_vec, [](const auto &tup1, const auto &tup2) { return std::get<2>(tup1) < std::get<2>(tup2); });

        // Propagate the state of all particles up to the first
        // collision using dense output, writing the updated state
        // into the m_data->final_* vectors.
        dense_propagate(std::get<2>(*coll_it));

        // NOTE: noexcept until the end of the block.

        // Update the time coordinate.
        m_data->time += std::get<2>(*coll_it);

        // Swap in the updated state.
        m_x.swap(m_data->final_x);
        m_y.swap(m_data->final_y);
        m_z.swap(m_data->final_z);
        m_vx.swap(m_data->final_vx);
        m_vy.swap(m_data->final_vy);
        m_vz.swap(m_data->final_vz);

        // Setup the interrupt data.
        m_int_info.emplace(std::array{std::get<0>(*coll_it), std::get<1>(*coll_it)});

        // Set the exit flag.
        oc = outcome::collision;
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
                    // Fetch an integrator from the cache, or create it.
                    std::unique_ptr<hy::taylor_adaptive_batch<double>> ta_ptr;

                    if (!m_data->b_ta_cache.try_pop(ta_ptr)) {
                        SPDLOG_LOGGER_DEBUG(logger, "Creating new batch integrator");

                        ta_ptr = std::make_unique<hy::taylor_adaptive_batch<double>>(m_data->b_ta);
                    }

                    // Cache a few variables.
                    auto &ta = *ta_ptr;

                    for (auto idx = range.begin(); idx < range.end(); idx += stride) {
                        // Particle indices corresponding to the current batch.
                        const auto pidx_begin = idx * batch_size;
                        const auto pidx_end = pidx_begin + batch_size;

                        // Setup the integrator.
                        init_batch_ta(ta, pidx_begin, pidx_end);

                        // Integrate a single step.
                        ta.step();

                        // Check for errors and accumulate into partial_sum.
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

                        // Check for errors and accumulate into partial_sum.
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
// writing the updated state into the m_data->final_* vectors.
// t is a time coordinate relative to the beginning of the current
// superstep.
void sim::dense_propagate(double t)
{
    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    const auto order = m_data->s_ta.get_order();

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range<size_type>(0, get_nparts()), [&](const auto &range) {
        using dfloat = heyoka::detail::dfloat<double>;

        const auto &s_data = m_data->s_data;
        const dfloat dt(t);

        for (auto pidx = range.begin(); pidx != range.end(); ++pidx) {
            const auto &tcoords = s_data[pidx].tcoords;
            const auto tcoords_begin = tcoords.begin();
            const auto tcoords_end = tcoords.end();

            assert(tcoords_begin != tcoords_end);

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
            // into the final_* vectors.
            m_data->final_x[pidx] = fx;
            m_data->final_y[pidx] = fy;
            m_data->final_z[pidx] = fz;
            m_data->final_vx[pidx] = fvx;
            m_data->final_vy[pidx] = fvy;
            m_data->final_vz[pidx] = fvz;
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
                // into the m_final_* vectors.
                dense_propagate(static_cast<double>(final_t - orig_t));

                // NOTE: everything noexcept from now on.

                // Update the time coordinate.
                m_data->time = final_t;

                // Swap in the updated state.
                m_x.swap(m_data->final_x);
                m_y.swap(m_data->final_y);
                m_z.swap(m_data->final_z);
                m_vx.swap(m_data->final_vx);
                m_vy.swap(m_data->final_vy);
                m_vz.swap(m_data->final_vz);

                // NOTE: m_int_info has already been reset by the
                // step() function.

                return outcome::time_limit;
            }
        } else {
            // Successful step with interruption.
            // TODO fix.
            // assert(cur_oc == outcome::interrupt);

            if (m_data->time > final_t) {
                // If the interruption happened *after*
                // final_t, we need to:
                // - roll back the state to final_t,
                // - update the time,
                // - update the interrupt info.

                // Propagate the state of the system up
                // to the final time, writing the new state
                // into the m_final_* vectors.
                dense_propagate(static_cast<double>(final_t - orig_t));

                // NOTE: everything noexcept from now on.

                // Update the time coordinate.
                m_data->time = final_t;

                // Swap in the updated state.
                m_x.swap(m_data->final_x);
                m_y.swap(m_data->final_y);
                m_z.swap(m_data->final_z);
                m_vx.swap(m_data->final_vx);
                m_vy.swap(m_data->final_vy);
                m_vz.swap(m_data->final_vz);

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

} // namespace cascade
