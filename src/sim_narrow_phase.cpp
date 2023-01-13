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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include <boost/cstdint.hpp>
#include <boost/math/policies/policy.hpp>
#include <boost/math/tools/toms748_solve.hpp>
#include <boost/numeric/conversion/cast.hpp>
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

// Generic branchless sign function.
template <typename T>
int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}

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

// Evaluate the first derivative of a polynomial.
// Requires random-access iterator.
template <typename InputIt, typename T>
auto poly_eval_1(InputIt a, T x, std::uint32_t n)
{
    assert(n >= 2u); // LCOV_EXCL_LINE

    // Init the return value.
    auto ret1 = a[n] * n;

    for (std::uint32_t i = 1; i < n; ++i) {
        ret1 = a[n - i] * (n - i) + ret1 * x;
    }

    return ret1;
}

// Given an input polynomial a(x), substitute
// x with x_1 * scal and write to ret the resulting
// polynomial in the new variable x_1. Requires
// random-access iterators.
// NOTE: aliasing allowed.
template <typename OutputIt, typename InputIt, typename T>
void poly_rescale(OutputIt ret, InputIt a, const T &scal, std::uint32_t n)
{
    T cur_f(1);

    for (std::uint32_t i = 0; i <= n; ++i) {
        ret[i] = cur_f * a[i];
        cur_f *= scal;
    }
}

// Transform the polynomial a(x) into 2**n * a(x / 2).
// Requires random-access iterators.
// NOTE: aliasing allowed.
template <typename OutputIt, typename InputIt>
void poly_rescale_p2(OutputIt ret, InputIt a, std::uint32_t n)
{
    using value_type = typename std::iterator_traits<InputIt>::value_type;

    value_type cur_f(1);

    for (std::uint32_t i = 0; i <= n; ++i) {
        ret[n - i] = cur_f * a[n - i];
        cur_f *= 2;
    }
}

// Find the only existing root for the polynomial poly of the given order
// existing in [lb, ub).
template <typename T>
std::tuple<T, int> bracketed_root_find(const T *poly, std::uint32_t order, T lb, T ub)
{
    using std::isfinite;
    using std::nextafter;

    // NOTE: the Boost root finding routine searches in a closed interval,
    // but the goal here is to find a root in [lb, ub). Thus, we move ub
    // one position down so that it is not considered in the root finding routine.
    if (isfinite(lb) && isfinite(ub) && ub > lb) {
        ub = nextafter(ub, lb);
    }

    // NOTE: perhaps this should depend on T? E.g., we could use the number
    // of binary digits in the significand.
    constexpr boost::uintmax_t iter_limit = 100;
    auto max_iter = iter_limit;

    // Ensure that root finding does not throw on error,
    // rather it will write something to errno instead.
    // https://www.boost.org/doc/libs/1_75_0/libs/math/doc/html/math_toolkit/pol_tutorial/namespace_policies.html
    using boost::math::policies::domain_error;
    using boost::math::policies::errno_on_error;
    using boost::math::policies::evaluation_error;
    using boost::math::policies::overflow_error;
    using boost::math::policies::pole_error;
    using boost::math::policies::policy;

    using pol = policy<domain_error<errno_on_error>, pole_error<errno_on_error>, overflow_error<errno_on_error>,
                       evaluation_error<errno_on_error>>;

    // Clear out errno before running the root finding.
    errno = 0;

    // Run the root finder.
    const auto p = boost::math::tools::toms748_solve([poly, order](T x) { return poly_eval(poly, x, order); }, lb, ub,
                                                     boost::math::tools::eps_tolerance<T>(), max_iter, pol{});
    const auto ret = (p.first + p.second) / 2;

    if (errno > 0) {
        // Some error condition arose during root finding,
        // return zero and errno.
        return std::tuple{T(0), errno};
    }

    if (max_iter < iter_limit) {
        // Root finding terminated within the
        // iteration limit, return ret and success.
        return std::tuple{ret, 0};
    } else {
        // LCOV_EXCL_START
        // Root finding needed too many iterations,
        // return the (possibly wrong) result
        // and flag -1.
        return std::tuple{ret, -1};
        // LCOV_EXCL_STOP
    }
}

} // namespace

} // namespace detail

void sim::sim_data::np_data::pwrap::back_to_cache()
{
    // NOTE: the cache does not allow empty vectors.
    if (!v.empty()) {
        assert(pc.empty() || pc[0].size() == v.size());

        // Move v into the cache.
        pc.push_back(std::move(v));
    }
}

std::vector<double> sim::sim_data::np_data::pwrap::get_poly_from_cache(std::uint32_t n)
{
    if (pc.empty()) {
        // No polynomials are available, create a new one.
        return std::vector<double>(boost::numeric_cast<std::vector<double>::size_type>(n + 1u));
    } else {
        // Extract an existing polynomial from the cache.
        auto retval = std::move(pc.back());
        pc.pop_back();

        return retval;
    }
}

sim::sim_data::np_data::pwrap::pwrap(std::vector<std::vector<double>> &cache, std::uint32_t n)
    : pc(cache), v(get_poly_from_cache(n))
{
}

sim::sim_data::np_data::pwrap::pwrap(pwrap &&other) noexcept : pc(other.pc), v(std::move(other.v))
{
    // Make sure we moved from a valid pwrap.
    assert(!v.empty()); // LCOV_EXCL_LINE
}

sim::sim_data::np_data::pwrap &sim::sim_data::np_data::pwrap::operator=(pwrap &&other) noexcept
{
    // Disallow self move.
    assert(this != &other); // LCOV_EXCL_LINE

    // Make sure the polyomial caches match.
    assert(&pc == &other.pc); // LCOV_EXCL_LINE

    // Make sure we are not moving from an
    // invalid pwrap.
    assert(!other.v.empty()); // LCOV_EXCL_LINE

    // Put the current v in the cache.
    back_to_cache();

    // Do the move-assignment.
    v = std::move(other.v);

    return *this;
}

sim::sim_data::np_data::pwrap::~pwrap()
{
    // Put the current v in the cache.
    back_to_cache();
}

// Narrow phase collision detection: the trajectories
// of the particle pairs identified during broad
// phase collision detection are tested for intersection
// using polynomial root finding.
void sim::narrow_phase_parallel()
{
    namespace hy = heyoka;
    using dfloat = hy::detail::dfloat<double>;
    namespace stdex = std::experimental;

    spdlog::stopwatch sw;

    auto *logger = detail::get_logger();

    // Cache a few bits.
    const auto nchunks = m_data->nchunks;
    const auto order = m_data->s_ta.get_order();
    const auto &s_data = m_data->s_data;
    const auto pta = m_data->pta;
    const auto pssdiff3 = m_data->pssdiff3;
    const auto fex_check = m_data->fex_check;
    const auto rtscc = m_data->rtscc;
    const auto pt1 = m_data->pt1;

    // Reset the collision vector.
    m_data->coll_vec.clear();

    // Fetch a view on the state vector in order to
    // access the particles' sizes.
    stdex::mdspan sv(std::as_const(m_state)->data(),
                     stdex::extents<size_type, stdex::dynamic_extent, 7u>(get_nparts()));

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(0u, nchunks), [&](const auto &range) {
        for (auto chunk_idx = range.begin(); chunk_idx != range.end(); ++chunk_idx) {
            // Fetch a reference to the chunk-specific broad
            // phase collision vector.
            const auto &bpc = m_data->bp_coll[chunk_idx];

            // Fetch a reference to the chunk-specific caches.
            auto &np_cache_ptr = m_data->np_caches[chunk_idx];
            // NOTE: the pointer will require initialisation the first time
            // it is used.
            if (!np_cache_ptr) {
                np_cache_ptr = std::make_unique<typename decltype(m_data->np_caches)::value_type::element_type>();
            }
            auto &np_cache = *np_cache_ptr;

            // The time coordinate, relative to init_time, of
            // the chunk's begin/end.
            const auto [c_begin, c_end] = m_data->get_chunk_begin_end(chunk_idx, m_ct);
            const auto chunk_begin = dfloat(c_begin);
            const auto chunk_end = dfloat(c_end);

#if !defined(NDEBUG)
            // Counter for the number of failed fast exclusion checks.
            std::atomic<std::size_t> n_ffex(0);
#endif

            // Iterate over all collisions.
            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(bpc.begin(), bpc.end()), [&](const auto &rn) {
#if !defined(NDEBUG)
                // Local version of n_ffex.
                std::size_t local_n_ffex = 0;
#endif

                // Fetch the polynomial caches.
                std::unique_ptr<sim_data::np_data> pcaches;

                if (np_cache.try_pop(pcaches)) {
#if !defined(NDEBUG)
                    assert(pcaches);

                    for (auto &v : pcaches->dist2) {
                        assert(v.size() == order + 1u);
                    }
#endif
                } else {
                    SPDLOG_LOGGER_DEBUG(logger, "Creating new local polynomials for narrow phase collision detection");

                    // Init pcaches.
                    pcaches = std::make_unique<sim_data::np_data>();

                    for (auto &v : pcaches->dist2) {
                        v.resize(boost::numeric_cast<decltype(v.size())>(order + 1u));
                    }
                }

                // Cache a few quantities.
                auto &[xi_temp, yi_temp, zi_temp, xj_temp, yj_temp, zj_temp, ss_diff] = pcaches->dist2;
                auto &wlist = pcaches->wlist;
                auto &isol = pcaches->isol;
                auto &r_iso_cache = pcaches->r_iso_cache;

                // NOTE: bracket further so that the pwrap objects
                // are destroyed *before* pcaches is moved into np_cache.
                // This is essential, because otherwise these pwraps
                // will be destroyed *after* pcaches has been already pushed
                // back to the concurrent queue and possibly already in use by
                // another thread.
                {
                    // Temporary polynomials used in the bisection loop.
                    using pwrap = sim_data::np_data::pwrap;
                    pwrap tmp1(r_iso_cache, order), tmp2(r_iso_cache, order), tmp(r_iso_cache, order);

                    for (const auto &pc : rn) {
                        const auto [pi, pj] = pc;

                        assert(pi != pj);

                        // Fetch a reference to the substep data
                        // for the two particles.
                        const auto &sd_i = s_data[pi];
                        const auto &sd_j = s_data[pj];

                        // Load the particle radiuses.
                        const auto p_rad_i = sv(pi, 6);
                        const auto p_rad_j = sv(pj, 6);

                        // Cache the range of end times of the substeps.
                        const auto tcoords_begin_i = sd_i.tcoords.begin();
                        const auto tcoords_end_i = sd_i.tcoords.end();

                        const auto tcoords_begin_j = sd_j.tcoords.begin();
                        const auto tcoords_end_j = sd_j.tcoords.end();

                        // Determine, for both particles, the range of substeps
                        // that fully includes the current chunk.
                        // NOTE: same code as in sim_propagate.cpp.
                        const auto ss_it_begin_i = std::upper_bound(tcoords_begin_i, tcoords_end_i, chunk_begin);
                        auto ss_it_end_i = std::lower_bound(ss_it_begin_i, tcoords_end_i, chunk_end);
                        ss_it_end_i += (ss_it_end_i != tcoords_end_i);

                        const auto ss_it_begin_j = std::upper_bound(tcoords_begin_j, tcoords_end_j, chunk_begin);
                        auto ss_it_end_j = std::lower_bound(ss_it_begin_j, tcoords_end_j, chunk_end);
                        ss_it_end_j += (ss_it_end_j != tcoords_end_j);

                        // Iterate until we get to the end of at least one range.
                        // NOTE: if either range is empty, this loop is never entered.
                        // This should never happen, but see the comments in
                        // dense_propagate().
                        for (auto it_i = ss_it_begin_i, it_j = ss_it_begin_j;
                             it_i != ss_it_end_i && it_j != ss_it_end_j;) {
                            // Initial time coordinates of the substeps of i and j,
                            // relative to init_time.
                            const auto ss_start_i
                                = (it_i == tcoords_begin_i) ? hy::detail::dfloat<double>(0) : *(it_i - 1);
                            const auto ss_start_j
                                = (it_j == tcoords_begin_j) ? hy::detail::dfloat<double>(0) : *(it_j - 1);

                            // Determine the intersections of the two substeps
                            // with the current chunk.
                            // NOTE: min/max is fine here: values in tcoords are always checked
                            // for finiteness, chunk_begin/end are also checked in
                            // get_chunk_begin_end().
                            const auto lb_i = std::max(chunk_begin, ss_start_i);
                            const auto ub_i = std::min(chunk_end, *it_i);
                            const auto lb_j = std::max(chunk_begin, ss_start_j);
                            const auto ub_j = std::min(chunk_end, *it_j);

                            // Determine the intersection between the two intervals
                            // we just computed. This will be the time range
                            // within which we need to do polynomial root finding.
                            // NOTE: at this stage lb_rf/ub_rf are still time coordinates wrt
                            // init_time.
                            // NOTE: min/max fine here, all quantities are safe.
                            const auto lb_rf = std::max(lb_i, lb_j);
                            const auto ub_rf = std::min(ub_i, ub_j);

                            // The Taylor polynomials for the two particles are time polynomials
                            // in which time is counted from the beginning of the substep. In order to
                            // create the polynomial representing the distance square, we need first to
                            // translate the polynomials of both particles so that they refer to a
                            // common time coordinate, the time elapsed from lb_rf.

                            // Compute the translation amount for the two particles.
                            const auto delta_i = static_cast<double>(lb_rf - ss_start_i);
                            const auto delta_j = static_cast<double>(lb_rf - ss_start_j);

                            // Compute the time interval within which we will be performing root finding.
                            const auto rf_int = static_cast<double>(ub_rf - lb_rf);

                            // Do some checking before moving on.
                            if (!std::isfinite(delta_i) || !std::isfinite(delta_j) || !std::isfinite(rf_int)
                                || delta_i < 0 || delta_j < 0 || rf_int < 0) {
                                // Bail out in case of errors.
                                logger->warn("During the narrow phase collision detection of particles {} and {}, "
                                             "an invalid time interval for polynomial root finding was generated - the "
                                             "collision will be skipped",
                                             pi, pj);

                                break;
                            }

                            // Fetch pointers to the original Taylor polynomials for the two particles.
                            // NOTE: static_cast because:
                            // - we have verified during the propagation that we can safely compute
                            //   differences between iterators of tcoords vectors (see overflow checking in the
                            //   step() function), and
                            // - we know that there are multiple Taylor coefficients being recorded
                            //   for each time coordinate, thus the size type of the vector of Taylor
                            //   coefficients can certainly represent the size of the tcoords vectors.
                            const auto ss_idx_i = static_cast<decltype(s_data[pi].tc_x.size())>(it_i - tcoords_begin_i);
                            const auto ss_idx_j = static_cast<decltype(s_data[pj].tc_x.size())>(it_j - tcoords_begin_j);

                            const auto *poly_xi = s_data[pi].tc_x.data() + ss_idx_i * (order + 1u);
                            const auto *poly_yi = s_data[pi].tc_y.data() + ss_idx_i * (order + 1u);
                            const auto *poly_zi = s_data[pi].tc_z.data() + ss_idx_i * (order + 1u);

                            const auto *poly_xj = s_data[pj].tc_x.data() + ss_idx_j * (order + 1u);
                            const auto *poly_yj = s_data[pj].tc_y.data() + ss_idx_j * (order + 1u);
                            const auto *poly_zj = s_data[pj].tc_z.data() + ss_idx_j * (order + 1u);

                            // Perform the translations, if needed.
                            // NOTE: perhaps we can write a dedicated function
                            // that does the translation for all 3 coordinates
                            // at once, for better performance?
                            if (delta_i != 0) {
                                poly_xi = pta(xi_temp.data(), poly_xi, delta_i);
                                poly_yi = pta(yi_temp.data(), poly_yi, delta_i);
                                poly_zi = pta(zi_temp.data(), poly_zi, delta_i);
                            }

                            if (delta_j != 0) {
                                poly_xj = pta(xj_temp.data(), poly_xj, delta_j);
                                poly_yj = pta(yj_temp.data(), poly_yj, delta_j);
                                poly_zj = pta(zj_temp.data(), poly_zj, delta_j);
                            }

                            // We can now construct the polynomial for the
                            // square of the distance.
                            auto *ss_diff_ptr = ss_diff.data();
                            pssdiff3(ss_diff_ptr, poly_xi, poly_yi, poly_zi, poly_xj, poly_yj, poly_zj);

                            // Modify the constant term of the polynomial to account for
                            // particle sizes.
                            ss_diff_ptr[0] -= (p_rad_i + p_rad_j) * (p_rad_i + p_rad_j);

                            // Run the fast exclusion check.
                            std::uint32_t fex_check_res, back_flag = 0;
                            fex_check(ss_diff_ptr, &rf_int, &back_flag, &fex_check_res);
                            if (!fex_check_res) {
                                // Fast exclusion check failed, we need to run the real root isolation algorithm.

#if !defined(NDEBUG)
                                // Update local_n_ffex.
                                ++local_n_ffex;
#endif

                                // Clear out the list of isolating intervals.
                                isol.clear();

                                // Reset the working list.
                                wlist.clear();

                                // Helper to add a detected root to the global vector of collisions.
                                // NOTE: the root here is expected to be already rescaled
                                // to the [0, rf_int) range.
                                auto add_root = [&](double root) {
                                    // NOTE: we do one last check on the root in order to
                                    // avoid non-finite event times.
                                    if (!std::isfinite(root)) {
                                        // LCOV_EXCL_START
                                        logger->warn("Polynomial root finding produced a non-finite root of {} - "
                                                     "skipping the collision between particles {} and {}",
                                                     root, pi, pj);
                                        return;
                                        // LCOV_EXCL_STOP
                                    }

                                    // Evaluate the derivative and its absolute value.
                                    const auto der = detail::poly_eval_1(ss_diff_ptr, root, order);

                                    // Check it before proceeding.
                                    if (!std::isfinite(der)) {
                                        // LCOV_EXCL_START
                                        logger->warn("Polynomial root finding produced the root {} with "
                                                     "nonfinite derivative {} - "
                                                     "skipping the collision between particles {} and {}",
                                                     root, der, pi, pj);
                                        return;
                                        // LCOV_EXCL_STOP
                                    }

                                    // Compute sign of the derivative.
                                    const auto d_sgn = detail::sgn(der);

                                    // Record the collision only if the derivative
                                    // is negative.
                                    if (d_sgn < 0) {
                                        // Compute the time coordinate of the collision with respect
                                        // to the beginning of the superstep.
                                        const auto tcoll = static_cast<double>(lb_rf + root);

                                        if (!std::isfinite(tcoll)) {
                                            // LCOV_EXCL_START
                                            logger->warn(
                                                "Polynomial root finding produced a non-finite collision time of {} - "
                                                "skipping the collision between particles {} and {}",
                                                tcoll, pi, pj);
                                            return;
                                            // LCOV_EXCL_STOP
                                        }

                                        // Add it.
                                        m_data->coll_vec.emplace_back(pi, pj, tcoll);
                                    }
                                };

                                // Rescale ss_diff so that the range [0, rf_int)
                                // becomes [0, 1), and write the resulting polynomial into tmp.
                                // NOTE: at the first iteration (i.e., for the first
                                // substep of the first pair
                                // of particles which requires real root isolation),
                                // tmp has been constructed correctly outside the loop.
                                // Below, tmp will first be moved into wlist (thus rendering
                                // it invalid) but it will immediately be revived at the
                                // first iteration of the do/while loop. Thus, when we get
                                // here again, tmp will be again in a well-formed state.
                                assert(!tmp.v.empty());             // LCOV_EXCL_LINE
                                assert(tmp.v.size() - 1u == order); // LCOV_EXCL_LINE
                                detail::poly_rescale(tmp.v.data(), ss_diff_ptr, rf_int, order);

                                // Place the first element in the working list.
                                wlist.emplace_back(0, 1, std::move(tmp));

                                // Flag to signal that the do-while loop below failed.
                                bool loop_failed = false;

                                do {
                                    // Fetch the current interval and polynomial from the working list.
                                    // NOTE: from now on, tmp contains the polynomial referred
                                    // to as q(x) in the real-root isolation wikipedia page.
                                    // NOTE: q(x) is the transformed polynomial whose roots in the x range [0, 1) we
                                    // will be looking for. lb and ub represent what 0 and 1 correspond to in the
                                    // *original* [0, 1) range.
                                    auto lb = std::get<0>(wlist.back());
                                    auto ub = std::get<1>(wlist.back());
                                    // NOTE: this will either revive an invalid tmp (first iteration),
                                    // or it will replace it with one of the bisecting polynomials.
                                    tmp = std::move(std::get<2>(wlist.back()));
                                    wlist.pop_back();

                                    // Check for a root at the lower bound, which occurs
                                    // if the constant term of the polynomial is zero. We also
                                    // check for finiteness of all the other coefficients, otherwise
                                    // we cannot really claim to have detected a root.
                                    // When we do proper root finding below, the
                                    // algorithm should be able to detect non-finite
                                    // polynomials.
                                    if (tmp.v[0] == 0 // LCOV_EXCL_LINE
                                        && std::all_of(tmp.v.data() + 1, tmp.v.data() + 1 + order,
                                                       [](const auto &x) { return std::isfinite(x); })) {
                                        // NOTE: the original range had been rescaled wrt to rf_int.
                                        // Thus, we need to rescale back when adding the detected
                                        // root.
                                        add_root(lb * rf_int);
                                    }

                                    // Reverse tmp into tmp1, translate tmp1 by 1 with output
                                    // in tmp2, and count the sign changes in tmp2.
                                    std::uint32_t n_sc;
                                    rtscc(tmp1.v.data(), tmp2.v.data(), &n_sc, tmp.v.data());

                                    if (n_sc == 1u) {
                                        // Found isolating interval, add it to isol.
                                        isol.emplace_back(lb, ub);
                                    } else if (n_sc > 1u) {
                                        // No isolating interval found, bisect.

                                        // First we transform q into 2**n * q(x/2) and store the result
                                        // into tmp1.
                                        detail::poly_rescale_p2(tmp1.v.data(), tmp.v.data(), order);
                                        // Then we take tmp1 and translate it to produce 2**n * q((x+1)/2).
                                        pt1(tmp2.v.data(), tmp1.v.data());

                                        // Finally we add tmp1 and tmp2 to the working list.
                                        const auto mid = (lb + ub) / 2;

                                        wlist.emplace_back(lb, mid, std::move(tmp1));
                                        // Revive tmp1.
                                        tmp1 = pwrap(r_iso_cache, order);

                                        wlist.emplace_back(mid, ub, std::move(tmp2));
                                        // Revive tmp2.
                                        tmp2 = pwrap(r_iso_cache, order);
                                    }

                                    // LCOV_EXCL_START
                                    // We want to put limits in order to avoid an endless loop when the algorithm fails.
                                    // The first check is on the working list size and it is based
                                    // on heuristic observation of the algorithm's behaviour in pathological
                                    // cases. The second check is that we cannot possibly find more isolating
                                    // intervals than the degree of the polynomial.
                                    if (wlist.size() > 250u || isol.size() > order) {
                                        logger->warn(
                                            "The polynomial root isolation algorithm failed during collision "
                                            "detection: "
                                            "the working list size is {} and the number of isolating intervals is {}",
                                            wlist.size(), isol.size());

                                        loop_failed = true;

                                        break;
                                    }
                                    // LCOV_EXCL_STOP

                                } while (!wlist.empty());

                                // Don't do root finding if the loop failed,
                                // or if the list of isolating intervals is empty. Just
                                // move to the next substep.
                                if (!isol.empty() && !loop_failed) {
                                    // Reconstruct a version of the original polynomial
                                    // in which the range [0, rf_int) is rescaled to [0, 1). We need
                                    // to do root finding on the rescaled polynomial because the
                                    // isolating intervals are also rescaled to [0, 1).
                                    // NOTE: tmp1 was either created with the correct size outside this
                                    // function, or it was re-created in the bisection above.
                                    detail::poly_rescale(tmp1.v.data(), ss_diff_ptr, rf_int, order);

                                    // Run the root finding in the isolating intervals.
                                    for (auto &[lb, ub] : isol) {
                                        // Run the root finding.
                                        const auto [root, cflag]
                                            = detail::bracketed_root_find(tmp1.v.data(), order, lb, ub);

                                        if (cflag == 0) {
                                            // Root finding finished successfully, record the root.
                                            // The found root needs to be rescaled by h.
                                            add_root(root * rf_int);
                                        } else {
                                            // LCOV_EXCL_START
                                            // Root finding encountered some issue. Ignore the
                                            // root and log the issue.
                                            if (cflag == -1) {
                                                logger->warn(
                                                    "Polynomial root finding during collision detection failed "
                                                    "due to too many iterations");
                                            } else {
                                                logger->warn("Polynomial root finding during collision detection "
                                                             "returned a nonzero errno with message '{}'",
                                                             std::strerror(cflag));
                                            }
                                            // LCOV_EXCL_STOP
                                        }
                                    }
                                }
                            }

                            // Update the substep iterators.
                            if (*it_i < *it_j) {
                                // The substep for particle i ends
                                // before the substep for particle j.
                                ++it_i;
                            } else if (*it_j < *it_i) {
                                // The substep for particle j ends
                                // before the substep for particle i.
                                ++it_j;
                            } else {
                                // Both substeps end at the same time.
                                // This happens at the last substeps of a chunk
                                // or in the very unlikely case in which both
                                // steps end exactly at the same time.
                                ++it_i;
                                ++it_j;
                            }
                        }
                    }
                }

                // Put the polynomials back into the caches.
                np_cache.push(std::move(pcaches));

#if !defined(NDEBUG)
                // Update n_ffex.
                n_ffex.fetch_add(local_n_ffex, std::memory_order::relaxed);
#endif
            });

            SPDLOG_LOGGER_DEBUG(logger,
                                "Number of failed fast exclusion checks for chunk {}: {} vs {} broad phase collisions",
                                chunk_idx, n_ffex.load(std::memory_order::relaxed), bpc.size());
        }
    });

    logger->trace("Narrow phase collision detection time: {}s", sw);
    logger->trace("Total number of collisions detected: {}", m_data->coll_vec.size());
}

void sim::narrow_phase_parallel_new()
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
    m_data->coll_vec2.clear();

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
                                    m_data->coll_vec2.emplace_back(pi, pj, static_cast<double>(tcoll - m_data->time));
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
    logger->trace("Total number of collisions detected: {}", m_data->coll_vec2.size());
}

} // namespace cascade
