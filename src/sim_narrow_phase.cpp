// Copyright 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
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
#include <memory>
#include <numeric>
#include <optional>
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

#include <cascade/detail/logging_impl.hpp>
#include <cascade/detail/sim_data.hpp>
#include <cascade/sim.hpp>

#include "detail/ival.hpp"

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
// It will return:
// - 0 if val is 0,
// - +-1 if val is greater/less than zero.
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
    using retval_t = decltype(a[0] * x);

    retval_t ret(a[n]);

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
        return std::tuple{static_cast<T>(0), errno};
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

// Polynomial root finding routine, extracted for re-use.
// NOTE: in the future we can get rid of the DT template parameter
// if/when we move away from a double-length representation of tcoords.
template <typename T, typename Isol, typename Wlist, typename FexCheck, typename Rtscc, typename Pt1, typename Pidx,
          typename Logger, typename CollVec, typename DT, typename PWrap, typename RIsoCache>
void run_poly_root_finding(const T *poly, std::uint32_t order, T rf_int, Isol &isol, Wlist &wlist, FexCheck *fex_check,
                           Rtscc *rtscc, Pt1 *pt1, Pidx pi, Pidx pj, Logger *logger, int direction, CollVec &coll_vec,
                           const DT &lb_rf, PWrap &tmp, PWrap &tmp1, PWrap &tmp2, RIsoCache &r_iso_cache)
{
    assert(direction == 0 || direction == 1 || direction == -1);

    // Run the fast exclusion check.
    std::uint32_t fex_check_res = 0, back_flag = 0;
    fex_check(poly, &rf_int, &back_flag, &fex_check_res);

    if (fex_check_res != 0u) {
        return;
    }

    // Fast exclusion check failed, we need to run the real root isolation algorithm.

    // Clear out the list of isolating intervals.
    isol.clear();

    // Reset the working list.
    wlist.clear();

    // Helper to add a detected root to the global vector of collisions.
    // NOTE: the root here is expected to be already rescaled
    // to the [0, rf_int) range.
    auto add_root = [&](double root) {
        // NOTE: we do one last check on the root in order to
        // avoid non-finite times.
        if (!std::isfinite(root)) {
            // LCOV_EXCL_START
            logger->warn("Polynomial root finding produced a non-finite root of {} - "
                         "skipping the collision/conjunction between particles {} and {}",
                         root, pi, pj);
            return;
            // LCOV_EXCL_STOP
        }

        // Check the direction, if needed.
        bool accept_root = true;
        if (direction != 0) {
            // Evaluate the derivative and its absolute value.
            const auto der = poly_eval_1(poly, root, order);

            // Check it before proceeding.
            if (!std::isfinite(der)) {
                // LCOV_EXCL_START
                logger->warn("Polynomial root finding produced the root {} with "
                             "nonfinite derivative {} - "
                             "skipping the collision/conjunction between particles {} and {}",
                             root, der, pi, pj);
                return;
                // LCOV_EXCL_STOP
            }

            // Compute sign of the derivative.
            const auto d_sgn = sgn(der);

            // Accept the root only if the sign of the derivative
            // matches the direction.
            if (d_sgn != direction) {
                accept_root = false;
            }
        }

        if (accept_root) {
            // Compute the time coordinate of the collision with respect
            // to the lb_rf offset.
            const auto tcoll = static_cast<double>(lb_rf + root);

            if (!std::isfinite(tcoll)) {
                // LCOV_EXCL_START
                logger->warn("Polynomial root finding produced a non-finite collision/conjunction time of {} - "
                             "skipping the collision/conjunction between particles {} and {}",
                             tcoll, pi, pj);
                return;
                // LCOV_EXCL_STOP
            }

            // Add it.
            coll_vec.emplace_back(pi, pj, tcoll);
        }
    };

    // Rescale ss_diff so that the range [0, rf_int)
    // becomes [0, 1), and write the resulting polynomial into tmp.
    // NOTE: we assume a valid tmp has been passed to this function.
    // Below, tmp will first be moved into wlist (thus rendering
    // it invalid) but it will immediately be revived at the
    // first iteration of the do/while loop. Thus, when we get
    // here again, tmp will be again in a well-formed state.
    assert(!tmp.v.empty());             // LCOV_EXCL_LINE
    assert(tmp.v.size() - 1u == order); // LCOV_EXCL_LINE
    poly_rescale(tmp.v.data(), poly, rf_int, order);

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
        std::uint32_t n_sc = 0;
        rtscc(tmp1.v.data(), tmp2.v.data(), &n_sc, tmp.v.data());

        if (n_sc == 1u) {
            // Found isolating interval, add it to isol.
            isol.emplace_back(lb, ub);
        } else if (n_sc > 1u) {
            // No isolating interval found, bisect.

            // First we transform q into 2**n * q(x/2) and store the result
            // into tmp1.
            poly_rescale_p2(tmp1.v.data(), tmp.v.data(), order);
            // Then we take tmp1 and translate it to produce 2**n * q((x+1)/2).
            pt1(tmp2.v.data(), tmp1.v.data());

            // Finally we add tmp1 and tmp2 to the working list.
            const auto mid = (lb + ub) / 2;

            wlist.emplace_back(lb, mid, std::move(tmp1));
            // Revive tmp1.
            tmp1 = PWrap(r_iso_cache, order);

            wlist.emplace_back(mid, ub, std::move(tmp2));
            // Revive tmp2.
            tmp2 = PWrap(r_iso_cache, order);
        }

        // LCOV_EXCL_START
        // We want to put limits in order to avoid an endless loop when the algorithm fails.
        // The first check is on the working list size and it is based
        // on heuristic observation of the algorithm's behaviour in pathological
        // cases. The second check is that we cannot possibly find more isolating
        // intervals than the degree of the polynomial.
        if (wlist.size() > 250u || isol.size() > order) {
            logger->warn("The polynomial root isolation algorithm failed during collision/conjunction "
                         "detection: the working list size is {} and the number of isolating intervals is {}",
                         wlist.size(), isol.size());

            loop_failed = true;

            break;
        }
        // LCOV_EXCL_STOP

    } while (!wlist.empty());

    // Don't do root finding if the loop failed,
    // or if the list of isolating intervals is empty.
    if (!isol.empty() && !loop_failed) {
        // Reconstruct a version of the original polynomial
        // in which the range [0, rf_int) is rescaled to [0, 1). We need
        // to do root finding on the rescaled polynomial because the
        // isolating intervals are also rescaled to [0, 1).
        // NOTE: tmp1 was either created with the correct size outside this
        // function, or it was re-created in the bisection above.
        poly_rescale(tmp1.v.data(), poly, rf_int, order);

        // Run the root finding in the isolating intervals.
        for (auto &[lb, ub] : isol) {
            // Run the root finding.
            const auto [root, cflag] = bracketed_root_find(tmp1.v.data(), order, lb, ub);

            if (cflag == 0) {
                // Root finding finished successfully, record the root.
                // The found root needs to be rescaled by h.
                add_root(root * rf_int);
            } else {
                // LCOV_EXCL_START
                // Root finding encountered some issue. Ignore the
                // root and log the issue.
                if (cflag == -1) {
                    logger->warn("Polynomial root finding during collision/conjunction detection failed "
                                 "due to too many iterations");
                } else {
                    logger->warn("Polynomial root finding during collision/conjunction detection "
                                 "returned a nonzero errno with code '{}'",
                                 cflag);
                }
                // LCOV_EXCL_STOP
            }
        }
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
    const auto pta_cfunc = m_data->pta_cfunc;
    const auto pssdiff3_cfunc = m_data->pssdiff3_cfunc;
    const auto fex_check = m_data->fex_check;
    const auto rtscc = m_data->rtscc;
    const auto pt1 = m_data->pt1;
    const auto conj_thresh2 = m_conj_thresh * m_conj_thresh;

    if (!std::isfinite(conj_thresh2)) {
        // LCOV_EXCL_START
        throw std::invalid_argument(
            fmt::format("A conjunction threshold of {} is too large and results in an overflow error", m_conj_thresh));
        // LCOV_EXCL_STOP
    }

    const auto with_conj = (m_conj_thresh != 0);

    // The time coordinate at the beginning of
    // the superstep.
    const auto init_time = m_data->time;

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

            // Fetch a reference to the detected conjunctions vector
            // for the current chunk and clear it out.
            auto &cl_conj_vec = m_data->conj_vecs[chunk_idx];
            cl_conj_vec.clear();

            // The time coordinate, relative to init_time, of
            // the chunk's begin/end.
            const auto [c_begin, c_end] = m_data->get_chunk_begin_end(chunk_idx, m_ct);
            const auto chunk_begin = dfloat(c_begin);
            const auto chunk_end = dfloat(c_end);

            // Iterate over all collisions.
            oneapi::tbb::parallel_for(oneapi::tbb::blocked_range(bpc.begin(), bpc.end()), [&](const auto &rn) {
                // Fetch the polynomial caches.
                std::unique_ptr<sim_data::np_data> pcaches;

                if (np_cache.try_pop(pcaches)) {
#if !defined(NDEBUG)
                    assert(pcaches);

                    for (auto &v : pcaches->dist2) {
                        assert(v.size() == order + 1u);
                    }

                    using safe_size_t = boost::safe_numerics::safe<decltype(pcaches->diff_input.size())>;
                    assert(pcaches->diff_input.size() == (order + 1u) * safe_size_t(6));
#endif
                } else {
                    SPDLOG_LOGGER_DEBUG(logger, "Creating new local polynomials for narrow phase collision detection");

                    // Init pcaches.
                    pcaches = std::make_unique<sim_data::np_data>();

                    for (auto &v : pcaches->dist2) {
                        v.resize(boost::numeric_cast<decltype(v.size())>(order + 1u));
                    }

                    using safe_size_t = boost::safe_numerics::safe<decltype(pcaches->diff_input.size())>;
                    pcaches->diff_input.resize((order + 1u) * safe_size_t(6));
                }

                // Cache a few quantities.
                auto &[xi_temp, yi_temp, zi_temp, xj_temp, yj_temp, zj_temp, ss_diff, ss_diff_der] = pcaches->dist2;
                auto &diff_input = pcaches->diff_input;
                auto &wlist = pcaches->wlist;
                auto &isol = pcaches->isol;
                auto &r_iso_cache = pcaches->r_iso_cache;
                auto &tmp_conj_vec = pcaches->tmp_conj_vec;
                auto &local_conj_vec = pcaches->local_conj_vec;

                // Prepare the local conjunction vector.
                local_conj_vec.clear();

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
                        const auto pi = pc.first;
                        const auto pj = pc.second;

                        assert(pi != pj);

                        // Fetch a reference to the substep data
                        // for the two particles.
                        const auto &sd_i = s_data[pi];
                        const auto &sd_j = s_data[pj];

                        // Fetch views for reading the Taylor coefficients
                        // for the two particles.
                        using tc_size_t = decltype(sd_i.tcs.size());
                        stdex::mdspan tcs_i(sd_i.tcs.data(),
                                            stdex::extents<tc_size_t, stdex::dynamic_extent, 7u, stdex::dynamic_extent>(
                                                sd_i.tcoords.size(), order + 1u));
                        stdex::mdspan tcs_j(sd_j.tcs.data(),
                                            stdex::extents<tc_size_t, stdex::dynamic_extent, 7u, stdex::dynamic_extent>(
                                                sd_j.tcoords.size(), order + 1u));

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
                                // LCOV_EXCL_START
                                // Bail out in case of errors.
                                logger->warn("During the narrow phase collision detection of particles {} and {}, "
                                             "an invalid time interval for polynomial root finding was generated - the "
                                             "collision will be skipped",
                                             pi, pj);

                                break;
                                // LCOV_EXCL_STOP
                            }

                            // Fetch pointers to the original Taylor polynomials for the two particles.
                            // NOTE: static_cast because:
                            // - we have verified during the propagation that we can safely compute
                            //   differences between iterators of tcoords vectors (see overflow checking in the
                            //   step() function), and
                            // - we know that there are multiple Taylor coefficients being recorded
                            //   for each time coordinate, thus the size type of the vector of Taylor
                            //   coefficients can certainly represent the size of the tcoords vectors.
                            const auto ss_idx_i = static_cast<tc_size_t>(it_i - tcoords_begin_i);
                            const auto ss_idx_j = static_cast<tc_size_t>(it_j - tcoords_begin_j);

                            const auto *poly_xi = &tcs_i(ss_idx_i, 0, 0);
                            const auto *poly_yi = &tcs_i(ss_idx_i, 1, 0);
                            const auto *poly_zi = &tcs_i(ss_idx_i, 2, 0);

                            const auto *poly_xj = &tcs_j(ss_idx_j, 0, 0);
                            const auto *poly_yj = &tcs_j(ss_idx_j, 1, 0);
                            const auto *poly_zj = &tcs_j(ss_idx_j, 2, 0);

                            // Perform the translations, if needed.
                            // NOTE: perhaps we can write a dedicated function
                            // that does the translation for all 3 coordinates
                            // at once, for better performance?
                            // NOTE: need to re-assign the poly_*i pointers if the
                            // translation happens, otherwise we can keep the pointer
                            // to the original polynomials.
                            if (delta_i != 0) {
                                pta_cfunc(xi_temp.data(), poly_xi, &delta_i);
                                poly_xi = xi_temp.data();
                                pta_cfunc(yi_temp.data(), poly_yi, &delta_i);
                                poly_yi = yi_temp.data();
                                pta_cfunc(zi_temp.data(), poly_zi, &delta_i);
                                poly_zi = zi_temp.data();
                            }

                            if (delta_j != 0) {
                                pta_cfunc(xj_temp.data(), poly_xj, &delta_j);
                                poly_xj = xj_temp.data();
                                pta_cfunc(yj_temp.data(), poly_yj, &delta_j);
                                poly_yj = yj_temp.data();
                                pta_cfunc(zj_temp.data(), poly_zj, &delta_j);
                                poly_zj = zj_temp.data();
                            }

                            // Copy over the data to diff_input.
                            using di_size_t = decltype(diff_input.size());
                            std::copy(poly_xi, poly_xi + (order + 1u), diff_input.data());
                            std::copy(poly_yi, poly_yi + (order + 1u), diff_input.data() + (order + 1u));
                            std::copy(poly_zi, poly_zi + (order + 1u),
                                      diff_input.data() + static_cast<di_size_t>(2) * (order + 1u));
                            std::copy(poly_xj, poly_xj + (order + 1u),
                                      diff_input.data() + static_cast<di_size_t>(3) * (order + 1u));
                            std::copy(poly_yj, poly_yj + (order + 1u),
                                      diff_input.data() + static_cast<di_size_t>(4) * (order + 1u));
                            std::copy(poly_zj, poly_zj + (order + 1u),
                                      diff_input.data() + static_cast<di_size_t>(5) * (order + 1u));

                            // We can now construct the polynomial for the
                            // square of the distance.
                            auto *ss_diff_ptr = ss_diff.data();
                            pssdiff3_cfunc(ss_diff_ptr, diff_input.data(), nullptr);

                            // Remember the original constant term of the polynomial.
                            const auto orig_const_cf = ss_diff_ptr[0];

                            // Step 1: detect physical collision.
                            // Modify the constant term of the polynomial to account for
                            // particle sizes.
                            ss_diff_ptr[0] -= (p_rad_i + p_rad_j) * (p_rad_i + p_rad_j);

                            // Run polynomial root finding.
                            detail::run_poly_root_finding(ss_diff_ptr, order, rf_int, isol, wlist, fex_check, rtscc,
                                                          pt1, pi, pj, logger, -1, m_data->coll_vec, lb_rf, tmp, tmp1,
                                                          tmp2, r_iso_cache);

                            // Step 2: do conjunction tracking, if requested.
                            if (with_conj) {
                                // Restore the original constant term of the polynomial.
                                ss_diff_ptr[0] = orig_const_cf;

                                // Evaluate the dist2 polynomial in the [0, rf_int) interval.
                                const auto dist2_ieval = detail::poly_eval(ss_diff_ptr, detail::ival(0, rf_int), order);

                                if (!std::isfinite(dist2_ieval.lower) || !std::isfinite(dist2_ieval.upper)) {
                                    // LCOV_EXCL_START
                                    logger->warn("Non-finite value(s) detected during conjunction tracking for "
                                                 "particles {} and {} - the conjunction will not be tracked",
                                                 pi, pj);

                                    break;
                                    // LCOV_EXCL_STOP
                                }

                                if (dist2_ieval.lower < conj_thresh2) {
                                    // The mutual distance between the particles might end up being
                                    // less than the conjunction threshold during the current time interval.
                                    // This means that a conjunction *may* happen.

                                    // Compute the time derivative of the dist2 poly.
                                    auto *ss_diff_der_ptr = ss_diff_der.data();
                                    for (std::uint32_t i = 0; i < order; ++i) {
                                        ss_diff_der_ptr[i] = (i + 1u) * ss_diff_ptr[i + 1u];
                                    }
                                    // NOTE: the highest-order term needs to be set to zero manually.
                                    ss_diff_der_ptr[order] = 0;

                                    // Prepare tmp_conj_vec.
                                    tmp_conj_vec.clear();

                                    // Run polynomial root finding to detect conjunctions.
                                    detail::run_poly_root_finding(
                                        ss_diff_der_ptr, order, rf_int, isol, wlist, fex_check, rtscc, pt1, pi, pj,
                                        logger,
                                        // NOTE: positive direction to detect only distance minima.
                                        1, tmp_conj_vec,
                                        // NOTE: invoke with lb_rf = 0 so that we get the
                                        // conjunction time wrt the current time interval,
                                        // rather than wrt the beginning of the superstep.
                                        dfloat(0.), tmp, tmp1, tmp2, r_iso_cache);

                                    // For each detected conjunction, we need to:
                                    // - verify that indeed the conjunction happens below
                                    //   the threshold,
                                    // - compute the conjunction distance and absolute
                                    //   time coordinate.
                                    for (const auto &[_1, _2, conj_tm] : tmp_conj_vec) {
                                        assert(_1 == pi);
                                        assert(_2 == pj);

                                        // Compute the conjunction distance square.
                                        const auto conj_dist2 = detail::poly_eval(ss_diff_ptr, conj_tm, order);

                                        if (!std::isfinite(conj_dist2)) {
                                            // LCOV_EXCL_START
                                            logger->warn(
                                                "A non-finite conjunction distance square of {} was computed for the "
                                                "particles at indices {} and {}, the conjunction will be ignored",
                                                conj_dist2, pi, pj);
                                            continue;
                                            // LCOV_EXCL_STOP
                                        }

                                        if (conj_dist2 < conj_thresh2) {
                                            local_conj_vec.emplace_back(
#if defined(__clang__)
                                                conjunction {
#endif
                                                    pi, pj,
                                                        // NOTE: we want to store here the absolute
                                                        // time coordinate of the conjunction. conj_tm
                                                        // is a time coordinate relative to the root
                                                        // finding interval, so we need to first refer it
                                                        // to the beginning of the superstep, and then,
                                                        // finally to the absolute time coordinate.
                                                        static_cast<double>(init_time + (lb_rf + conj_tm)),
                                                        // NOTE: conj_dist2 is finite but it could still
                                                        // be negative due to floating-point rounding
                                                        // (e.g., zero-distance conjunctions). Ensure
                                                        // we do not produce NaN here.
                                                        std::sqrt(std::max(conj_dist2, 0.))
#if defined(__clang__)
                                                }
#endif
                                            );
                                        } else {
                                            SPDLOG_LOGGER_DEBUG(logger, "Conjunction ignored because the conjunction "
                                                                        "distance is less than the threshold");
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

                // Atomically merge the local conjunction vector into the chunk-specific one.
                if (with_conj) {
                    cl_conj_vec.grow_by(local_conj_vec.begin(), local_conj_vec.end());
                }

                // Put the polynomials back into the caches.
                np_cache.push(std::move(pcaches));
            });
        }
    });

    // NOTE: this is used only for logging purposes.
    std::optional<decltype(m_det_conj->size())> n_det_conjs;

    if (with_conj) {
        // If conjunction detection is active, we want to prepare
        // the global conjunction vector for the new detected conjunctions
        // which are currently stored in m_data->conj_vecs. The objective
        // is to avoid reallocating in the step() function, where
        // we need the noexcept guarantee.

        // We begin by determining how many conjunctions were detected.
        // NOTE: perhaps determining n_new_conj can be done in parallel
        // earlier. We just need to take care of avoiding overflow somehow.
        using safe_size_t = boost::safe_numerics::safe<decltype(m_det_conj->size())>;
        const auto n_new_conj = std::accumulate(m_data->conj_vecs.begin(), m_data->conj_vecs.end(), safe_size_t(0),
                                                [](const auto &acc, const auto &cur) { return acc + cur.size(); });

        // Do we have enough storage in m_det_conj to store the new conjunctions?
        if (m_det_conj->size() + n_new_conj > m_det_conj->capacity()) {
            // m_det_conj cannot store the new conjunctions without reallocating.
            // We thus prepare a new vector with twice the needed capacity.
            std::vector<conjunction> new_det_conj;
            new_det_conj.reserve(2u * (m_det_conj->size() + n_new_conj));

            // Copy over the existing conjunctions.
            new_det_conj.insert(new_det_conj.end(), m_det_conj->begin(), m_det_conj->end());

            // Assign the new conjunction vector.
            m_det_conj = std::make_shared<std::vector<conjunction>>(std::move(new_det_conj));
        }

        // Set the logging variable.
        n_det_conjs.emplace(n_new_conj);
    }

    logger->trace("Narrow phase collision detection time: {}s", sw);
    logger->trace("Total number of collisions detected: {}", m_data->coll_vec.size());

    if (n_det_conjs) {
        logger->trace("Total number of conjunctions detected: {}", *n_det_conjs);
    }
}

} // namespace cascade
