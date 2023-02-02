// Copyright 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <boost/safe_numerics/safe_integer.hpp>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <oneapi/tbb/blocked_range.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/parallel_invoke.h>

#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/pow.hpp>
#include <heyoka/math/sum.hpp>
#include <heyoka/math/sum_sq.hpp>
#include <heyoka/taylor.hpp>
#include <heyoka/variable.hpp>

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

// The list of allowed dynamical symbols, first in dynamical order then in alphabetical order.
// NOLINTNEXTLINE(cert-err58-cpp)
const std::array<std::string, 6> allowed_vars = {"x", "y", "z", "vx", "vy", "vz"};

// NOLINTNEXTLINE(cert-err58-cpp)
const std::set<std::string> allowed_vars_alph(allowed_vars.begin(), allowed_vars.end());

} // namespace

} // namespace detail

// Helper to compute the begin and end time coordinates for a chunk within
// a superstep for a given collisional timestep. As usual, the time coordinates
// are referred to the beginning of the superstep.
std::array<double, 2> sim::sim_data::get_chunk_begin_end(unsigned chunk_idx, double ct) const
{
    assert(nchunks > 0u);
    assert(std::isfinite(delta_t) && delta_t > 0);
    assert(std::isfinite(ct) && ct > 0);

    auto cbegin = ct * chunk_idx;
    // NOTE: for the last chunk we force the ending
    // at delta_t.
    auto cend = (chunk_idx == nchunks - 1u) ? delta_t : (ct * (chunk_idx + 1u));

    if (!std::isfinite(cbegin) || !std::isfinite(cend) || !(cend > cbegin) || cbegin < 0 || cend > delta_t) {
        throw std::invalid_argument(fmt::format("Invalid chunk range [{}, {})", cbegin, cend));
    }

    return {cbegin, cend};
}

sim::sim() : sim(std::vector<double>{}, 1) {}

sim::sim(ptag_t, std::vector<double> state, double ct)
    : m_state(std::make_shared<std::vector<double>>(std::move(state))), m_ct(ct),
      m_det_conj(std::make_shared<std::vector<conjunction>>())
{
}

sim::sim(const sim &other)
    : m_state(std::make_shared<std::vector<double>>(*other.m_state)),
      m_pars(std::make_shared<std::vector<double>>(*other.m_pars)), m_ct(other.m_ct), m_n_par_ct(other.m_n_par_ct),
      m_int_info(other.m_int_info), m_reentry_radius(other.m_reentry_radius), m_d_radius(other.m_d_radius),
      m_npars(other.m_npars), m_conj_thresh(other.m_conj_thresh),
      m_det_conj(std::make_shared<std::vector<conjunction>>(*other.m_det_conj)),
      m_min_coll_radius(other.m_min_coll_radius), m_coll_whitelist(other.m_coll_whitelist),
      m_conj_whitelist(other.m_conj_whitelist)
{
    // For m_data, we will be copying only:
    // - the integrator templates,
    // - the llvm state,
    // - the time coordinate.
    // Below, we will also be assigning the function pointers
    // for the jitted functions. The rest of sim_data's members are set up
    // explicitly at the beginning of each timestep.

#if defined(__clang__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

    auto new_data = std::make_unique<sim_data>(
        sim_data{other.m_data->s_ta, other.m_data->b_ta, other.m_data->state, other.m_data->time});

#pragma GCC diagnostic pop

#else

    auto new_data
        = std::make_unique<sim_data>(other.m_data->s_ta, other.m_data->b_ta, other.m_data->state, other.m_data->time);

#endif

    // Need to assign the JIT function pointers.
    new_data->pta_cfunc = reinterpret_cast<decltype(new_data->pta_cfunc)>(new_data->state.jit_lookup("pta_cfunc"));
    new_data->pssdiff3_cfunc
        = reinterpret_cast<decltype(new_data->pssdiff3_cfunc)>(new_data->state.jit_lookup("ssdiff3_cfunc"));
    new_data->fex_check = reinterpret_cast<decltype(new_data->fex_check)>(new_data->state.jit_lookup("fex_check"));
    new_data->rtscc = reinterpret_cast<decltype(new_data->rtscc)>(new_data->state.jit_lookup("poly_rtscc"));
    // NOTE: this is implicitly added by llvm_add_poly_rtscc().
    new_data->pt1 = reinterpret_cast<decltype(new_data->pt1)>(new_data->state.jit_lookup("poly_translate_1"));

    // Assign the new pointer.
    m_data = std::move(new_data);
}

sim::sim(sim &&) noexcept = default;

sim &sim::operator=(const sim &other)
{
    if (this != &other) {
        *this = sim(other);
    }

    return *this;
}

sim &sim::operator=(sim &&) noexcept = default;

sim::~sim() = default;

double sim::get_ct() const
{
    return m_ct;
}

void sim::set_ct(double ct)
{
    if (!std::isfinite(ct) || ct <= 0) {
        throw std::invalid_argument(
            fmt::format("The collisional timestep must be finite and positive, but it is {} instead", ct));
    }

    m_ct = ct;
}

std::uint32_t sim::get_n_par_ct() const
{
    return m_n_par_ct;
}

void sim::set_n_par_ct(std::uint32_t n_par_ct)
{
    if (n_par_ct == 0u) {
        throw std::invalid_argument("The number of collisional timesteps to be processed in parallel cannot be zero");
    }

    m_n_par_ct = n_par_ct;
}

void sim::set_conj_thresh(double conj_thresh)
{
    if (!std::isfinite(conj_thresh) || conj_thresh < 0) {
        throw std::invalid_argument(fmt::format(
            "The conjunction threshold value {} is invalid: it must be finite and non-negative", conj_thresh));
    }

    m_conj_thresh = conj_thresh;
}

std::uint32_t sim::get_npars() const
{
    return m_npars;
}

double sim::get_tol() const
{
    assert(m_data->s_ta.get_tol() == m_data->b_ta.get_tol());

    return m_data->s_ta.get_tol();
}

std::variant<double, std::vector<double>> sim::get_reentry_radius() const
{
    return m_reentry_radius;
}

bool sim::get_high_accuracy() const
{
    assert(m_data->s_ta.get_high_accuracy() == m_data->b_ta.get_high_accuracy());

    return m_data->s_ta.get_high_accuracy();
}

// A helper that validates (and possibly modifies in-place) the input
// array of parameters pars. The validation checks that pars is consistent
// both with m_npars (the number of parameters in the dynamics) and with
// the number of particles in the simulation nparts. Note that nparts
// is passed explicitly to this function (rather than being established via
// get_nparts()) because this function is also used in set_new_state_pars(),
// where nparts is given by the new state vector.
void sim::validate_pars_vector(std::vector<double> &pars, size_type nparts) const
{
    using safe_size_t = boost::safe_numerics::safe<std::vector<double>::size_type>;

    if (m_npars == 0u) {
        // If there are no params in the dynamics, then the array of
        // param values must be empty.
        if (!pars.empty()) {
            throw std::invalid_argument(
                "The input array of parameter values must be empty when the number of parameters "
                "in the dynamics is zero");
        }
    } else {
        if (pars.empty()) {
            // There are parameters in the dynamics but the user did not
            // provide an array of param values (or an empty one as provided).
            // In such a case, zero-init the array of param values with the correct size.
            pars.resize(safe_size_t(nparts) * m_npars);
        } else if (pars.size() % m_npars != 0u || pars.size() / m_npars != nparts) {
            // There are parameters in the dynamics and the user provided
            // an array of param values, but the shape is wrong.
            throw std::invalid_argument(fmt::format("The input array of parameter values must have shape ({}, {}), "
                                                    "but instead its flattened size is {}",
                                                    nparts, m_npars, pars.size()));
        }
    }
}

void sim::remove_particles(std::vector<size_type> idxs)
{
    // Sort the indices.
    std::sort(idxs.begin(), idxs.end());

    // Remove consecutive (adjacent) duplicates.
    idxs.erase(std::unique(idxs.begin(), idxs.end()), idxs.end());

    // Create the new state/pars filtering out
    // the particles in idxs.
    std::vector<double> new_state, new_pars;
    auto idxs_it = idxs.cbegin();
    const auto idxs_end = idxs.cend();

    const auto nparts = get_nparts();
    const auto npars = get_npars();

    for (size_type i = 0; i < nparts; ++i) {
        if (idxs_it != idxs_end && *idxs_it == i) {
            ++idxs_it;
            continue;
        }

        for (auto j = 0u; j < 7u; ++j) {
            new_state.push_back((*m_state)[i * 7u + j]);
        }

        for (std::uint32_t j = 0; j < npars; ++j) {
            new_pars.push_back((*m_pars)[i * npars + j]);
        }
    }

    if (idxs_it != idxs_end) {
        throw std::invalid_argument(
            fmt::format("An invalid vector of indices was passed to the function for particle removal: {}", idxs));
    }

    // NOTE: the new state/pars do not need additional validation.
#if !defined(NDEBUG)

    assert(new_state.size() % 7u == 0u);
    const auto new_nparts = new_state.size() / 7u;

    if (npars == 0u) {
        assert(new_pars.empty());
    } else {
        assert(new_pars.size() % npars == 0u);
        assert(new_pars.size() / npars == new_nparts);
    }

#endif

    // Create and assign the new vectors.
    auto new_st_ptr = std::make_shared<std::vector<double>>(std::move(new_state));
    auto new_pars_ptr = std::make_shared<std::vector<double>>(std::move(new_pars));
    // NOTE: noexcept from here.
    m_state = std::move(new_st_ptr);
    m_pars = std::move(new_pars_ptr);
}

void sim::set_new_state_pars(std::vector<double> new_state, std::vector<double> new_pars)
{
    // Verify the new state.
    verify_state_vector(new_state);

    // Fetch the new number of particles.
    const auto new_nparts = new_state.size() / 7u;

    // Validate/prepare the new parameters vector.
    validate_pars_vector(new_pars, new_nparts);

    // Create and assign the new vectors.
    auto new_st_ptr = std::make_shared<std::vector<double>>(std::move(new_state));
    auto new_pars_ptr = std::make_shared<std::vector<double>>(std::move(new_pars));
    // NOTE: noexcept from here.
    m_state = std::move(new_st_ptr);
    m_pars = std::move(new_pars_ptr);
}

void sim::finalise_ctor(std::vector<std::pair<heyoka::expression, heyoka::expression>> dyn, std::vector<double> pars,
                        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                        std::variant<double, std::vector<double>> reentry_radius, double d_radius, double tol, bool ha,
                        // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
                        std::uint32_t n_par_ct, double conj_thresh, double min_coll_radius, whitelist_t coll_whitelist,
                        whitelist_t conj_whitelist)
{
    namespace hy = heyoka;

    using safe_size_t = boost::safe_numerics::safe<std::vector<double>::size_type>;

    auto *logger = detail::get_logger();

    // Check that the state vector's size is a multiple of 7
    // (i.e., the state vector of each particle contains 7 values:
    // cartesian pos/vel + size).
    // NOTE: this is a bit redundant with the checks run in verify_state_vector(),
    // but it does not matter.
    if (m_state->size() % 7u != 0u) {
        throw std::invalid_argument(
            fmt::format("The size of the state vector is {}, which is not a multiple of 7", m_state->size()));
    }

    // Cache the number of particles.
    const auto nparts = get_nparts();

    // Check the collisional timestep.
    if (!std::isfinite(m_ct) || m_ct <= 0) {
        throw std::invalid_argument(
            fmt::format("The collisional timestep must be finite and positive, but it is {} instead", m_ct));
    }

    // Set the number of parallel collisional timesteps.
    set_n_par_ct(n_par_ct);

    // Set the conjunction threshold.
    set_conj_thresh(conj_thresh);

    // Set the minimum collisional radius.
    set_min_coll_radius(min_coll_radius);

    // Set the whitelists.
    set_coll_whitelist(std::move(coll_whitelist));
    set_conj_whitelist(std::move(conj_whitelist));

    if (dyn.empty()) {
        // Default is Keplerian dynamics with unitary mu.
        dyn = dynamics::kepler();
    }

    // Check the dynamics.
    if (dyn.size() != 6u) {
        throw std::invalid_argument(
            fmt::format("6 dynamical equations are expected, but {} were provided instead", dyn.size()));
    }

    // Assign the pars.
    m_pars = std::make_shared<std::vector<double>>(std::move(pars));

    // Record the number of pars in the dynamical equations.
    std::uint32_t npars = 0;

    for (auto i = 0u; i < 6u; ++i) {
        const auto &[var, eq] = dyn[i];

        // Check that the LHS is a variable with the correct name.
        if (!std::holds_alternative<hy::variable>(var.value())
            || std::get<hy::variable>(var.value()).name() != detail::allowed_vars[i]) {
            throw std::invalid_argument(fmt::format("The LHS of the dynamics at index {} must be a variable named "
                                                    "\"{}\", but instead it is the expression \"{}\"",
                                                    i, detail::allowed_vars[i], var));
        }

        // Update the number of pars.
        npars = std::max(npars, hy::get_param_size(eq));

        // Check the list of variables in the RHS.
        const auto eq_vars = hy::get_variables(eq);
        std::vector<std::string> set_diff;
#if defined(__clang__)
        std::set_difference(eq_vars.cbegin(), eq_vars.cend(), detail::allowed_vars_alph.cbegin(),
                            detail::allowed_vars_alph.cend(), std::back_inserter(set_diff));
#else
        std::ranges::set_difference(eq_vars, detail::allowed_vars_alph, std::back_inserter(set_diff));
#endif

        if (!set_diff.empty()) {
            throw std::invalid_argument(
                fmt::format("The RHS of the differential equation for the variable \"{}\" contains the invalid "
                            "variables {} (the allowed variables are {})",
                            std::get<hy::variable>(var.value()).name(), set_diff, detail::allowed_vars));
        }
    }

    // Assign m_npars.
    m_npars = npars;

    // Validate m_pars.
    validate_pars_vector(*m_pars, nparts);

    // Add the differential equation for r.
    const auto sym_vars = hy::make_vars("x", "y", "z", "vx", "vy", "vz", "r");
    const auto &x = sym_vars[0];
    const auto &y = sym_vars[1];
    const auto &z = sym_vars[2];
    const auto &vx = sym_vars[3];
    const auto &vy = sym_vars[4];
    const auto &vz = sym_vars[5];
    const auto &r = sym_vars[6];
    dyn.push_back(hy::prime(r) = hy::sum({x * vx, y * vy, z * vz}) / r);

    // Check and assign reentry_radius.
    if (const auto *vcr_ptr = std::get_if<std::vector<double>>(&reentry_radius)) {
        if (vcr_ptr->size() != 3u) {
            throw std::invalid_argument(fmt::format(
                "The reentry_radius argument must be either a scalar (for a spherical central body) "
                "or a vector of 3 elements (for a triaxial ellipsoid), but instead it is a vector of {} element(s)",
                vcr_ptr->size()));
        }

        if (std::any_of(vcr_ptr->cbegin(), vcr_ptr->cend(),
                        [](double val) { return !std::isfinite(val) || val <= 0; })) {
            throw std::invalid_argument(fmt::format(
                "A non-finite or non-positive value was detected among the 3 semiaxes of the central body: {}",
                *vcr_ptr));
        }
    } else {
        const auto cr_val = std::get<double>(reentry_radius);

        if (!std::isfinite(cr_val) || cr_val < 0) {
            throw std::invalid_argument(fmt::format(
                "The radius of the central body must be finite and non-negative, but it is {} instead", cr_val));
        }
    }
    m_reentry_radius = std::move(reentry_radius);

    // Check and assign d_radius.
    if (!std::isfinite(d_radius) || d_radius < 0) {
        throw std::invalid_argument(
            fmt::format("The domain radius must be finite and non-negative, but it is {} instead", d_radius));
    }
    m_d_radius = d_radius;

    // Machinery to construct the integrators.
    std::optional<hy::taylor_adaptive<double>> s_ta;
    std::optional<hy::taylor_adaptive_batch<double>> b_ta;

    auto integrators_setup = [&]() {
        // Helpers to create the exit/reentry event equations.
        auto make_exit_eq = [&]() {
            assert(m_d_radius > 0);

            return hy::sum_sq({x, y, z}) - m_d_radius * m_d_radius;
        };

        auto make_reentry_eq = [&]() {
            if (auto *dbl_ptr = std::get_if<double>(&m_reentry_radius)) {
                assert(*dbl_ptr > 0);

                return hy::sum_sq({x, y, z}) - *dbl_ptr * *dbl_ptr;
            } else {
                const auto &ax_vec = std::get<std::vector<double>>(m_reentry_radius);

                assert(ax_vec.size() == 3u);

                const auto ax_a = ax_vec[0];
                const auto ax_b = ax_vec[1];
                const auto ax_c = ax_vec[2];

                return hy::sum_sq({ax_b * ax_c * x, ax_a * ax_c * y, ax_a * ax_b * z})
                       - ax_a * ax_a * ax_b * ax_b * ax_c * ax_c;
            }
        };

        oneapi::tbb::parallel_invoke(
            [&]() {
                using ev_t = hy::taylor_adaptive<double>::t_event_t;
                std::vector<ev_t> t_events;

                if (with_exit_event()) {
                    t_events.emplace_back(
                        make_exit_eq(),
                        // NOTE: direction is positive in order to detect only domain exit (not entrance).
                        hy::kw::direction = hy::event_direction::positive);
                }

                if (with_reentry_event()) {
                    t_events.emplace_back(make_reentry_eq(),
                                          // NOTE: direction is negative in order to detect only crashing into.
                                          hy::kw::direction = hy::event_direction::negative);
                }

                s_ta.emplace(dyn, std::vector<double>(7u), hy::kw::t_events = std::move(t_events), hy::kw::tol = tol,
                             hy::kw::high_accuracy = ha);
            },
            [&]() {
                const std::uint32_t batch_size = hy::recommended_simd_size<double>();

                using ev_t = hy::taylor_adaptive_batch<double>::t_event_t;
                std::vector<ev_t> t_events;

                if (with_exit_event()) {
                    t_events.emplace_back(
                        make_exit_eq(),
                        // NOTE: direction is positive in order to detect only domain exit (not entrance).
                        hy::kw::direction = hy::event_direction::positive);
                }

                if (with_reentry_event()) {
                    t_events.emplace_back(make_reentry_eq(),
                                          // NOTE: direction is negative in order to detect only crashing into.
                                          hy::kw::direction = hy::event_direction::negative);
                }

                const std::vector<double>::size_type state_size = safe_size_t(7) * batch_size;
                b_ta.emplace(dyn, std::vector<double>(state_size), batch_size, hy::kw::t_events = std::move(t_events),
                             hy::kw::tol = tol, hy::kw::high_accuracy = ha);
            });
    };

    spdlog::stopwatch sw;

    // Concurrently:
    // - setup the heyoka integrators,
    // - check the state vector.
    oneapi::tbb::parallel_invoke(integrators_setup, [this]() { verify_state_vector(*m_state); });

    logger->trace("Integrators setup time: {}s", sw);

    assert(s_ta);
    assert(b_ta);

#if defined(__clang__)

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"

    auto data_ptr = std::make_unique<sim_data>(sim_data{std::move(*s_ta), std::move(*b_ta)});

#pragma GCC diagnostic pop

#else

    auto data_ptr = std::make_unique<sim_data>(std::move(*s_ta), std::move(*b_ta));

#endif

    m_data = std::move(data_ptr);

    sw.reset();

    add_jit_functions();

    logger->trace("JIT functions setup time: {}s", sw);
}

double sim::get_time() const
{
    return static_cast<double>(m_data->time);
}

void sim::set_time(double t)
{
    if (!std::isfinite(t)) {
        throw std::invalid_argument(fmt::format("Cannot set the simulation time to the non-finite value {}", t));
    }

    m_data->time = decltype(m_data->time)(t);
}

bool sim::with_reentry_event() const
{
    if (const auto *dbl_ptr = std::get_if<double>(&m_reentry_radius)) {
        assert(std::isfinite(*dbl_ptr));
        assert(*dbl_ptr >= 0);

        return *dbl_ptr > 0;
    } else {
#if !defined(NDEBUG)
        const auto &vec = std::get<std::vector<double>>(m_reentry_radius);
        assert(vec.size() == 3u);
        assert(std::all_of(vec.cbegin(), vec.cend(), [](double val) { return std::isfinite(val) && val > 0; }));
#endif

        return true;
    }
}

bool sim::with_exit_event() const
{
    assert(std::isfinite(m_d_radius));
    assert(m_d_radius >= 0);

    return m_d_radius > 0;
}

// Helpers to compute the indices of the reentry/exit events.
std::uint32_t sim::reentry_event_idx() const
{
    assert(with_reentry_event());

    // NOTE: reentry event at index 0 or 1, depending
    // on whether we have exit event or not.

    return static_cast<std::uint32_t>(with_exit_event());
}

std::uint32_t sim::exit_event_idx() const
{
    assert(with_exit_event());

    // NOTE: exit event is always at index 0, if
    // it exists.

    return 0;
}

// Run sanity checks on a state vector and verify that it
// is compatible with the simulation setup. Specifically,
// verify that:
// - all values are finite,
// - if central and/or domain radius are defined in the
//   simulation, no position falls within the central body
//   or outside the domain,
// - no particle size is negative.
void sim::verify_state_vector(const std::vector<double> &st) const
{
    namespace stdex = std::experimental;

    // Check that the state vector's size is a multiple of 7
    // (i.e., the state vector of each particle contains 7 values:
    // cartesian pos/vel + size).
    if (st.size() % 7u != 0u) {
        throw std::invalid_argument(
            fmt::format("The size of the state vector is {}, which is not a multiple of 7", st.size()));
    }

    // Infer the number of particles.
    const auto nparts = st.size() / 7u;

    // Init the span for accessing st as a 2D array.
    stdex::mdspan sv(st.data(), stdex::extents<size_type, stdex::dynamic_extent, 7u>(nparts));

    // Check if reentry/exit events are defined in the current simulation.
    const auto with_reentry = with_reentry_event();
    const auto with_exit = with_exit_event();

    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<size_type>(0, nparts), [sv, with_reentry, with_exit, this](const auto &range) {
            for (auto pidx = range.begin(); pidx != range.end(); ++pidx) {
                // Positions.
                const auto x = sv(pidx, 0);
                const auto y = sv(pidx, 1);
                const auto z = sv(pidx, 2);

                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                    throw std::invalid_argument(fmt::format(
                        "An non-finite position was detected in the state vector of the particle at index {}", pidx));
                }

                if (with_reentry) {
                    if (const auto *dbl_ptr = std::get_if<double>(&m_reentry_radius)) {
                        if (x * x + y * y + z * z < *dbl_ptr * *dbl_ptr) {
                            throw std::invalid_argument(
                                fmt::format("The particle at index {} is inside the spherical central body", pidx));
                        }
                    } else {
                        const auto &ax_vec = std::get<std::vector<double>>(m_reentry_radius);
                        const auto ax_a = ax_vec[0];
                        const auto ax_b = ax_vec[1];
                        const auto ax_c = ax_vec[2];

                        if (x * x / (ax_a * ax_a) + y * y / (ax_b * ax_b) + z * z / (ax_c * ax_c) < 1) {
                            throw std::invalid_argument(
                                fmt::format("The particle at index {} is inside the ellipsoidal central body", pidx));
                        }
                    }
                }

                if (with_exit) {
                    if (x * x + y * y + z * z >= m_d_radius * m_d_radius) {
                        throw std::invalid_argument(
                            fmt::format("The particle at index {} is outside the domain radius {}", pidx, m_d_radius));
                    }
                }

                // Velocities
                if (!std::isfinite(sv(pidx, 3)) || !std::isfinite(sv(pidx, 4)) || !std::isfinite(sv(pidx, 5))) {
                    throw std::invalid_argument(fmt::format(
                        "An non-finite velocity was detected in the state vector of the particle at index {}", pidx));
                }

                // Size.
                if (!std::isfinite(sv(pidx, 6)) || sv(pidx, 6) < 0) {
                    throw std::invalid_argument(fmt::format(
                        "An invalid particle size of {} was detected for the particle at index {}", sv(pidx, 6), pidx));
                }
            }
        });
}

void sim::reset_conjunctions()
{
    m_det_conj = std::make_shared<std::vector<conjunction>>();
}

std::ostream &operator<<(std::ostream &os, const sim &s)
{
    os << "Total number of particles: " << s.get_nparts() << '\n';
    os << "Collisional timestep     : " << s.get_ct() << '\n';

    return os;
}

std::ostream &operator<<(std::ostream &os, const sim::conjunction &c)
{
    return os << "{" << c.i << ", " << c.j << ", " << fmt::format("{}", c.time) << ", " << fmt::format("{}", c.dist)
              << "}";
}

void sim::set_min_coll_radius(double min_coll_radius)
{
    if (std::isnan(min_coll_radius) || min_coll_radius < 0) {
        throw std::invalid_argument(fmt::format(
            "The minimum collisional radius cannot be NaN or negative, but the invalid value {} was provided",
            min_coll_radius));
    }

    m_min_coll_radius = min_coll_radius;
}

void sim::set_coll_whitelist(whitelist_t wl)
{
    m_coll_whitelist = std::move(wl);
}

void sim::set_conj_whitelist(whitelist_t wl)
{
    m_conj_whitelist = std::move(wl);
}

} // namespace cascade
