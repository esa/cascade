// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CASCADE_SIM_HPP
#define CASCADE_SIM_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <heyoka/detail/igor.hpp>
#include <heyoka/expression.hpp>

#include <cascade/detail/visibility.hpp>

namespace cascade
{

namespace detail
{

template <typename... Args>
inline constexpr bool always_false_v = false;

} // namespace detail

namespace dynamics
{

CASCADE_DLL_PUBLIC std::vector<std::pair<heyoka::expression, heyoka::expression>> kepler(double = 1.);

} // namespace dynamics

namespace kw
{

IGOR_MAKE_NAMED_ARGUMENT(dyn);
IGOR_MAKE_NAMED_ARGUMENT(pars);
IGOR_MAKE_NAMED_ARGUMENT(c_radius);
IGOR_MAKE_NAMED_ARGUMENT(d_radius);
IGOR_MAKE_NAMED_ARGUMENT(tol);
IGOR_MAKE_NAMED_ARGUMENT(high_accuracy);

} // namespace kw

#define CASCADE_CONCEPT_DECL concept

template <typename T>
CASCADE_CONCEPT_DECL di_range
    = std::ranges::input_range<T> && std::convertible_to<std::ranges::range_reference_t<T>, double>;

template <typename T>
CASCADE_CONCEPT_DECL di_range_range = std::ranges::input_range<T> && di_range<std::ranges::range_reference_t<T>>;

#undef CASCADE_CONCEPT_DECL

enum class outcome { success, time_limit, collision, reentry, exit, err_nf_state };

class CASCADE_DLL_PUBLIC sim
{
public:
    using size_type = std::vector<double>::size_type;

private:
    struct sim_data;

    // NOTE: wrap into shared pointers to enable
    // safe resizing on the Python side: when a
    // NumPy array is constructed from the internal
    // state/pars, it grabs a copy of the shared pointer,
    // so that if m_state/m_pars are reset with new values,
    // the existing NumPy arrays still work safely.
    std::shared_ptr<std::vector<double>> m_state;
    std::shared_ptr<std::vector<double>> m_pars;
    // The collisional timestep.
    double m_ct = 0;
    // The internal implementation-detail data (buffers, caches, etc.).
    sim_data *m_data = nullptr;
    // Simulation interrupt info.
    // NOTE: the three possibilities in the variant are:
    // - particle-particle collision (two particle indices),
    // - reentry/exit (single particle index),
    // - nf state (particle idx + time coordinate at the beginning of the dynamical step
    //   in which the non-finite step was generated, measured relative to the beginning
    //   of the superstep).
    std::optional<std::variant<std::array<size_type, 2>, size_type, std::tuple<size_type, double>>> m_int_info;
    // Central body radius(es).
    std::variant<double, std::vector<double>> m_c_radius;
    // Domain radius.
    double m_d_radius = 0;
    // Number of params in the dynamics.
    size_type m_npars = 0;

    void finalise_ctor(std::vector<std::pair<heyoka::expression, heyoka::expression>>, std::vector<double>,
                       std::variant<double, std::vector<double>>, double, double, bool);
    CASCADE_DLL_LOCAL void add_jit_functions();
    CASCADE_DLL_LOCAL void morton_encode_sort_parallel();
    CASCADE_DLL_LOCAL void construct_bvh_trees_parallel();
    CASCADE_DLL_LOCAL void verify_bvh_trees_parallel() const;
    CASCADE_DLL_LOCAL void broad_phase_parallel();
    CASCADE_DLL_LOCAL void verify_broad_phase_parallel() const;
    CASCADE_DLL_LOCAL void narrow_phase_parallel();
    CASCADE_DLL_LOCAL double infer_superstep();
    CASCADE_DLL_LOCAL void verify_global_aabbs() const;
    CASCADE_DLL_LOCAL void dense_propagate(double);
    template <typename T>
    CASCADE_DLL_LOCAL outcome propagate_until_impl(const T &, double);
    [[nodiscard]] CASCADE_DLL_LOCAL bool with_reentry_event() const;
    [[nodiscard]] CASCADE_DLL_LOCAL bool with_exit_event() const;
    [[nodiscard]] CASCADE_DLL_LOCAL std::uint32_t reentry_event_idx() const;
    [[nodiscard]] CASCADE_DLL_LOCAL std::uint32_t exit_event_idx() const;
    CASCADE_DLL_LOCAL void verify_state_vector(const std::vector<double> &) const;
    CASCADE_DLL_LOCAL void copy_from_final_state() noexcept;

public:
    sim();
    template <typename... KwArgs>
    explicit sim(std::vector<double> state, double ct, KwArgs &&...kw_args)
        : m_state(std::make_shared<std::vector<double>>(std::move(state))), m_ct(ct)
    {
        igor::parser p{kw_args...};

        if constexpr (p.has_unnamed_arguments()) {
            static_assert(detail::always_false_v<KwArgs...>,
                          "The variadic arguments to the constructor of a simulation "
                          "contain unnamed arguments.");
            throw;
        }

        // Dynamics.
        std::vector<std::pair<heyoka::expression, heyoka::expression>> dyn;
        if constexpr (p.has(kw::dyn)) {
            if constexpr (std::assignable_from<decltype(dyn) &, decltype(p(kw::dyn))>) {
                dyn = std::forward<decltype(p(kw::dyn))>(p(kw::dyn));
            } else {
                static_assert(detail::always_false_v<KwArgs...>, "The 'dyn' keyword argument is of the wrong type.");
            }
        }

        // Values of runtime parameters.
        std::vector<double> pars;
        if constexpr (p.has(kw::pars)) {
            if constexpr (std::assignable_from<decltype(pars) &, decltype(p(kw::pars))>) {
                pars = std::forward<decltype(p(kw::pars))>(p(kw::pars));
            } else {
                static_assert(detail::always_false_v<KwArgs...>, "The 'pars' keyword argument is of the wrong type.");
            }
        }

        // Radius of the central body.
        std::variant<double, std::vector<double>> c_radius(0.);
        if constexpr (p.has(kw::c_radius)) {
            if constexpr (std::convertible_to<decltype(p(kw::c_radius)), double>) {
                c_radius = static_cast<double>(std::forward<decltype(p(kw::c_radius))>(p(kw::c_radius)));
            } else if constexpr (di_range<decltype(p(kw::c_radius))>) {
                // NOTE: turn it into an lvalue.
                auto &&tmp_range = p(kw::c_radius);

                std::vector<double> vd;
                for (auto &&val : tmp_range) {
                    vd.push_back(static_cast<double>(val));
                }

                c_radius = std::move(vd);
            } else {
                static_assert(detail::always_false_v<KwArgs...>,
                              "The 'c_radius' keyword argument is of the wrong type.");
            }
        }

        // Domain radius.
        double d_radius = 0;
        if constexpr (p.has(kw::d_radius)) {
            if constexpr (std::convertible_to<decltype(p(kw::d_radius)), double>) {
                d_radius = static_cast<double>(std::forward<decltype(p(kw::d_radius))>(p(kw::d_radius)));
            } else {
                static_assert(detail::always_false_v<KwArgs...>,
                              "The 'd_radius' keyword argument is of the wrong type.");
            }
        }

        // Integration tolerance (defaults to zero, which means
        // auto-detected).
        auto tol = 0.;
        if constexpr (p.has(kw::tol)) {
            if constexpr (std::convertible_to<decltype(p(kw::tol)), double>) {
                tol = static_cast<double>(std::forward<decltype(p(kw::tol))>(p(kw::tol)));
            } else {
                static_assert(detail::always_false_v<KwArgs...>, "The 'tol' keyword argument is of the wrong type.");
            }
        }

        // High accuracy (defaults to false).
        bool ha = false;
        if constexpr (p.has(kw::high_accuracy)) {
            if constexpr (std::convertible_to<decltype(p(kw::high_accuracy)), bool>) {
                ha = static_cast<bool>(std::forward<decltype(p(kw::high_accuracy))>(p(kw::high_accuracy)));
            } else {
                static_assert(detail::always_false_v<KwArgs...>,
                              "The 'high_accuracy' keyword argument is of the wrong type.");
            }
        }

        finalise_ctor(std::move(dyn), std::move(pars), std::move(c_radius), d_radius, tol, ha);
    }
    sim(const sim &);
    sim(sim &&) noexcept;
    ~sim();

    // NOTE: do we need a swap() specialisation as well?
    sim &operator=(const sim &);
    sim &operator=(sim &&) noexcept;

    [[nodiscard]] const auto &get_interrupt_info() const
    {
        return m_int_info;
    }
    [[nodiscard]] const auto &get_state() const
    {
        return *m_state;
    }
    [[nodiscard]] const auto *get_state_data() const
    {
        return m_state->data();
    }
    [[nodiscard]] auto *get_state_data()
    {
        return m_state->data();
    }
    [[nodiscard]] const auto &get_pars() const
    {
        return *m_pars;
    }
    [[nodiscard]] const auto *get_pars_data() const
    {
        return m_pars->data();
    }
    [[nodiscard]] auto *get_pars_data()
    {
        return m_pars->data();
    }
    [[nodiscard]] size_type get_nparts() const
    {
        return get_state().size() / 7u;
    }
    [[nodiscard]] double get_time() const;
    void set_time(double);

    [[nodiscard]] double get_ct() const;
    void set_ct(double);

    [[nodiscard]] double get_tol() const;
    [[nodiscard]] bool get_high_accuracy() const;
    [[nodiscard]] size_type get_npars() const;

    void set_new_state(std::vector<double>);

    outcome step(double = 0);
    outcome propagate_until(double, double = 0);

    // NOTE: these two helpers are used to fetch
    // copies of the shared pointers storing the state
    // vector and the pars vector. The intended purpose
    // is to increase the reference count of the shared
    // pointers so that the destruction of the shared
    // pointers in this does not necessarily lead to
    // the destruction of the vectors contained in the
    // shared pointers. This is inteded to be used
    // on the Python side in order to guarantee that
    // the destruction of a sim object or the invocation
    // of set_new_state() do not trigger the destruction of
    // the state/params vector if a NumPy array holds
    // a reference to them.
    // NOTE: it is prohibited to resize the vectors
    // stored in the returned shared pointers.
    // NOTE: these are to be considered as private
    // implementation details, **not** for public use.
    [[nodiscard]] auto _get_state_ptr() const
    {
        return m_state;
    }
    [[nodiscard]] auto _get_pars_ptr() const
    {
        return m_pars;
    }

private:
    template <typename T>
    CASCADE_DLL_LOCAL void init_scalar_ta(T &, size_type) const;
    template <typename T>
    CASCADE_DLL_LOCAL void init_batch_ta(T &, size_type, size_type) const;
    template <typename T>
    CASCADE_DLL_LOCAL void compute_particle_aabb(unsigned, const T &, const T &, size_type);
};

CASCADE_DLL_PUBLIC std::ostream &operator<<(std::ostream &, const sim &);

} // namespace cascade

#endif
