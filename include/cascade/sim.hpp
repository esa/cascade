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

}

namespace dynamics
{

CASCADE_DLL_PUBLIC std::vector<std::pair<heyoka::expression, heyoka::expression>> kepler(double = 1.);

}

namespace kw
{

IGOR_MAKE_NAMED_ARGUMENT(dyn);
IGOR_MAKE_NAMED_ARGUMENT(pars);
IGOR_MAKE_NAMED_ARGUMENT(c_radius);
IGOR_MAKE_NAMED_ARGUMENT(d_radius);

} // namespace kw

#define CASCADE_CONCEPT_DECL concept

template <typename T>
CASCADE_CONCEPT_DECL di_range
    = std::ranges::input_range<T> &&std::convertible_to<std::ranges::range_reference_t<T>, double>;

template <typename T>
CASCADE_CONCEPT_DECL di_range_range = std::ranges::input_range<T> &&di_range<std::ranges::range_reference_t<T>>;

#undef CASCADE_CONCEPT_DECL

enum class outcome { success, time_limit, collision, reentry, exit, err_nf_state };

class CASCADE_DLL_PUBLIC sim
{
public:
    using size_type = std::vector<double>::size_type;

private:
    struct sim_data;

    std::vector<double> m_x, m_y, m_z, m_vx, m_vy, m_vz, m_sizes;
    std::vector<std::vector<double>> m_pars;
    double m_ct;
    sim_data *m_data = nullptr;
    std::optional<std::variant<std::array<size_type, 2>, size_type>> m_int_info;
    std::variant<double, std::vector<double>> m_c_radius;
    double m_d_radius = 0;

    void finalise_ctor(std::vector<std::pair<heyoka::expression, heyoka::expression>>, std::vector<std::vector<double>>,
                       std::variant<double, std::vector<double>>, double);
    void set_new_state_impl(std::array<std::vector<double>, 7> &, std::vector<std::vector<double>>);
    CASCADE_DLL_LOCAL void add_jit_functions();
    CASCADE_DLL_LOCAL void morton_encode_sort();
    CASCADE_DLL_LOCAL void construct_bvh_trees();
    CASCADE_DLL_LOCAL void verify_bvh_trees() const;
    CASCADE_DLL_LOCAL void broad_phase();
    CASCADE_DLL_LOCAL void verify_broad_phase() const;
    CASCADE_DLL_LOCAL void narrow_phase();
    CASCADE_DLL_LOCAL double infer_superstep();
    CASCADE_DLL_LOCAL void verify_global_aabbs() const;
    CASCADE_DLL_LOCAL void dense_propagate(double);
    template <typename T>
    CASCADE_DLL_LOCAL outcome propagate_until_impl(const T &, double);
    CASCADE_DLL_LOCAL bool with_c_radius() const;
    CASCADE_DLL_LOCAL bool with_d_radius() const;
    CASCADE_DLL_LOCAL void check_positions(const std::vector<double> &, const std::vector<double> &,
                                           const std::vector<double> &) const;

    template <typename InTup, typename OutTup, std::size_t... I>
    static void state_set_impl(const InTup &in_tup, OutTup &out_tup, std::index_sequence<I...>)
    {
        auto func = [&](auto ic) {
            constexpr auto Idx = decltype(ic)::value;

            // The type of the input range.
            using in_t = std::tuple_element_t<Idx, InTup>;

            // The input/output vectors.
            auto &in_vec = std::get<Idx>(in_tup);
            auto &out_vec = std::get<Idx>(out_tup);

            if constexpr (std::is_same_v<std::vector<double>, std::remove_cvref_t<in_t>>) {
                // The input range is already a vector<double>: copy/move it in.
                out_vec = std::forward<in_t>(in_vec);
            } else {
                if constexpr (std::ranges::sized_range<in_t>) {
                    if constexpr (std::integral<std::ranges::range_size_t<in_t>>) {
                        // The input range is sized and the size type is a C++ integral.
                        // Prepare the internal vector.
                        out_vec.reserve(boost::numeric_cast<decltype(out_vec.size())>(std::ranges::size(in_vec)));
                    }
                }

                // Add the values.
                for (auto &&val : in_vec) {
                    out_vec.push_back(static_cast<double>(val));
                }
            }
        };

        (func(std::integral_constant<std::size_t, I>{}), ...);
    }

public:
    sim();
    template <di_range X, di_range Y, di_range Z, di_range VX, di_range VY, di_range VZ, di_range S, typename... KwArgs>
    explicit sim(X &&x, Y &&y, Z &&z, VX &&vx, VY &&vy, VZ &&vz, S &&s, double ct, KwArgs &&...kw_args) : m_ct(ct)
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
        std::vector<std::vector<double>> pars;
        if constexpr (p.has(kw::pars)) {
            if constexpr (di_range_range<decltype(p(kw::pars))>) {
                // NOTE: turn it into an lvalue.
                auto &&tmp_range = p(kw::pars);

                for (auto &&rng : tmp_range) {
                    pars.emplace_back();

                    // NOTE: possible optimisation: reserve/move in if possible,
                    // instead of push_back().
                    for (auto &&val : rng) {
                        pars.back().push_back(static_cast<double>(val));
                    }
                }
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

        auto in_tup
            = std::forward_as_tuple(std::forward<X>(x), std::forward<Y>(y), std::forward<Z>(z), std::forward<VX>(vx),
                                    std::forward<VY>(vy), std::forward<VZ>(vz), std::forward<S>(s));
        auto out_tup = std::make_tuple(std::ref(m_x), std::ref(m_y), std::ref(m_z), std::ref(m_vx), std::ref(m_vy),
                                       std::ref(m_vz), std::ref(m_sizes));
        state_set_impl(in_tup, out_tup, std::make_index_sequence<std::tuple_size_v<decltype(in_tup)>>{});

        finalise_ctor(std::move(dyn), std::move(pars), std::move(c_radius), d_radius);
    }
    sim(const sim &);
    sim(sim &&) noexcept;
    ~sim();

    // NOTE: do we need a swap() specialisation as well?
    sim &operator=(const sim &);
    sim &operator=(sim &&) noexcept;

    const std::optional<std::variant<std::array<size_type, 2>, size_type>> &get_interrupt_info() const
    {
        return m_int_info;
    }
    size_type get_nparts() const
    {
        return m_x.size();
    }
    const auto &get_x() const
    {
        return m_x;
    }
    const auto &get_y() const
    {
        return m_y;
    }
    const auto &get_z() const
    {
        return m_z;
    }
    const auto &get_vx() const
    {
        return m_vx;
    }
    const auto &get_vy() const
    {
        return m_vy;
    }
    const auto &get_vz() const
    {
        return m_vz;
    }
    const auto &get_pars() const
    {
        return m_pars;
    }
    const auto &get_sizes() const
    {
        return m_sizes;
    }
    double get_time() const;
    void set_time(double);

    double get_ct() const;
    void set_ct(double);

    template <di_range X, di_range Y, di_range Z, di_range VX, di_range VY, di_range VZ, di_range S>
    void set_new_state(X &&x, Y &&y, Z &&z, VX &&vx, VY &&vy, VZ &&vz, S &&s,
                       std::vector<std::vector<double>> pars = {})
    {
        auto in_tup
            = std::forward_as_tuple(std::forward<X>(x), std::forward<Y>(y), std::forward<Z>(z), std::forward<VX>(vx),
                                    std::forward<VY>(vy), std::forward<VZ>(vz), std::forward<S>(s));

        std::array<std::vector<double>, 7> new_state;

        state_set_impl(in_tup, new_state, std::make_index_sequence<7>{});

        set_new_state_impl(new_state, pars);
    }

    outcome step(double = 0);
    outcome propagate_until(double, double = 0);

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
