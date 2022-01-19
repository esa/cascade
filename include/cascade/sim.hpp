// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CASCADE_SIM_HPP
#define CASCADE_SIM_HPP

#include <concepts>
#include <cstddef>
#include <functional>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <cascade/detail/visibility.hpp>

namespace cascade
{

#define CASCADE_CONCEPT_DECL concept

template <typename T>
CASCADE_CONCEPT_DECL di_range
    = std::ranges::input_range<T> &&std::convertible_to<std::ranges::range_reference_t<T>, double>;

#undef CASCADE_CONCEPT_DECL

class CASCADE_DLL_PUBLIC sim
{
    struct sim_data;

    std::vector<double> m_x, m_y, m_z, m_vx, m_vy, m_vz, m_sizes;
    sim_data *m_data = nullptr;

    void finalise_ctor();
    CASCADE_DLL_LOCAL void add_jit_functions();
    CASCADE_DLL_LOCAL void morton_encode_sort();
    CASCADE_DLL_LOCAL void construct_bvh_trees();
    CASCADE_DLL_LOCAL void verify_bvh_trees();
    CASCADE_DLL_LOCAL void broad_phase();
    CASCADE_DLL_LOCAL void verify_broad_phase();
    CASCADE_DLL_LOCAL void narrow_phase(double);
    CASCADE_DLL_LOCAL double infer_superstep();

    template <typename InTup, typename OutTup, std::size_t... I>
    void ctor_impl(const InTup &in_tup, const OutTup &out_tup, std::index_sequence<I...>)
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
    using size_type = std::vector<double>::size_type;

    sim();
    template <di_range X, di_range Y, di_range Z, di_range VX, di_range VY, di_range VZ, di_range S>
    explicit sim(X &&x, Y &&y, Z &&z, VX &&vx, VY &&vy, VZ &&vz, S &&s)
    {
        auto in_tup
            = std::forward_as_tuple(std::forward<X>(x), std::forward<Y>(y), std::forward<Z>(z), std::forward<VX>(vx),
                                    std::forward<VY>(vy), std::forward<VZ>(vz), std::forward<S>(s));
        auto out_tup = std::make_tuple(std::ref(m_x), std::ref(m_y), std::ref(m_z), std::ref(m_vx), std::ref(m_vy),
                                       std::ref(m_vz), std::ref(m_sizes));
        ctor_impl(in_tup, out_tup, std::make_index_sequence<std::tuple_size_v<decltype(out_tup)>>{});

        finalise_ctor();
    }
    sim(const sim &);
    sim(sim &&) noexcept;
    ~sim();

    sim &operator=(const sim &);
    sim &operator=(sim &&) noexcept;

    size_type get_nparts() const
    {
        return m_x.size();
    }

    const auto &get_x() const
    {
        return m_x;
    }

    void propagate_for(double);
};

} // namespace cascade

#endif
