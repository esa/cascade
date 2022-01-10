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
#include <ranges>
#include <vector>

#include <boost/numeric/conversion/cast.hpp>

#include <xtensor/xadapt.hpp>

#include <cascade/detail/visibility.hpp>

namespace cascade
{

#define CASCADE_CONCEPT_DECL concept

template <typename T>
CASCADE_CONCEPT_DECL di_range = std::ranges::input_range<T>
    &&std::convertible_to<std::ranges::range_reference_t<T>, double> &&std::integral<std::ranges::range_size_t<T>>;

#undef CASCADE_CONCEPT_DECL

class CASCADE_DLL_PUBLIC sim
{
    struct sim_data;

    std::vector<double> m_x, m_y, m_z, m_vx, m_vy, m_vz, m_r;
    sim_data *m_data = nullptr;

    void finalise_ctor();
    CASCADE_DLL_LOCAL void morton_encode_sort();
    CASCADE_DLL_LOCAL void construct_bvh_trees();

public:
    using size_type = std::vector<double>::size_type;

    sim();
    // TODO optimise for input std::vector<double> rvalue.
    explicit sim(di_range auto const &x, di_range auto const &y, di_range auto const &z, di_range auto const &vx,
                 di_range auto const &vy, di_range auto const &vz)
    {
        if constexpr (std::ranges::sized_range<decltype(x)>) {
            m_x.reserve(boost::numeric_cast<decltype(m_x.size())>(std::ranges::size(x)));
        }

        if constexpr (std::ranges::sized_range<decltype(y)>) {
            m_y.reserve(boost::numeric_cast<decltype(m_y.size())>(std::ranges::size(y)));
        }

        if constexpr (std::ranges::sized_range<decltype(z)>) {
            m_z.reserve(boost::numeric_cast<decltype(m_z.size())>(std::ranges::size(z)));
        }

        if constexpr (std::ranges::sized_range<decltype(vx)>) {
            m_vx.reserve(boost::numeric_cast<decltype(m_vx.size())>(std::ranges::size(vx)));
        }

        if constexpr (std::ranges::sized_range<decltype(vy)>) {
            m_vy.reserve(boost::numeric_cast<decltype(m_vy.size())>(std::ranges::size(vy)));
        }

        if constexpr (std::ranges::sized_range<decltype(vz)>) {
            m_vz.reserve(boost::numeric_cast<decltype(m_vz.size())>(std::ranges::size(vz)));
        }

        for (auto &&val : x) {
            m_x.push_back(static_cast<double>(val));
        }

        for (auto &&val : y) {
            m_y.push_back(static_cast<double>(val));
        }

        for (auto &&val : z) {
            m_z.push_back(static_cast<double>(val));
        }

        for (auto &&val : vx) {
            m_vx.push_back(static_cast<double>(val));
        }

        for (auto &&val : vy) {
            m_vy.push_back(static_cast<double>(val));
        }

        for (auto &&val : vz) {
            m_vz.push_back(static_cast<double>(val));
        }

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

    void propagate_for(double);
};

} // namespace cascade

#endif
