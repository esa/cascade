// Copyright 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CASCADE_DETAIL_IVAL_HPP
#define CASCADE_DETAIL_IVAL_HPP

#include <algorithm>

namespace cascade::detail
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
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    explicit ival(double l, double u) : lower(l), upper(u) {}
};

// NOTE: see https://en.wikipedia.org/wiki/Interval_arithmetic.
inline ival operator+(ival a, ival b)
{
    return ival(a.lower + b.lower, a.upper + b.upper);
}

inline ival operator*(ival a, ival b)
{
    const auto tmp1 = a.lower * b.lower;
    const auto tmp2 = a.lower * b.upper;
    const auto tmp3 = a.upper * b.lower;
    const auto tmp4 = a.upper * b.upper;

    const auto l = std::min(std::min(tmp1, tmp2), std::min(tmp3, tmp4));
    const auto u = std::max(std::max(tmp1, tmp2), std::max(tmp3, tmp4));

    return ival(l, u);
}

} // namespace cascade::detail

#endif
