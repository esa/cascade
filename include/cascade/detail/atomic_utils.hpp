// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CASCADE_DETAIL_ATOMIC_UTILS_HPP
#define CASCADE_DETAIL_ATOMIC_UTILS_HPP

namespace cascade::detail
{

template <typename T>
void lb_atomic_update(T &, T);

template <typename T>
void ub_atomic_update(T &, T);

} // namespace cascade::detail

#endif
