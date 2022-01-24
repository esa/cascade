// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CASCADE_DETAIL_ATOMIC_UTILS_HPP
#define CASCADE_DETAIL_ATOMIC_UTILS_HPP

#include <algorithm>
#include <atomic>

namespace cascade::detail
{

// Helper to atomically update the lower bound out_ with the
// value in val.
template <typename T>
inline void lb_atomic_update(T &out_, T val)
{
    // Create an atomic reference for out_.
    std::atomic_ref<T> out(out_);

    // Load the current value from the atomic.
    auto orig_val = out.load(std::memory_order_relaxed);
    T new_val;

    do {
        // Compute the new value.
        // NOTE: min usage safe, we checked outside that
        // there are no NaN values at this point.
        new_val = std::min(val, orig_val);
    } while (!out.compare_exchange_weak(orig_val, new_val, std::memory_order_relaxed, std::memory_order_relaxed));
}

// Helper to atomically update the upper bound out_ with the
// value in val.
template <typename T>
inline void ub_atomic_update(T &out_, T val)
{
    // Create an atomic reference for out_.
    std::atomic_ref<T> out(out_);

    // Load the current value from the atomic.
    auto orig_val = out.load(std::memory_order_relaxed);
    T new_val;

    do {
        // Compute the new value.
        // NOTE: max usage safe, we checked outside that
        // there are no NaN values at this point.
        new_val = std::max(val, orig_val);
    } while (!out.compare_exchange_weak(orig_val, new_val, std::memory_order_relaxed, std::memory_order_relaxed));
}

} // namespace cascade::detail

#endif
