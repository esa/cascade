// Copyright 2022 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <vector>

#include <llvm/IR/Attributes.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>

#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/llvm_state.hpp>

#include <cascade/detail/sim_data.hpp>
#include <cascade/sim.hpp>

namespace cascade
{

namespace detail
{

namespace
{

namespace hy = heyoka;

// Add a polynomial translation function, that will perform
// the change of variable x -> x' + a for a polynomial
// of the given order.
void add_poly_translator_a(hy::llvm_state &s, std::uint32_t order)
{
    assert(order > 0u); // LCOV_EXCL_LINE

    auto &builder = s.builder();
    auto &context = s.context();

    // The scalar floating-point type and the corresponding pointer type.
    auto *fp_t = hy::detail::to_llvm_type<double>(context);
    auto *fp_ptr_t = llvm::PointerType::getUnqual(fp_t);

    // Helper to fetch the (i, j) binomial coefficient from
    // a precomputed global array.
    auto get_bc = [&, bc_ptr = hy::detail::llvm_add_bc_array<double>(s, order)](llvm::Value *i, llvm::Value *j) {
        // NOTE: overflow checking for the indexing into bc_ptr is already
        // done in llvm_add_bc_array().
        auto idx = builder.CreateMul(i, builder.getInt32(order + 1u));
        idx = builder.CreateAdd(idx, j);

        return builder.CreateLoad(fp_t, builder.CreateInBoundsGEP(fp_t, bc_ptr, idx));
    };

    // Fetch the current insertion block.
    auto orig_bb = builder.GetInsertBlock();

    // The function arguments:
    // - the output pointer,
    // - the pointer to the poly coefficients (read-only),
    // - the translation value a.
    // No overlap is allowed.
    std::vector<llvm::Type *> fargs{fp_ptr_t, fp_ptr_t, fp_t};
    // The function return a pointer.
    auto *ft = llvm::FunctionType::get(fp_ptr_t, fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE
    // Now create the function.
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "poly_translate_a", &s.module());
    // LCOV_EXCL_START
    if (f == nullptr) {
        throw std::invalid_argument("Unable to create a function for polynomial translation");
    }
    // LCOV_EXCL_STOP

    // Set the names/attributes of the function arguments.
    auto out_ptr = f->args().begin();
    out_ptr->setName("out_ptr");
    out_ptr->addAttr(llvm::Attribute::NoCapture);
    out_ptr->addAttr(llvm::Attribute::NoAlias);

    auto cf_ptr = f->args().begin() + 1;
    cf_ptr->setName("cf_ptr");
    cf_ptr->addAttr(llvm::Attribute::NoCapture);
    cf_ptr->addAttr(llvm::Attribute::NoAlias);
    cf_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto a_val = f->args().begin() + 2;

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr); // LCOV_EXCL_LINE
    builder.SetInsertPoint(bb);

    // Pre-compute the powers of a.
    std::vector<llvm::Value *> a_pows;
    a_pows.push_back(llvm::ConstantFP::get(fp_t, 1.));
    for (std::uint32_t i = 0; i < order; ++i) {
        a_pows.push_back(builder.CreateFMul(a_val, a_pows.back()));
    }

    // Do the translation.
    for (std::uint32_t i = 0; i <= order; ++i) {
        auto new_cf = static_cast<llvm::Value *>(llvm::ConstantFP::get(fp_t, 0.));

        for (std::uint32_t k = i; k <= order; ++k) {
            // Load the original coefficient.
            auto ck_ptr = builder.CreateInBoundsGEP(fp_t, cf_ptr, builder.getInt32(k));
            auto ck = builder.CreateLoad(fp_t, ck_ptr);

            // Fetch the binomial coefficient.
            auto bc = get_bc(builder.getInt32(k), builder.getInt32(k - i));

            // Multiply and accumulate.
            auto tmp1 = builder.CreateFMul(a_pows[k - i], bc);
            auto tmp2 = builder.CreateFMul(tmp1, ck);
            new_cf = builder.CreateFAdd(new_cf, tmp2);
        }

        // Store the new coefficient.
        auto new_cf_ptr = builder.CreateInBoundsGEP(fp_t, out_ptr, builder.getInt32(i));
        builder.CreateStore(new_cf, new_cf_ptr);
    }

    // Create the return value.
    builder.CreateRet(out_ptr);

    // Verify the function.
    s.verify_function(f);

    // Restore the original insertion block.
    builder.SetInsertPoint(orig_bb);
}

} // namespace

} // namespace detail

void sim::add_jit_functions()
{
    auto &state = m_data->state;

    detail::add_poly_translator_a(state, m_data->s_ta.get_order());

    state.optimise();
    state.compile();

    m_data->pta = reinterpret_cast<decltype(m_data->pta)>(state.jit_lookup("poly_translate_a"));
}

} // namespace cascade
