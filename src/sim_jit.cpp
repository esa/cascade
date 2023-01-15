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
#include <utility>
#include <vector>

#include <boost/math/special_functions/binomial.hpp>
#include <boost/numeric/conversion/cast.hpp>

#include <fmt/core.h>

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

#include <heyoka/detail/event_detection.hpp>
#include <heyoka/detail/llvm_helpers.hpp>
#include <heyoka/expression.hpp>
#include <heyoka/llvm_state.hpp>
#include <heyoka/math/sum.hpp>

#include <cascade/detail/sim_data.hpp>
#include <cascade/sim.hpp>

namespace cascade
{

namespace detail
{

namespace
{

namespace hy = heyoka;

// Add a compiled function for the computation of the translation of a polynomial.
// That is, given a polynomial represented as a list of coefficients c_i, the function
// will compute the coefficients c'_i of the polynomial resulting from substituting
// the polynomial variable x with x + a, where a = par[0] is a numerical constant.
// The formula for the translated coefficients is:
//
// c'_i = sum_{k=i}^n (c_k * choose(k, k-i) * a**(k-i))
//
// (where n == order of the polynomial).
void add_poly_translator_a(hy::llvm_state &s, std::uint32_t order)
{
    using namespace hy::literals;

    // The translation amount 'a' is implemented as the
    // first and only parameter of the compiled function.
    auto a = hy::par[0];

    // Pre-compute the powers of a up to 'order'.
    std::vector a_pows = {1_dbl, a};
    for (std::uint32_t i = 2; i <= order; ++i) {
        // NOTE: do it like this, rather than the more
        // straightforward way of multiplying repeatedly
        // by a, in order to improve instruction-level
        // parallelism.
        if (i % 2u == 0u) {
            a_pows.push_back(a_pows[i / 2u] * a_pows[i / 2u]);
        } else {
            a_pows.push_back(a_pows[i / 2u + 1u] * a_pows[i / 2u]);
        }
    }

    // The original polynomial coefficients are the
    // input variables for the compiled function.
    std::vector<hy::expression> cfs;
    for (std::uint32_t i = 0; i <= order; ++i) {
        cfs.emplace_back(fmt::format("c_{}", i));
    }

    // The new coefficients are the function outputs.
    std::vector<hy::expression> out, tmp;
    for (std::uint32_t i = 0; i <= order; ++i) {
        tmp.clear();

        for (std::uint32_t k = i; k <= order; ++k) {
            tmp.push_back(cfs[k]
                          * boost::math::binomial_coefficient<double>(boost::numeric_cast<unsigned>(k),
                                                                      boost::numeric_cast<unsigned>(k - i))
                          * a_pows[k - i]);
        }

        out.push_back(hy::sum(std::move(tmp)));
    }

    // Add the compiled function.
    hy::add_cfunc<double>(s, "pta_cfunc", out, hy::kw::vars = std::move(cfs));
}

// Add a function to compute the sum of the squares
// of the differences between three polynomials.
// NOTE: the squares are computed in truncated arithmetics,
// i.e., given input polynomials of order n the result is
// still a polynomial of order n (rather than 2*n).
void add_poly_ssdiff3(hy::llvm_state &s, std::uint32_t order)
{
    assert(order > 0u); // LCOV_EXCL_LINE

    auto &builder = s.builder();
    auto &context = s.context();

    // The scalar floating-point type and the corresponding pointer type.
    auto *fp_t = hy::detail::to_llvm_type<double>(context);
    auto *fp_ptr_t = llvm::PointerType::getUnqual(fp_t);

    // Fetch the current insertion block.
    auto orig_bb = builder.GetInsertBlock();

    // The function arguments:
    // - the output pointer,
    // - the pointers to the poly coefficients (read-only).
    // No overlap is allowed.
    std::vector<llvm::Type *> fargs(7u, fp_ptr_t);
    // The function returns nothing.
    auto *ft = llvm::FunctionType::get(builder.getVoidTy(), fargs, false);
    assert(ft != nullptr); // LCOV_EXCL_LINE
    // Now create the function.
    auto *f = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "poly_ssdiff3", &s.module());
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

    auto xi_ptr = f->args().begin() + 1;
    xi_ptr->setName("xi_ptr");
    xi_ptr->addAttr(llvm::Attribute::NoCapture);
    xi_ptr->addAttr(llvm::Attribute::NoAlias);
    xi_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto yi_ptr = f->args().begin() + 2;
    yi_ptr->setName("yi_ptr");
    yi_ptr->addAttr(llvm::Attribute::NoCapture);
    yi_ptr->addAttr(llvm::Attribute::NoAlias);
    yi_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto zi_ptr = f->args().begin() + 3;
    zi_ptr->setName("zi_ptr");
    zi_ptr->addAttr(llvm::Attribute::NoCapture);
    zi_ptr->addAttr(llvm::Attribute::NoAlias);
    zi_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto xj_ptr = f->args().begin() + 4;
    xj_ptr->setName("xj_ptr");
    xj_ptr->addAttr(llvm::Attribute::NoCapture);
    xj_ptr->addAttr(llvm::Attribute::NoAlias);
    xj_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto yj_ptr = f->args().begin() + 5;
    yj_ptr->setName("yj_ptr");
    yj_ptr->addAttr(llvm::Attribute::NoCapture);
    yj_ptr->addAttr(llvm::Attribute::NoAlias);
    yj_ptr->addAttr(llvm::Attribute::ReadOnly);

    auto zj_ptr = f->args().begin() + 6;
    zj_ptr->setName("zj_ptr");
    zj_ptr->addAttr(llvm::Attribute::NoCapture);
    zj_ptr->addAttr(llvm::Attribute::NoAlias);
    zj_ptr->addAttr(llvm::Attribute::ReadOnly);

    // Create a new basic block to start insertion into.
    auto *bb = llvm::BasicBlock::Create(context, "entry", f);
    assert(bb != nullptr); // LCOV_EXCL_LINE
    builder.SetInsertPoint(bb);

    // Helper to compute the square of the difference between
    // two polynomials a and b.
    auto pdiff_square = [&](auto *a_ptr, auto *b_ptr) {
        // Init the return value with zeros.
        std::vector<llvm::Value *> ret(boost::numeric_cast<std::vector<llvm::Value *>::size_type>(order + 1u),
                                       static_cast<llvm::Value *>(llvm::ConstantFP::get(fp_t, 0.)));

        for (std::uint32_t i = 0; i <= order / 2u; ++i) {
            auto cf_ai_ptr = builder.CreateInBoundsGEP(fp_t, a_ptr, builder.getInt32(i));
            auto cf_ai = builder.CreateLoad(fp_t, cf_ai_ptr);

            auto cf_bi_ptr = builder.CreateInBoundsGEP(fp_t, b_ptr, builder.getInt32(i));
            auto cf_bi = builder.CreateLoad(fp_t, cf_bi_ptr);

            auto diff_i = builder.CreateFSub(cf_ai, cf_bi);

            ret[2u * i] = builder.CreateFAdd(ret[2u * i], builder.CreateFMul(diff_i, diff_i));

            for (auto j = i + 1u; j <= order - i; ++j) {
                auto cf_aj_ptr = builder.CreateInBoundsGEP(fp_t, a_ptr, builder.getInt32(j));
                auto cf_aj = builder.CreateLoad(fp_t, cf_aj_ptr);

                auto cf_bj_ptr = builder.CreateInBoundsGEP(fp_t, b_ptr, builder.getInt32(j));
                auto cf_bj = builder.CreateLoad(fp_t, cf_bj_ptr);

                auto diff_j = builder.CreateFSub(cf_aj, cf_bj);

                auto tmp = builder.CreateFMul(diff_i, diff_j);
                tmp = builder.CreateFMul(llvm::ConstantFP::get(fp_t, 2.), tmp);

                ret[i + j] = builder.CreateFAdd(ret[i + j], tmp);
            }
        }

        return ret;
    };

    auto diffx2 = pdiff_square(xi_ptr, xj_ptr);
    auto diffy2 = pdiff_square(yi_ptr, yj_ptr);
    auto diffz2 = pdiff_square(zi_ptr, zj_ptr);

    // Write out the sum.
    for (std::uint32_t i = 0; i <= order; ++i) {
        auto out_cf_ptr = builder.CreateInBoundsGEP(fp_t, out_ptr, builder.getInt32(i));

        auto out_cf = builder.CreateFAdd(diffx2[i], diffy2[i]);
        out_cf = builder.CreateFAdd(out_cf, diffz2[i]);

        builder.CreateStore(out_cf, out_cf_ptr);
    }

    // Create the return value.
    builder.CreateRetVoid();

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

    auto *fp_t = heyoka::detail::to_llvm_type<double>(state.context());

    detail::add_poly_translator_a(state, m_data->s_ta.get_order());
    detail::add_poly_ssdiff3(state, m_data->s_ta.get_order());
    heyoka::detail::llvm_add_fex_check(state, fp_t, m_data->s_ta.get_order(), 1);
    heyoka::detail::llvm_add_poly_rtscc(state, fp_t, m_data->s_ta.get_order(), 1);

    state.optimise();

    state.compile();

    m_data->pta_cfunc = reinterpret_cast<decltype(m_data->pta_cfunc)>(state.jit_lookup("pta_cfunc"));
    m_data->pssdiff3 = reinterpret_cast<decltype(m_data->pssdiff3)>(state.jit_lookup("poly_ssdiff3"));
    m_data->fex_check = reinterpret_cast<decltype(m_data->fex_check)>(state.jit_lookup("fex_check"));
    m_data->rtscc = reinterpret_cast<decltype(m_data->rtscc)>(state.jit_lookup("poly_rtscc"));
    // NOTE: this is implicitly added by llvm_add_poly_rtscc().
    m_data->pt1 = reinterpret_cast<decltype(m_data->pt1)>(state.jit_lookup("poly_translate_1"));
}

} // namespace cascade
