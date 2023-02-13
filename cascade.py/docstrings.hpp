// Copyright 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef CASCADE_PY_DOCSTRINGS_HPP
#define CASCADE_PY_DOCSTRINGS_HPP

#include <string>

namespace cascade_py::docstrings
{

std::string dynamics_kepler_docstring();

std::string outcome_docstring();

std::string sim_docstring();
std::string sim_init_docstring();
std::string sim_time_docstring();
std::string sim_ct_docstring();
std::string sim_n_par_ct_docstring();
std::string sim_conj_thresh_docstring();
std::string sim_min_coll_radius_docstring();
std::string sim_coll_whitelist_docstring();
std::string sim_conj_whitelist_docstring();
std::string sim_nparts_docstring();
std::string sim_npars_docstring();
std::string sim_tol_docstring();
std::string sim_high_accuracy_docstring();
std::string sim_compact_mode_docstring();
std::string sim_reentry_radius_docstring();
std::string sim_exit_radius_docstring();
std::string sim_pars_docstring();
std::string sim_interrupt_info_docstring();
std::string sim_step_docstring();

} // namespace cascade_py::docstrings

#endif
