# Copyright 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the cascade library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

def rotating_mascon(Gconst, points, masses, omega):
    """Returns heyoka expressions to be used as dynamics in :class:`~cascade.sim` and corresponding
    to a purely gravitational, rotating mascon model

    Args:
        Gconst (float): Cavendish constant
        points (Nx3 array): Cartesian coordinates of the mascons
        masses (N array): mass of each mascon
        omega (3D array): asteroid angular velocity

    Returns:
        list of tuples (:class:`heyoka.expression`,:class:`heyoka.expression`): The dynamics in SI units. Can be used to instantiate a :class:`~cascade.sim`.
    """
    from heyoka import make_mascon_system
    return make_mascon_system(Gconst, points, masses, omega)

