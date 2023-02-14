# Copyright 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the cascade library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import typing
import heyoka as hy
import numpy as np

def mascon_asteroid(Gconst: float, points, masses, omega) -> typing.List[typing.Tuple[hy.expression, hy.expression]]:
    """Dynamics around a rotating asteroid.
    
    This function generates the dynamical equations for objects orbiting around an irregular
    rotating body. The equations are written in the body fixed rotating frame.

    Specifically, the returned system of differential equations will be:

    .. math::

        \\begin{array}{ll}
            \\left\\{
            \\begin{array}{l}
            \\dot{\mathbf r} = \mathbf v \\\\
            \\dot {\mathbf v} = -G \sum_{j=0}^N \\frac {m_j}{|\mathbf r - \mathbf r_j|^3} (\\mathbf r - \\mathbf r_j) - 2 \\boldsymbol\\omega \\times \\mathbf v - \\boldsymbol \\omega \\times\\boldsymbol\\omega \\times \\mathbf r
            \\end{array}
            \\right.
        \\end{array}

    Args:
        Gconst (float): Cavendish constant
        points (array (N,3)): Cartesian coordinates of the mascons
        masses (array (N,)): mass of each mascon
        omega (array (3,)): asteroid angular velocity

    Returns:
        The dynamics in the same units used by the input arguments. Can be used directly to instantiate a :class:`~cascade.sim`.
    """
    from heyoka import make_mascon_system
    return make_mascon_system(Gconst, points, masses, omega)

def mascon_asteroid_energy(state, Gconst: float, points, masses, omega):
    """Energy in the :class:`~cascade.dynamics.mascon_asteroid` dynamics.
    
    This function returns the energy of a given state in the mascon asteroid dynamics.
     
    Args:
        state (array (6,1)): x, y, z, vx, vy, vz in the units used.
        Gconst (float): Cavendish constant
        points (array (N,3)): Cartesian coordinates of the mascons
        masses (array (N,)): mass of each mascon
        omega (array (3,)): asteroid angular velocity

    Returns:
        The energy.
    """
    pe, qe, re = omega
    kinetic = (state[3] * state[3] + state[4] * state[4] + state[5] * state[5]) / 2.
    potential_g = 0.
    for i in range(len(masses)):
        distance = np.sqrt((state[0] - points[i][0]) * (state[0] - points[i][0])
                         + (state[1] - points[i][1]) * (state[1] - points[i][1])
                         + (state[2] - points[i][2]) * (state[2] - points[i][2]))
        potential_g -= Gconst * masses[i] / distance
    
    potential_c = - 0.5 * (pe * pe + qe * qe + re * re) * (state[0] * state[0] + state[1] * state[1] + state[2] * state[2])  + 0.5 * (state[0] * pe + state[1] * qe + state[2] * re) * (state[0] * pe + state[1] * qe + state[2] * re)
    return kinetic + potential_g + potential_c 

