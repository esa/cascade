// Copyright 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
//
// This file is part of the cascade.py library.
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include <string>

#include "docstrings.hpp"

namespace cascade_py::docstrings
{

std::string dynamics_kepler_docstring()
{
    return R"(kepler(mu: float = 1.) -> typing.List[typing.Tuple[heyoka.expression, heyoka.expression]]

Keplerian dynamics

This function generates the dynamical equations of purely-Keplerian dynamics. Specifically,
the returned system of differential equations will be:

.. math::

   \begin{cases}
     \frac{dx}{dt} & = v_x\\
     \frac{dy}{dt} & = v_y\\
     \frac{dz}{dt} & = v_z\\
     \frac{dv_x}{dt} & = -\frac{\mu x}{\left( x^2+y^2+z^2 \right)^{\frac{3}{2}}}\\
     \frac{dv_y}{dt} & = -\frac{\mu y}{\left( x^2+y^2+z^2 \right)^{\frac{3}{2}}}\\
     \frac{dv_z}{dt} & = -\frac{\mu z}{\left( x^2+y^2+z^2 \right)^{\frac{3}{2}}}
   \end{cases}.

Parameters
----------

mu: float = 1.
    The gravitational parameter.

Returns
-------

typing.List[typing.Tuple[heyoka.expression, heyoka.expression]]
    The system of differential equations corresponding to Keplerian dynamics.

)";
}

std::string outcome_docstring()
{
    return R"(The simulation outcome enum

This enum is used as the return value for the propagation methods of the :class:`~cascade.sim`
class (such as :meth:`cascade.sim.step()` and :meth:`cascade.sim.propagate_until()`).

The possible values for the enum are:

- ``success``, indicating the successful completion of a single step,
- ``time_limit``, indicating that a time limit was reached,
- ``collision``, indicating that a particle-particle collision occurred,
- ``reentry``, indicating that a particle entered the reentry domain,
- ``exit``, indicating that a particle exited the simulation domain,
- ``err_nf_state``, indicating that one or more non-finite values were
  detected in the state of the simulation.

)";
}

std::string sim_docstring()
{
    return R"(The simulation class

This class acts as a container for a collection of spherical particles, propagating their
states in time according to user-defined dynamical equations and accounting for
collisions and/or conjunctions.

The state of each particle is described by the following quantities:

- the three Cartesian positions :math:`x`, :math:`y` and :math:`z`,
- the three Cartesian velocities :math:`v_x`, :math:`v_y` and :math:`v_z`,
- the particle radius :math:`s`.

The dynamical equations describing the evolution in time of :math:`\left( x,y,z,v_x,v_y,v_z \right)`
can be formulated via the `heyoka.py <https://github.com/bluescarni/heyoka.py>`__ expression system.
Alternatively, ready-made dynamical equations for a variety of use cases are available in the
:mod:`cascade.dynamics` module.

)";
}

std::string sim_init_docstring()
{
    return R"(__init__(state: numpy.ndarray[numpy.double], ct: float, dyn: typing.Optional[typing.List[typing.Tuple[heyoka.expression, heyoka.expression]]] = None, reentry_radius: typing.Optional[float | typing.List[float]] = None, exit_radius: typing.Optional[float] = None, pars: typing.Optional[numpy.ndarray[numpy.double]] = None, tol: typing.Optional[float] = None, high_accuracy: bool = False, compact_mode: bool = False, n_par_ct: int = 1, conj_thresh: float = 0.0, min_coll_radius: float = 0.0, coll_whitelist: typing.Set[int] = set(), conj_whitelist: typing.Set[int] = set())

Constructor

The only two mandatory arguments for the constructor are the initial state and the collisional
timestep. Several additional configuration options for the simulation can be specified either at
or after construction.

Parameters
----------

state: numpy.ndarray[numpy.double]
    The initial state vector for the simulation. Must be a two-dimensional array of shape
    :math:`n\times 7`, where the number of rows :math:`n` is the number of particles in the simulation,
    the first 6 columns contain the cartesian state variables :math:`\left( x,y,z,v_x,v_y,v_z \right)`
    of each particle, and the seventh column contains the particle sizes.

    After construction, the state vector can be accessed via the :attr:`state` attribute.
ct: float
    The length in time units of the collisional timestep. Must be positive and finite.
    After construction, the collisional timestep can be changed at any time via the
    :attr:`ct` attribute.
dyn: typing.Optional[typing.List[typing.Tuple[heyoka.expression, heyoka.expression]]] = None
    The particles' dynamical equations. By default, the dynamics is purely Keplerian (see
    :func:`cascade.dynamics.kepler()`).

    Note that the dynamical equations **cannot** be changed after construction.
reentry_radius: typing.Optional[float | typing.List[float]] = None
    The radius of the reentry domain. If a single scalar is provided, then the reentry
    domain is a sphere of the given radius centred in the origin. If a list of 3 values is
    provided, then the reentry domain is a triaxial ellipsoid centred in the origin whose
    three semi-axes lengths :math:`\left( a,b,c \right)` are given by the values in the list.

    When a particle enters the reentry domain, a reentry event is triggered and the simulation
    is stopped. If no reentry radius is provided (the default), then no reentry events are possible.
exit_radius: typing.Optional[float] = None
    Exit radius. If provided, the simulation will track the distance of all particles
    from the origin, and when a particle's distance from the origin exceeds the provided limit,
    the simulation will stop with an exit event. By default, no exit radius is defined.
pars: typing.Optional[numpy.ndarray[numpy.double]] = None
    Values of the dynamical parameters. If the particles' dynamical equations contain
    runtime parameters, their values can be provided via this argument, which must
    be a two-dimensional array of shape :math:`n\times N_p`, where :math:`n` is the
    number of particles in the simulation and :math:`N_p` is the number of runtime
    parameters appearing in the dynamical equations. Each row in the array contains
    the values of the runtime parameters for the dynamics of the corresponding
    particle.

    If this argument is not provided (the default), then all runtime parameters for
    all particles are initialised to zero. Note that the values of the runtime parameters
    can be changed at any time via the :attr:`pars` attribute.
tol: typing.Optional[float] = None
    The tolerance used when numerically solving the dynamical equations. If not provided,
    it defaults to the double-precision epsilon (:math:`\sim 2.2\times 10^{-16}`).
high_accuracy: bool = False
    High-accuracy mode. If enabled, the numerical integrator will employ techniques
    to minimise the accumulation of floating-point truncation errors, at the price
    of a small performance penalty. This can be useful to maintain high accuracy
    in long-running simulations.
compact_mode: bool = False
    Compact mode. If enabled, the just-in-time compilation process will manipulate efficiently
    also very long expression for the dynamics. This is useful, for example, when using long expansions
    to model distrubances, or when gravity is modelled via mascon models. This comes at the price
    of a performance penalty (~<x2) in the resulting numerical integrtor.
n_par_ct: int = 1
    Number of collisional timesteps to be processed in parallel. This is a
    tuning parameter that, while not affecting the correctness of the simulation,
    can greatly influence its performance. The optimal value of this parameter
    depends heavily on the specifics of the simulation, and thus users are advised
    to experiment with different values to determine which one works best.
conj_thresh: float = 0.0
    Conjunction threshold. Conjunctions are tracked only if the conjunction distance
    is less than this threshold. By default, this value is set to zero, which means
    that conjunction tracking is disabled. The conjunction threshold can be changed
    at any time via the ``conj_thresh`` attribute.
min_coll_radius: float = 0.0
    Minimum collisional radius. A collision between two particles is detected
    only if the radius of at least one particle is greater than this value. By default,
    this value is zero, which means that only collisions between point-like particles
    are skipped. If this value is set to :math:`+\infty`, then collision detection
    is disabled for all particles. The minimum collisional radius can be changed
    at any time via the ``min_coll_radius`` attribute.
coll_whitelist: typing.Set[int] = set()
    Collision whitelist. If **not** empty, then a collision between two particles is
    detected only if at least one particle is in the whitelist. By default, the collision
    whitelist is empty, which means that the whitelisting mechanism is disabled. The
    collision whitelist can be changed at any time via the ``coll_whitelist`` attribute.
conj_whitelist: typing.Set[int] = set()
    Conjunction whitelist. If **not** empty, then a conjunction between two particles is
    detected only if at least one particle is in the whitelist. By default, the conjunction
    whitelist is empty, which means that the whitelisting mechanism is disabled. The
    conjunction whitelist can be changed at any time via the ``conj_whitelist`` attribute.

)";
}

std::string sim_pars_docstring()
{
    return R"(Values of the runtime parameters

The parameters are stored in a two-dimensional :class:`~numpy.ndarray` of shape :math:`n\times N_p`, where
:math:`n` is the number of particles in the simulation and :math:`N_p` is the number of runtime
parameters appearing in the dynamical equations.

While this is a read-only property (in the sense that it is not possible to set
a new array of runtime parameters via this property), the values contained in the
array *can* be written to.

)";
}

std::string sim_conj_whitelist_docstring()
{
    return R"(Conjunction whitelist

This :class:`set` contains the list of particles considered during the detection of
conjunction events. If this list is empty, then *all* particles are considered during
the detection of conjunction events.

)";
}

std::string sim_interrupt_info_docstring()
{
    return "Interrupt info";
}

std::string sim_step_docstring()
{
    return R"(step() -> None

)";
}

} // namespace cascade_py::docstrings
