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

Keplerian dynamics.

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
    return R"(The simulation outcome enum.

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
    return R"(The simulation class.

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

Constructor.

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
    of a performance penalty (~<x2) in the resulting numerical integrator.
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

std::string sim_time_docstring()
{
    return R"(Current simulation time.

This :class:`float` contains the value of the current simulation time. It can be set, and
it is referred to in the dynamics equation as ``heyoka.time``.

)";
}

std::string sim_ct_docstring()
{
    return R"(Collisional time.

This :class:`float` represents the length in time units of the collisional timestep. Must be positive and finite. 
It can be set and its value will be used in the next call to the propagation methods
(such as :meth:`cascade.sim.step()` and :meth:`cascade.sim.propagate_until()`).

)";
}

std::string sim_n_par_ct_docstring()
{
    return R"(Number of collisional timesteps to be processed in parallel.

This is a tuning parameter that, while not affecting the correctness of the simulation,
can greatly influence its performance. The optimal value of this parameter
depends heavily on the specifics of the simulation, and thus users are advised
to experiment with different values to determine which one works best.

)";
}

std::string sim_conj_thresh_docstring()
{
    return R"(Conjunction threshold.

Conjunctions are tracked only if the conjunction distance
is less than this threshold. By default, this value is set to zero, which means
that conjunction tracking is disabled. The conjunction threshold can be changed
at any time.

)";
}

std::string sim_min_coll_radius_docstring()
{
    return R"(Minimum collisional radius. 

A collision between two particles is detected
only if the radius of at least one particle is greater than this value. By default,
this value is zero, which means that only collisions between point-like particles
are skipped. If this value is set to :math:`+\infty`, then collision detection
is disabled for all particles. The minimum collisional radius can be changed
at any time.

)";
}

std::string sim_coll_whitelist_docstring()
{
    return R"(Collision whitelist.

This :class:`set` contains the list of particles considered during the detection of
collision events. If this list is empty, then *all* particles are considered during
the detection of collision events.

)";
}

std::string sim_conj_whitelist_docstring()
{
    return R"(Conjunction whitelist.

This :class:`set` contains the list of particles considered during the detection of
conjunction events. If this list is empty, then *all* particles are considered during
the detection of conjunction events.

)";
}

std::string sim_nparts_docstring()
{
    return R"(Number of particles.

A read-only attribute containing the current number of particles. The number of particles ina  simulation can be controlled
by removing particles via methods such as :meth:`~sim.remove_particles()` or :meth:`~sim.set_new_state_pars()`.

)";
}

std::string sim_npars_docstring()
{
    return R"(Number of runtime paramenters.

A read-only attribute containing the number of runtime parameters in the dynamics :math:`N_p`.

)";
}

std::string sim_tol_docstring()
{
    return R"(Numerical integration tolerance.

The tolerance used when numerically solving the dynamical equations. If not provided,
it defaults to the double-precision epsilon (:math:`\sim 2.2\times 10^{-16}`). The integration
tolerance cannot be changed after construction.

)";
}

std::string sim_high_accuracy_docstring()
{
    return R"(High-accuracy mode. 

If enabled, the numerical integrator will employ techniques
to minimise the accumulation of floating-point truncation errors, at the price
of a small performance penalty. This can be useful to maintain high accuracy
in long-running simulations. High-accuracy mode cannot be changed after construction.

)";
}

std::string sim_compact_mode_docstring()
{
    return R"(Compact mode. 

If enabled, the just-in-time compilation process will manipulate efficiently
also very long expression for the dynamics. This is useful, for example, when using long expansions
to model distrubances, or when gravity is modelled via mascon models. This comes at the price
of a performance penalty (~<x2) in the resulting numerical integrator. Compact mode
cannot be changed after construction.

)";
}

std::string sim_reentry_radius_docstring()
{
    return R"(The radius of the reentry domain. 

The reentry domain is modelled either as a sphere (in which case this property
will be a single scalar representing the radius of the reentry sphere) or
as a triaxial ellipsoid (in which case this property will be a list of three
values representing the three semi-axes lengths :math:`\left( a,b,c \right)`
of the ellipsoid). If no reentry radius was specified upon construction, this
attribute contains the scalar ``0``.

)";
}

std::string sim_exit_radius_docstring()
{
    return R"(Exit radius.

If an exit radius is provided upon construction, the simulation will track the distance of all particles
from the origin, and when a particle's distance from the origin exceeds this limit,
the simulation will stop with an exit event. By default, no exit radius is defined
and this attribute contains the scalar ``0``.

)";
}

std::string sim_interrupt_info_docstring()
{
    return R"(Interrupt info.

Returns the information on the outcome of the latest call to 
the propagation methods of the :class:`~cascade.sim`
class (such as :meth:`cascade.sim.step()` and :meth:`cascade.sim.propagate_until()`).

The possible values of this attribute are:

- ``None``: the propagation method finished successfully and no stopping
  event was detected;
- ``(i, j)`` (a pair of integers): the propagation method stopped due to the collision
  between particles ``i`` and ``j``;
- ``i`` (a single integer): particle ``i`` either entered the reentry domain or exited
  the simulation domain;
- ``(i, tm)`` (integer + time coordinate): a non-finite state was detected for particle
  ``i`` at time ``tm``.

)";
}

std::string sim_step_docstring()
{
    return R"(step() -> cascade.outcome

Performs a single step of the simulation.

Cascade will try to advance the simulation time by
:attr:`~cascade.sim.ct` times :attr:`~cascade.sim.n_par_ct` time units. If stopping events
(e.g., collision, reentry, etc.) trigger in such a time
interval, the simulation will stop and set its state/time at the epoch the first event was triggered.

Returns
-------

cascade.outcome
    The outcome of the simulation step.

)";
}

std::string sim_propagate_until_docstring()
{
    return R"(propagate_until(t: float) -> cascade.outcome

Attempts to advance the simulation time to *t*.

Cascade will try to advance the simulation time to the
value given. If stopping events trigger in such a time interval,
the simulation will stop and set its state/time at
the epoch the first event was triggered.

Parameters
----------

t: float 
    The propagation time.

Returns
-------

cascade.outcome
    The outcome of the propagation.

)";
}

std::string sim_pars_docstring()
{
    return R"(Values of the runtime parameters appearing in the objects dynamics.

The parameters are stored in a two-dimensional :class:`~numpy.ndarray` of shape :math:`n\times N_p`, where
:math:`n` is the number of particles in the simulation and :math:`N_p` is the number of runtime
parameters appearing in the dynamical equations.

While this is a read-only property (in the sense that it is not possible to set
a new array of runtime parameters via this property), the values contained in the
array CAN be written to.

)";
}

std::string sim_state_docstring()
{
    return R"(State of all the particles currently in the simulation.

The state is stored in a two-dimensional :class:`~numpy.ndarray` of shape
:math:`n\times 7`, where the number of rows :math:`n` is the number of particles in the simulation,
the first 6 columns contain the cartesian state variables :math:`\left( x,y,z,v_x,v_y,v_z \right)`
of each particle, and the seventh column contains the particle sizes.

While this is a read-only property (in the sense that it is not possible to set
a new array of runtime parameters via this property), the values contained in the
array CAN be written to.

)";
}

std::string sim_set_new_state_pars_docstring()
{
    return R"(set_new_state_pars(new_state: numpy.ndarray, new_pars: numpy.ndarray = None) -> None

Sets new values for the simulation state and parameters. 

If no *pars* are passed only the state will be set and all
parameters, if present, will be set to zero.

Note that this method should be used **only** if you need to alter the number
of particles in the simulation. If you all you need to do is to change
the state and/or parameter values for one or more particles, you can write directly
into :attr:`~cascade.sim.state` and :attr:`~cascade.sim.pars`.

Parameters
----------

new_state: numpy.ndarray
    The new state for all particles (the shape must be :math:`n \times 7`).
new_pars: numpy.ndarray = None
    The new runtime parameters for the dynamics of all particles (the shape must be :math:`n \times N_p`).

)";
}

std::string sim_conjunctions_docstring()
{
    return R"(Conjunctions recorded during the simulation.

A read-only attribute containing all the conjunction events. The events are stored in a
:ref:`structured array <numpy:structured_arrays>` containing the following records:

- ``i``, (:class:`int`): id of the first particle involved in the conjunction event,
- ``j``, (:class:`int`): id of the first particle involved in the conjunction event,
- ``t``, (:class:`float`): time of the conjunction event,
- ``dist``, (:class:`float`): closest approach distance,
- ``state_i``, (:class:`~numpy.ndarray`): state (:math:`x,y,z,vx,vy,vz`) of the first particle at the conjunction,
- ``state_j``, (:class:`~numpy.ndarray`): state (:math:`x,y,z,vx,vy,vz`) of the second particle at the conjunction.

Whenever the cascade simulation is made with a :attr:`~cascade.sim.conj_thresh` larger than its
default zero value, all conjunction events of whitelisted objects (or of all objects in case no whitelist is provided)
are detected and tracked. A conjunction event will not stop the simulation hence the user cannot 'react' to it.

)";
}

std::string sim_reset_conjunctions_docstring()
{
    return R"(reset_conjunctions() -> None

Reset the conjunctions list.

Clears the :class:`~numpy.ndarray` storing conjunctions events, thus freeing all associated memory.
This is useful in very long simulations where the number of conjunction events can grow 
substantially. The user can then store them in some other form, or extract relevant statistics 
to then reset the conjunctions.

)";
}

std::string sim_remove_particles_docstring()
{
    return R"(remove_particles(idxs: typing.List[int]) -> None

Removes particles from the simulation.

This takes care to remove all the particles and their corresponding parameters. 

Parameters
----------

idxs: typing.List[int]
   The indices of the particles to be removed.

)";
}

std::string set_nthreads_docstring()
{
    return R"(set_nthreads(n: int) -> None

Sets the maximum number of threads allowed.

Cascade under the hood works with the Threading Building Blocks (TBB) API to control the parallelism of
all its parts. This function exposes the *max_allowed_parallelism* TBB global control setter.

Parameters
----------

n: int
   The maximum allowed number of threads.

)";
}

std::string get_nthreads_docstring()
{
    return R"(get_nthreads() -> int

Gets the maximum number of threads allowed.

Cascade under the hood works with the Threading Building Blocks (TBB) API to control the parallelism of
all its parts. This function exposes the *max_allowed_parallelism* TBB global control getter.

Returns
-------

n: int
   The maximum allowed number of threads.

)";
}

std::string set_logger_level_trace_docstring()
{
    return R"(set_logger_level_trace() -> None

Sets the logger level to "trace"

Cascade under the hood works with the `spdlog C++ logging library <https://github.com/gabime/spdlog>`__ API to control the screen verbosity of
its screen logs. This function sets the level to "trace".

When this level is activated cascade will also report on screen setuptimes for building and jitting the Taylor
integrators as well as, for each simulation step, the timings for the various parts of the cascade algorithm.

A typical output may look like:

.. code-block::

    [...] Integrators setup time: 0.810280909s
    [...] JIT functions setup time: 0.263069601s

    [...] ---- STEP BEGIN ---
    [...] Number of chunks: 10
    [...] Propagation + AABB computation time: 0.012357934s
    [...] Morton encoding and sorting time: 0.015623162s
    [...] BVH construction time: 0.044566595s
    [...] Broad phase collision detection time: 0.354104341s
    [...] Average number of AABB collisions per particle per chunk: 0.03350329934013198
    [...] Narrow phase collision detection time: 0.002799526s
    [...] Total number of collisions detected: 0
    [...] Total number of conjunctions detected: 0
    [...] Runtime for append_conj_data(): 2.24e-06s
    [...] Total propagation time: 0.431442951s
    [...] ---- STEP END ---

This information is useful, among other things, to tune simulation parameters such as the collisional timestep
and the number of parallel collisional timesteps.

)";
}

std::string set_logger_level_debug_docstring()
{
    return R"(set_logger_level_debug() -> None

Sets the logger level to "debug"

Cascade under the hood works with the `spdlog C++ logging library <https://github.com/gabime/spdlog>`__ API to control the screen verbosity of
its screen logs. This function sets the level to "debug".

When this level is activated cascade will also report on screen problems in the
dynamics propagation, for example invalid states generated.

)";
}

std::string set_logger_level_info_docstring()
{
    return R"(set_logger_level_info() -> None

Sets the logger level to "info"

Cascade under the hood works with the `spdlog C++ logging library <https://github.com/gabime/spdlog>`__ API to control the screen verbosity of
its screen logs. This function sets the level to "info".

When this level is activated cascade will also report on screen generic information.

)";
}

std::string set_logger_level_warn_docstring()
{
    return R"(set_logger_level_warn() -> None

Sets the logger level to "warn"

Cascade under the hood works with the `spdlog C++ logging library <https://github.com/gabime/spdlog>`__ API to control the screen verbosity of
its screen logs. This function sets the level to "warn".

When this level is activated cascade may also report on screen various polynomial root finding failiures as well as
other numerical issues that do not invalidate the computations.

)";
}

std::string set_logger_level_err_docstring()
{
    return R"(set_logger_level_err() -> None

Sets the logger level to "err"

Cascade under the hood works with the `spdlog C++ logging library <https://github.com/gabime/spdlog>`__ API to control the screen verbosity of
its screen logs. This function sets the level to "err".

Cascade is not currently making use of this log level.

)";
}

std::string set_logger_level_critical_docstring()
{
    return R"(set_logger_level_critical() -> None

Sets the logger level to "critical"

Cascade under the hood works with the `spdlog C++ logging library <https://github.com/gabime/spdlog>`__ API to control the screen verbosity of
its screen logs. This function sets the level to "critical".

Cascade is not currently making use of this log level.


)";
}

} // namespace cascade_py::docstrings
