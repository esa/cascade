.. _collision_algorithm:

Collision Algorithm
=========================

The collision algorithm in cascade exploits the availability, at each step, of Taylor polynomial expansions
of the system dynamics. These are a product of the Taylor integration scheme chosen to propagate the dynamics
and are provided at no additional cost (`heyoka <https://bluescarni.github.io/heyoka/>`__ API
allows for an easy extraction of such coefficients).

The following expressions are thus available:

.. math::
   
  \begin{align}
    x_i\left( \tau \right) & = \sum_{n=0}^p x_i^{\left[ n \right]}\left( t_0 \right)\tau^n,\\
    y_i\left( \tau \right) & = \sum_{n=0}^p y_i^{\left[ n \right]}\left( t_0 \right)\tau^n,\\
    z_i\left( \tau \right) & = \sum_{n=0}^p z_i^{\left[ n \right]}\left( t_0 \right)\tau^n,\\
    r_i\left( \tau \right) & = \sum_{n=0}^p r_i^{\left[ n \right]}\left( t_0 \right)\tau^n,
  \end{align}

where :math:`\tau \in \left[0, c_t\right]` is a time coordinate within the so called collisional
timestep :math:`c_t`, a user defined parameter. 

.. note::
  To be precise, these expressions are piecewise 
  continuous polynomials since the collisional timestep may be different from the adaptive
  timestep of the Taylor integrator :math:`h`. For the purpose of understanding the main algorithmic
  ideas, this is here ignored.

cascade makes use of these expressions to compute `Axis Aligned Bounding boxes (AABB) <https://en.wikipedia.org/wiki/Bounding_volume>`__ 
encapsulating the positions of each orbiting objects within the collisional timestep. 
This is done using `interval arithmetic<https://en.wikipedia.org/wiki/Interval_arithmetic>`__` to compute the polynomial expressions
above and considering that :math:`\tau \in \left[0, c_t\right]`. If the bounding boxes do not overlap, clearly
there will be no possible collision.

Broad Phase 
==============
The detection of overlaps between bounding boxes is a well-studied problem in computer graphics 
(where it is usually referred to as "broad phase collision detection").
In particular, several spatial data structures have been developed to improve upon the naive approach of checking 
for all possible overlaps (which has quadratic complexity in the number of particles :math:`N` in the simulation.
In cascade a `Bounding Volume Hierarchy (BVH) <https://en.wikipedia.org/wiki/Bounding_volume_hierarchy>`__ is used over the
4D AABB defined over the coordinates  :math:`x,y,z,r`.

.. note::
  Adding the :math:`r` coordinate is of great importance to exclude many collisions since orbital
  dynamics has a weak radial component.

Narrow Phase 
==============
Hopefully, most collisions have been excluded by now and in the remaining cases cascade resolves collisions 
using a computationally more demanding approach: the narrow phase. Assuming the collision between particles :math:`i, j` 
was not excluded in the broad phase, and since the polynomial expression for the states are anyway available, 
cascade computes also the polynomial expressions for the relative distance between the potentially colliding objects:

.. math::

  r_{ij}\left( \tau \right) = \sum_{n=0}^p r_{ij}^{\left[ n \right]}\left( t_0 \right)\tau^n.

at this point, to know wether this results in a collision or a miss, cascade only needs to run its clever polynomial
root finding algorithm which excludes efficiently the existence of any root applying
`Descartes rule<https://en.wikipedia.org/wiki/Descartes'_rule_of_signs>`__ , 
to only actually compute it in the few remaining cases.

.. note::
  The algorithm description above has been simplified to allow the user to understand the basic ideas and relate them 
  to cascade behaviour, in particular to the information logged by :func:`~cascade.set_logger_level_trace` 
  and to the choice of the collisional timestep :attr:`~cascade.sim.ct`

Parallelism
============================
The main source of parallelism in cascade derives from the execution of parallel collisional timesteps.
The user defines the number of parallel collisional timesteps :attr:`~cascade.sim.n_par_ct` here denoted with :math:`N_{c_t}`. 
As a consequence, at each :attr:`~cascade.sim.step`, cascade builds the piece-wise polyonomial representation of the system state
within a time interval of width :math:`N_{c_t} \cdot c_t` rather than only :math:`c_t`. This allows to perform the broad and
narrow collision detection in parallel over :math:`N_{c_t}` intervals. This strategy reveals to be very effective in situations
where most of the times the step concludes without any collision detected. It can, though, be detrimental in a collision
rich environment as when a collision is detected, all computations that are being performed over
future time intervals have to be discarded.


