.. _collision_algorithm:

Collision Algorithm
=========================

The collision algorithm in cascade is based on the availability, at each step, of Taylor polinomial expansions
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

where :math:`\tau \in \left[0, c_t\right]` is any time coordinate within the so called collisional
timestep :math:`c_t`, a user defined parameter. 

.. note::
  In reality these expressions are piecewise 
  continuous polynomials since the collisional timestep may be different from the adaptive
  timestep of the Taylor integrator :math:`h`. For the purpose of understnding the main algorithmic
  ideas, this is here ignored.

cascade makes use of these expressions to compute `Axis Aligned Bounding boxes (AABB) <https://en.wikipedia.org/wiki/Bounding_volume>`__ encapsulating
the positions of each orbiting objects within the collisional timestep. 
This is done applying interval arithmetic directly  considering that :math:`\tau \in \left[0, c_t\right]`.

Broad Phase 
================================
The detection of overlaps between bounding boxes is a well-studied problem in computer graphics 
(where it is usually referred to as "broad phase collision detection").
In particular, several spatial data structures have been developed to improve upon the naive approach of checking 
for all possible overlaps (which has quadratic complexity in the number of particles :math:`N` in the simulation.
In cascade a `Bounding Volume Hierarchy (BVH) <https://en.wikipedia.org/wiki/Bounding_volume_hierarchy>`__ is used over the
4D AABB defined over the coordinates  :math:`x,y,z,r`.

.. note::
  Adding the :math:`r` coordinate is of great importance to exclude many collisions since orbital
  dynamics has a weak spherical component.

Narrow Phase 
=================================
Hopefully, most collisions have been excluded by now and in the remaining cases cascade resolves collisions 
using a computationally more demanding approach: the narrow phase. Assuming the collision between particles :math:`i, j` 
was not excluded in the broad phases, and since the polinomial expression for the states are anyway available, 
cascade computes also the polinomial expressions for the relative distance between the potentially colliding objects:

.. math::

  r_{ij}\left( \tau \right) = \sum_{n=0}^p r_{ij}^{\left[ n \right]}\left( t_0 \right)\tau^n.

at this point, to know wether this results in a collision or a miss cascade only needs to run a (very clever) polinomial
root finding algorithm which excludes efficiently the existence of any root, to only actually compute it
in the few remaining cases.

.. note::
  The algorithm description above has been simplified to allow the user to understand the basic ideas and relate them 
  to cascade behaviour, in particular to the information logged by :func:`~cascade.set_logger_level_trace` 
  and to the choice of the collisional timestep in :class:`~cascade.sim`
