#!/usr/bin/env python
# coding: utf-8

import pykep as pk
import numpy as np
import json
import pickle as pkl

import cascade as csc
from copy import deepcopy
from tqdm.notebook import tqdm
import heyoka as hy

# added for the Cube approach implementation
from collections import defaultdict
import time
import sgp4
from sgp4.api import Satrec, SatrecArray

# multiprocessing
import multiprocessing as mp


# -------------------------------------------------------------------------------------
# Helper functions
def period(r, v, mu):
    """Computes the orbital period from the vis-viva equation

    Args:
        r (float): The radius (in L).
        v (float): The velocity (in L/T).
        mu (float): The gravitational parameter in L^3/T^2

    Returns:
        The orbital period (in T)
    """
    En = v**2/2 - mu / r
    a = -mu / En / 2
    if a < 0:
        raise ValueError("Hyperbola!!!")
    return np.sqrt(a**3/mu)*2*np.pi


def cubes(cartesian_points, cube_dimension):
    """Runs the Cube algorithm and returns satellites within the same cube

    Args:
        cartesian_points (Nx3 np.array): The cartesian position of the satellites (in L).
        cube_dimension (float): The cube dimentsion (in L).

    Returns:
        a list containing lists of satelites idx occupying the same cube
    """
    # init
    retval = []
    cubes = defaultdict(list)

    # We compute the floored Cartesian coordinates identifying the bins.
    pos = cartesian_points
    pos = pos / cube_dimension
    pos = np.floor(pos).astype(int)
    # We fill the bins
    for i, xyz in enumerate(pos):
        cubes[tuple(xyz)].append(i)
    # We find bins with more than one atellite
    for key in cubes:
        if len(cubes[key]) > 1:
            retval.append(cubes[key])
    return retval


def precompute_eph_sgp4(debris, sim_time=20, time_grid=5, t0=8073.603992389981):
    """Computes all satellites ephemerides on a time grid using sgp4

    Args:
        debris (list of pk.planets): The objects to propagate.
        sim_time (float): The total propagation time (in years).
        time_grid(float): The time resolution (in days).
        t0 (float): the starting epoch in mjd2000.

    Returns:
        a list containing lists of idx identifying the object occupying the same cube
    """
    # This list will contain all the sgp4 Satrec objects
    satellite_l = []
    for deb in debris:
        l1 = deb.line1
        l2 = deb.line2
        satellite_l.append(Satrec.twoline2rv(l1, l2))
    # Here we build the vectorized version allowing for speed
    satellites = SatrecArray(satellite_l)
    jd0, fr = pk.epoch(t0).jd, 0.0
    # The Julian dates are from jd0 to 20 years after
    jds = jd0 + np.arange(0, sim_time*365.25/time_grid)*time_grid
    frs = jds * 0
    return satellites.sgp4(jds, frs)


def simulate(debris, to_satcat, q, seed, sim_time=20, time_grid=5, t0=8073.603992389981, Lcube=10):
    np.random.seed(seed)
    # Runs the sgp4 simulation
    print("Precomputing the ephemerides")
    e, r, v = precompute_eph_sgp4( 
        np.array(debris)[to_satcat], sim_time=sim_time, time_grid=time_grid, t0=t0+np.random.random()*time_grid)
    n_collisions = 0
    print("Evaluating collisions")
    # We assume all satellites are valid at the starting epoch
    undecayed = set(np.arange(r.shape[0]))
    for i in range(r.shape[1]):
        # If signalled from the sgp4, we remove the indices of the decayed satellites
        decayed = set(np.where(e[:, i] > 0)[0])
        undecayed = undecayed - decayed
        undecayed_l = np.array([j for j in undecayed])
        # We detect all satellites couples in the same cube of Lcube km size
        collision = cubes(r[undecayed_l, i, :], cube_dimension=Lcube)
        for pair in collision:
            # we get the indexes in r,v
            idx1 = undecayed_l[pair[0]]
            idx2 = undecayed_l[pair[1]]
            # we store positions and velocities from r,v
            r1 = r[idx1, i, :]
            r2 = r[idx2, i, :]
            v1 = v[idx1, i, :]
            v2 = v[idx2, i, :]
            # we get the collision radiu from debris (indexed differently hence to_satcat is used)
            c_radius1 = debris[to_satcat[idx1]].collision_radius
            c_radius2 = debris[to_satcat[idx2]].collision_radius
            # Relative velocity
            Vrel = np.linalg.norm(v1-v2)
            # Collisional area of the couple (in km^2)
            sigma = np.pi*((c_radius1+c_radius2)/1000)**2
            # Volume of the cube (km^3)
            U = Lcube**3
            # We compute the spatial densities
            # Time spent in the cube (nd)
            tau1 = Lcube / np.linalg.norm(v1)
            tau2 = Lcube / np.linalg.norm(v2)
            # Orbital periods (s)
            T1 = period(np.linalg.norm(r1*1000),
                        np.linalg.norm(v1*1000), pk.MU_EARTH)
            T2 = period(np.linalg.norm(r2*1000),
                        np.linalg.norm(v2*1000), pk.MU_EARTH)
            # densities (from "Assessing collision algorithms for the newspace era" )
            s1 = 1./U
            s2 = 1./U
            # collision probability
            Pij = s1*s2*Vrel*sigma*U*time_grid*pk.DAY2SEC
            # Store
            if Pij > np.random.random():
                print(f"Collision! pair: {pair}, years: {i*time_grid/365.25}")
                n_collisions += 1
    q.put(n_collisions)


if __name__ == '__main__':
    n_cores = mp.cpu_count()
    print("Number of available cores is: ", n_cores)

    # We read the initial conditions used by the deterministic simulation
    with open("../../data/debris_simulation_ic.pk", "rb") as file:
        r_ic, v_ic, c_radius, to_satcat, satcat, debris = pkl.load(file)

    q = mp.Queue()
    keywords = {'time_grid': 5, 'Lcube': 10, 'sim_time': 20}

    n_jobs=100
    seeds  = np.random.randint(0, 123456789, (n_jobs, ))
    process_l = []
    for seed in seeds:
        p = mp.Process(target = simulate, args = (debris, to_satcat, q, seed), kwargs=keywords)
        process_l.append(p)
        p.start()
    for p in process_l:
        p.join()
    results = []
    for p in process_l:
        results.append(q.get())
    print("p:", results)
    print("Average: ", np.mean(results))
    print("Min: ", np.min(results))
    print("Max: ", np.max(results))

