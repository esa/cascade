#!/usr/bin/env python
# coding: utf-8

# In[1]:
import heyoka as hy
from tqdm.notebook import tqdm
from copy import deepcopy
import cascade as csc
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle as pkl
import json
import numpy as np
import pykep as pk
print("Imports", flush=True)


# In[2]:


# # We import the simulation initial conditions and atmospheric density model
# The files needed are:
# * **debris_simulation_ic.pk** - created by the notebook 1
# * **best_fit_density.pk** - created by the notebook 1b

# In[3]:
print("Data load", flush=True)
with open("data/debris_simulation_ic.pk", "rb") as file:
    r_ic, v_ic, c_radius, to_satcat, satcat, debris = pkl.load(file)


# * **r**: contains the initial position of all satellites to be simulated (SI units)
# * **v**: contains the initial velocity of all satellites to be simulated (SI units)
# * **radius**: contains all the estimated radii for the various objects (in meters)
# * **to_satcat**: contains the indexes in the satcat of the corresponding r,v,radius entry
# * **satcat**: the satcat
# * **debris**: the corresponding pykep planets

# In[7]:
# Dynamical variables.
x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")

# Constants.
GMe = pk.MU_EARTH
Re = pk.EARTH_RADIUS

# Create Keplerian dynamics.
dyn = csc.dynamics.kepler(mu=GMe)


# In[8]:
# Add the J2 terms.
magr2 = hy.sum_sq([x, y, z])
J2term1 = GMe*(Re**2)*np.sqrt(5)*C20/(2*magr2**(1./2))
J2term2 = 3/(magr2**2)
J2term3 = 15*(z**2)/(magr2**3)
fJ2x = J2term1*x*(J2term2 - J2term3)
fJ2y = J2term1*y*(J2term2 - J2term3)
fJ2z = J2term1*z*(3*J2term2 - J2term3)
dyn[3] = (dyn[3][0], dyn[3][1] + fJ2x)
dyn[4] = (dyn[4][0], dyn[4][1] + fJ2y)
dyn[5] = (dyn[5][0], dyn[5][1] + fJ2z)

# # We setup the simulation

# In[11]:
csc.set_logger_level_info()


# In[12]:
def remove_particle(idx, r_ic, v_ic,, to_satcat, c_radius):
    r_ic = np.delete(r_ic, idx, axis=0)
    v_ic = np.delete(v_ic, idx, axis=0)
    to_satcat = np.delete(to_satcat, idx, axis=0)
    c_radius = np.delete(c_radius, idx, axis=0)
    return r_ic, v_ic,, to_satcat, c_radius


min_radius = pk.EARTH_RADIUS+150000.
inside_the_radius = np.where(np.linalg.norm(r_ic, axis=1) < min_radius)[0]
print("Removing orbiting objects:", flush=True)
for idx in inside_the_radius:
    print(satcat[to_satcat[idx]]["OBJECT_NAME"],
          "-", satcat[to_satcat[idx]]["OBJECT_ID"])
r_ic, v_ic, to_satcat, c_radius = remove_particle(
    inside_the_radius, r_ic, v_ic, to_satcat, c_radius)
# In[14]:
print("Building the simulation:", flush=True)
sim = csc.sim(r_ic[:, 0], r_ic[:, 1], r_ic[:, 2], v_ic[:, 0], v_ic[:, 1], v_ic[:, 2],
              c_radius, 0.23 * 806.81, dyn=dyn, c_radius=min_radius)


# # We run the simulation

# In[15]:
new_r_ic = deepcopy(r_ic)
new_v_ic = deepcopy(v_ic)
new_c_radius = deepcopy(c_radius)
new_BSTARS = deepcopy(BSTARS)
new_to_satcat = deepcopy(to_satcat)


# In[ ]:
final_t = 365.25 * pk.DAY2SEC * 20

print("Starting the simulation:", flush=True)
current_year = 0
while sim.time < final_t:
    years_elapsed = sim.time * pk.SEC2DAY // 365.25
    if years_elapsed == current_year:
        with open("out/year_"+str(current_year)+".pk", "wb") as file:
            pkl.dump((new_r_ic, new_v_ic, new_c_radius, new_to_satcat), file)
        current_year += 1

    oc = sim.step()
    if oc == csc.outcome.collision:
        pi, pj = sim.interrupt_info
        # We log the event to file
        satcat_idx1 = to_satcat[pi]
        satcat_idx2 = to_satcat[pj]
        days_elapsed = sim.time * pk.SEC2DAY
        with open("out/collision_log.txt", "a") as file_object:
            file_object.write(
                f"{days_elapsed}, {satcat_idx1}, {satcat_idx2}, {sim.x[pi]}, {sim.y[pi]}, {sim.z[pi]}, {sim.vx[pi]}, {sim.vy[pi]}, {sim.vz[pi]}, {sim.x[pj]}, {sim.y[pj]}, {sim.z[pj]}, {sim.vx[pj]}, {sim.vy[pj]}, {sim.vz[pj]}\n")
        # We log the event to screen
        o1, o2 = satcat[satcat_idx1]["OBJECT_TYPE"], satcat[satcat_idx2]["OBJECT_TYPE"]
        s1, s2 = satcat[satcat_idx1]["RCS_SIZE"], satcat[satcat_idx2]["RCS_SIZE"]
        print(
            f"\nCollision detected, {o1} ({s1}) and {o2} ({s2}) after {days_elapsed} days\n")
        # We remove the objects and restart the simulation
        new_r_ic = np.vstack((sim.x, sim.y, sim.z)).transpose()
        new_v_ic = np.vstack((sim.vx, sim.vy, sim.vz)).transpose()
        new_r_ic, new_v_ic, new_to_satcat, new_c_radius = remove_particle(
            [pi, pj], new_r_ic, new_v_ic, new_to_satcat, new_c_radius)
        sim.set_new_state(new_r_ic[:, 0], new_r_ic[:, 1], new_r_ic[:, 2], new_v_ic[:, 0],
                          new_v_ic[:, 1], new_v_ic[:, 2], new_c_radius)

    elif oc == csc.outcome.reentry:
        pi = sim.interrupt_info
        # We log the event to file
        satcat_idx = to_satcat[pi]
        days_elapsed = sim.time * pk.SEC2DAY
        with open("out/decay_log.txt", "a") as file_object:
            file_object.write(f"{days_elapsed},{satcat_idx}\n")
        # We log the event to screen
        print(satcat[satcat_idx]["OBJECT_NAME"].strip(
        ) + ", " + satcat[satcat_idx]["OBJECT_ID"].strip() + ", ", days_elapsed, "REMOVED")
        # We remove the re-entered object and restart the simulation
        new_r_ic = np.vstack((sim.x, sim.y, sim.z)).transpose()
        new_v_ic = np.vstack((sim.vx, sim.vy, sim.vz)).transpose()
        new_r_ic, new_v_ic, new_to_satcat, new_c_radius = remove_particle(
            pi, new_r_ic, new_v_ic, new_to_satcat, new_c_radius)
        sim.set_new_state(new_r_ic[:, 0], new_r_ic[:, 1], new_r_ic[:, 2], new_v_ic[:, 0],
                          new_v_ic[:, 1], new_v_ic[:, 2], new_c_radius)
