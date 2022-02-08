#!/usr/bin/env python
# coding: utf-8

# In[1]:
print("Imports", flush = True)
import pykep as pk
import numpy as np
import json
import pickle as pkl

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


# In[2]:
import cascade as csc
from copy import deepcopy
from tqdm.notebook import tqdm
import heyoka as hy


# # We import the simulation initial conditions and atmospheric density model
# The files needed are:
# * **debris_simulation_ic.pk** - created by the notebook 1
# * **best_fit_density.pk** - created by the notebook 1b

# In[3]:
print("Data load", flush = True)
with open("data/debris_simulation_ic.pk", "rb") as file:
    r_ic,v_ic,c_radius,to_satcat,satcat,debris = pkl.load(file)


# * **r**: contains the initial position of all satellites to be simulated (SI units)
# * **v**: contains the initial velocity of all satellites to be simulated (SI units)
# * **radius**: contains all the estimated radii for the various objects (in meters)
# * **to_satcat**: contains the indexes in the satcat of the corresponding r,v,radius entry
# * **satcat**: the satcat
# * **debris**: the corresponding pykep planets

# In[4]:
with open("data/best_fit_density.pk", "rb") as file:
    best_x = pkl.load(file)


# In[5]:


# We need to create an array containing all B*
BSTARS = []
for idx in to_satcat:
    BSTARS.append(float(satcat[idx]["BSTAR"]))
# We put the BSTAR in SI units
BSTARS = np.array(BSTARS) / pk.EARTH_RADIUS
# We remove negative BSTARS setting the value to zero in those occasions
BSTARS[BSTARS<0] = 0.


# # We build the dynamical system to integrate

# In[6]:
# This little helper returns the heyoka expression for the density using
# the results from the data interpolation
def compute_density(h, best_x):
    """
    Returns the heyoka expression for the atmosheric density in kg.m^3. 
    Input is the altitude in m. 
    (when we fitted km were used here we change as to allow better expressions)
    """
    p1 = np.array(best_x[:4])
    p2 = np.array(best_x[4:8]) / 1000
    p3 = np.array(best_x[8:]) * 1000
    retval = 0.
    for alpha,beta, gamma in zip(p1,p2, p3):
        retval += alpha*hy.exp(-(h-gamma)*beta)
    return retval


# In[7]:
# Dynamical variables.
x,y,z,vx,vy,vz = hy.make_vars("x","y","z","vx","vy","vz")

# Constants.
GMe = pk.MU_EARTH
C20 = -4.84165371736e-4
C22 = 2.43914352398e-6
S22 = -1.40016683654e-6
Re = pk.EARTH_RADIUS

# Create Keplerian dynamics.
dyn = csc.dynamics.kepler(mu = GMe)


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


# In[9]:
# Add the Earth's C22 and S22 terms.
# This value represents the rotation of the Earth fixed system at t0
theta_g = (np.pi/180)*280.4606 #[rad] 
# This value represents the magnitude of the Earth rotation
nu_e = (np.pi/180)*(4.178074622024230e-3) #[rad/sec]

X =  x*hy.cos(theta_g + nu_e*hy.time) + y*hy.sin(theta_g + nu_e*hy.time)
Y = -x*hy.sin(theta_g + nu_e*hy.time) + y*hy.cos(theta_g + nu_e*hy.time)
Z = z

C22term1 = 5*GMe*(Re**2)*np.sqrt(15)*C22/(2*magr2**(7./2))
C22term2 = GMe*(Re**2)*np.sqrt(15)*C22/(magr2**(5./2))
fC22X = C22term1*X*(Y**2 - X**2) + C22term2*X
fC22Y = C22term1*Y*(Y**2 - X**2) - C22term2*Y
fC22Z = C22term1*Z*(Y**2 - X**2)

S22term1 = 5*GMe*(Re**2)*np.sqrt(15)*S22/(magr2**(7./2))
S22term2 = GMe*(Re**2)*np.sqrt(15)*S22/(magr2**(5./2))
fS22X = -S22term1*(X**2)*Y + S22term2*Y
fS22Y = -S22term1*X*(Y**2) + S22term2*X
fS22Z = -S22term1*X*Y*Z

fC22x = fC22X*hy.cos(theta_g + nu_e*hy.time) - fC22Y*hy.sin(theta_g + nu_e*hy.time)
fC22y = fC22X*hy.sin(theta_g + nu_e*hy.time) + fC22Y*hy.cos(theta_g + nu_e*hy.time)
fC22z = fC22Z

fS22x = fS22X*hy.cos(theta_g + nu_e*hy.time) - fS22Y*hy.sin(theta_g + nu_e*hy.time)
fS22y = fS22X*hy.sin(theta_g + nu_e*hy.time) + fS22Y*hy.cos(theta_g + nu_e*hy.time)
fS22z = fS22Z

dyn[3] = (dyn[3][0], dyn[3][1] + fC22x + fS22x)
dyn[4] = (dyn[4][0], dyn[4][1] + fC22y + fS22y)
dyn[5] = (dyn[5][0], dyn[5][1] + fC22z + fS22z)


# In[10]:
# Adds the drag force.
magv2 = hy.sum_sq([vx, vy, vz])
magv = hy.sqrt(magv2)
# Here we consider a spherical Earth ... would be easy to account for the oblateness effect.
altitude = (hy.sqrt(magr2) - Re)
density = compute_density(altitude, best_x)
ref_density = 0.1570 / Re
fdrag = density / ref_density * hy.par[0] * magv
fdragx = - fdrag * vx
fdragy = - fdrag * vy
fdragz = - fdrag * vz
dyn[3] = (dyn[3][0], dyn[3][1] + fdragx)
dyn[4] = (dyn[4][0], dyn[4][1] + fdragy)
dyn[5] = (dyn[5][0], dyn[5][1] + fdragz)


# # We setup the simulation

# In[11]:
csc.set_logger_level_info()


# In[12]:
def remove_particle(idx, r_ic, v_ic, BSTARS,to_satcat, c_radius):
    r_ic = np.delete(r_ic, idx, axis=0)
    BSTARS = np.delete(BSTARS, idx, axis=0)
    v_ic = np.delete(v_ic, idx, axis=0)
    to_satcat = np.delete(to_satcat, idx, axis=0)
    c_radius = np.delete(c_radius, idx, axis=0)
    return r_ic, v_ic, BSTARS, to_satcat, c_radius


# In[13]:
# Before starting we need to remove all particles inside our playing field
min_radius = pk.EARTH_RADIUS+150000.
inside_the_radius = np.where(np.linalg.norm(r_ic,axis=1) < min_radius)[0]
print("Removing orbiting objects:", flush=True)
for idx in inside_the_radius:
    print(satcat[to_satcat[idx]]["OBJECT_NAME"], "-", satcat[to_satcat[idx]]["OBJECT_ID"])
r_ic, v_ic, BSTARS,to_satcat, c_radius = remove_particle(inside_the_radius, r_ic, v_ic, BSTARS,to_satcat, c_radius)
# In[14]:
print("Building the simulation:", flush = True)
sim = csc.sim(r_ic[:,0],r_ic[:,1],r_ic[:,2],v_ic[:,0],v_ic[:,1],v_ic[:,2],c_radius,0.23 * 806.81,dyn=dyn,pars=[BSTARS], c_radius=min_radius)


# # We run the simulation

# In[15]:
new_r_ic  =deepcopy(r_ic)
new_v_ic  =deepcopy(v_ic)
new_c_radius  =deepcopy(c_radius)
new_BSTARS  =deepcopy(BSTARS)
new_to_satcat  =deepcopy(to_satcat)


# In[ ]:
final_t = 365.25 * pk.DAY2SEC * 20

print("Starting the simulation:", flush = True)
while sim.time < final_t:
    orig_time = sim.time
    
    oc = sim.step()   
    if oc == csc.outcome.collision:
        pi, pj = sim.interrupt_info
        # We log the event to file
        satcat_idx1 = to_satcat[pi]
        satcat_idx2 = to_satcat[pj]
        days_elapsed = sim.time * pk.SEC2DAY
        with open("out/collision_log.txt", "a") as file_object:
            file_object.write(f"{days_elapsed}, {satcat_idx1}, {satcat_idx2}, {sim.vx[pi]}, {sim.vy[pi]}, {sim.vz[pi]}, {sim.vx[pj]}, {sim.vy[pj]}, {sim.vz[pj]}\n")
        # We log the event to screen
        o1, o2 = satcat[satcat_idx1]["OBJECT_TYPE"], satcat[satcat_idx2]["OBJECT_TYPE"]
        s1, s2 = satcat[satcat_idx1]["RCS_SIZE"], satcat[satcat_idx2]["RCS_SIZE"]
        print(f"\nCollision detected, {o1} ({s1}) and {o2} ({s2}) after {days_elapsed} days\n")
        # We remove the objects and restart the simulation
        new_r_ic = np.vstack((sim.x,sim.y,sim.z)).transpose()
        new_v_ic = np.vstack((sim.vx,sim.vy,sim.vz)).transpose()
        new_r_ic, new_v_ic, new_BSTARS,new_to_satcat, new_c_radius = remove_particle([pi, pj], new_r_ic, new_v_ic, new_BSTARS,new_to_satcat, new_c_radius)
        sim.set_new_state(new_r_ic[:,0],new_r_ic[:,1],new_r_ic[:,2],new_v_ic[:,0],new_v_ic[:,1],new_v_ic[:,2],new_c_radius, pars=[new_BSTARS])

    elif oc == csc.outcome.reentry:
        pi = sim.interrupt_info
        # We log the event to file
        satcat_idx = to_satcat[pi]
        days_elapsed = sim.time * pk.SEC2DAY
        with open("out/decay_log.txt", "a") as file_object:
            file_object.write(f"{days_elapsed},{satcat_idx}\n")
        # We log the event to screen
        print(satcat[satcat_idx]["OBJECT_NAME"].strip() + ", " + satcat[satcat_idx]["OBJECT_ID"].strip() + ", ", days_elapsed, "REMOVED")
        # We remove the re-entered object and restart the simulation
        new_r_ic = np.vstack((sim.x,sim.y,sim.z)).transpose()
        new_v_ic = np.vstack((sim.vx,sim.vy,sim.vz)).transpose()
        new_r_ic, new_v_ic, new_BSTARS,new_to_satcat, new_c_radius = remove_particle(pi, new_r_ic, new_v_ic, new_BSTARS,new_to_satcat, new_c_radius)
        sim.set_new_state(new_r_ic[:,0],new_r_ic[:,1],new_r_ic[:,2],new_v_ic[:,0],new_v_ic[:,1],new_v_ic[:,2],new_c_radius, pars=[new_BSTARS])



