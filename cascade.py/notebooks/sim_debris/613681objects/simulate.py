#!/usr/bin/env python
# coding: utf-8
# In[0]:
import sys, os
print("Parse the script inputs", flush=True)
try:
    #seed = int(sys.argv[1])
    #cpus = int(sys.argv[2])
    seed = 0
    cpus = 64
    folder_name = 'out'+str(seed)
    print("output folder: ", folder_name)
    #os.mkdir(folder_name)
except: 
    print("USAGE:")
    print("{0} <seed> <cpus>".format(sys.argv[0]))
    sys.exit(2)

# In[1]:
print("Importing external dependencies ... numpy etc", flush=True)
import heyoka as hy
from tqdm.notebook import tqdm
from copy import deepcopy
import cascade as csc
import csv
import pickle as pkl
import numpy as np
import pykep as pk
import time


csc.set_nthreads(cpus)
np.random.seed(seed)

# In[3]:
print("Importing the ic of the > 600000 orbiting objects", flush=True)
# # We import the simulation initial conditions and atmospheric density model
# The files needed are:
# * **initial_population_and_1cm_debris.csv** - from LADDS ( https://github.com/esa/LADDS/blob/main/data/initial_population_and_1cm_debris.csv)
# * **best_fit_density.pk** - created by the notebook 1b
print("Data load: ic, radius, BSTAR", flush=True)
x,y,z,vx,vy,vz,bstar,radius = [],[],[],[],[],[],[],[]
with open("../../data/initial_population_and_1cm_debris.csv", "r") as file:
    data = csv.reader(file)
    for row in data:
        if data.line_num>1:
            bstar.append(float(row[3]))
            radius.append(float(row[5]))
            x.append(float(row[7]))
            y.append(float(row[8]))
            z.append(float(row[9]))
            vx.append(float(row[10]))
            vy.append(float(row[11]))
            vz.append(float(row[12]))
ic = np.array([x,y,z,vx,vy,vz]) * 1000 # (now all in SI)
ic = ic.transpose() # shape is [n, 6]
bstar = np.array(bstar) / pk.EARTH_RADIUS # (now all in SI)
radius = np.array(radius) # (all in SI)

# In[4]:
# We import the density model
print("Data load: atmospheric density", flush=True)
with open("../../data/best_fit_density.pk", "rb") as file:
    atm_fit = pkl.load(file)

# In[6]:
# # We build the dynamical system to integrate

# This little helper returns the heyoka expression for the density using
# the results from the data interpolation
print("Building the dynamical system equations", flush=True)


def compute_density(h, atm_fit):
    """
    Returns the heyoka expression for the atmosheric density in kg.m^3. 
    Input is the altitude in m. 
    (when we fitted km were used here we change as to allow better expressions)
    """
    p1 = np.array(atm_fit[:4])
    p2 = np.array(atm_fit[4:8]) / 1000
    p3 = np.array(atm_fit[8:]) * 1000
    retval = 0.
    for alpha, beta, gamma in zip(p1, p2, p3):
        retval += alpha*hy.exp(-(h-gamma)*beta)
    return retval


# In[7]:
# Dynamical variables.
x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")

# Constants.
GMe = pk.MU_EARTH
C20 = -4.84165371736e-4
C22 = 2.43914352398e-6
S22 = -1.40016683654e-6
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


# In[9]:
# Add the Earth's C22 and S22 terms.
# This value represents the rotation of the Earth fixed system at t0
theta_g = (np.pi/180)*280.4606  # [rad]
# This value represents the magnitude of the Earth rotation
nu_e = (np.pi/180)*(4.178074622024230e-3)  # [rad/sec]

X = x*hy.cos(theta_g + nu_e*hy.time) + y*hy.sin(theta_g + nu_e*hy.time)
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

fC22x = fC22X*hy.cos(theta_g + nu_e*hy.time) - fC22Y * \
    hy.sin(theta_g + nu_e*hy.time)
fC22y = fC22X*hy.sin(theta_g + nu_e*hy.time) + fC22Y * \
    hy.cos(theta_g + nu_e*hy.time)
fC22z = fC22Z

fS22x = fS22X*hy.cos(theta_g + nu_e*hy.time) - fS22Y * \
    hy.sin(theta_g + nu_e*hy.time)
fS22y = fS22X*hy.sin(theta_g + nu_e*hy.time) + fS22Y * \
    hy.cos(theta_g + nu_e*hy.time)
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
density = compute_density(altitude, atm_fit)
ref_density = 0.1570 / pk.EARTH_RADIUS # (now in SI)
fdrag = density / ref_density * hy.par[0] * magv
fdragx = - fdrag * vx
fdragy = - fdrag * vy
fdragz = - fdrag * vz
dyn[3] = (dyn[3][0], dyn[3][1] + fdragx)
dyn[4] = (dyn[4][0], dyn[4][1] + fdragy)
dyn[5] = (dyn[5][0], dyn[5][1] + fdragz)

# In[13]:
# We remove all particles inside our playing field (min_radius)
min_radius = pk.EARTH_RADIUS+150000.
inside_the_radius = np.where(np.linalg.norm(ic[:,:3], axis=1) < min_radius+50000)[0]
print(f"Removing {len(inside_the_radius)} orbiting objects.", flush=True)
ic = np.delete(ic, inside_the_radius, 0)
radius = np.delete(radius, inside_the_radius)
bstar = np.delete(bstar, inside_the_radius)
print(f"Final size: {len(bstar)}")

ic_state = np.hstack([ic, radius.reshape((-1, 1))])
pars = bstar.reshape((-1, 1))
#np.savetxt("test_ic_613681.txt", ic_state.reshape(-1, 1))
#np.savetxt("test_par_613681.txt", pars)


# In[]
# # We setup the simulation
#----------------------------- We setup the simulation--------------------------------
print("Building the simulation:", flush=True)
sim = csc.sim(ic_state, 0.08 * 806.81, dyn=dyn, pars=pars*0., c_radius=min_radius/2.)

# csc.set_logger_level_info()
csc.set_logger_level_trace()


# In[ ]:
# # We run the simulation
#final_t = 365.25 * pk.DAY2SEC * 20
final_t = 10. * pk.DAY2SEC

print("Starting the simulation:", flush=True)
current_year = 0

# get the start time
st = time.time()

while sim.time < final_t:
    years_elapsed = sim.time * pk.SEC2DAY // 365.25

    oc = sim.step()
    print(sim.time* pk.SEC2DAY, flush=True)
    if oc == csc.outcome.collision:
        pi, pj = sim.interrupt_info
        # We log the event to file
        days_elapsed = sim.time * pk.SEC2DAY
        print(
            f"\n{days_elapsed}: Collision detected, {pi}-{pj} after {days_elapsed} days\n")
        # We remove the objects and restart the simulation
        ic = np.delete(ic, [pi,pj], 0)
        radius = np.delete(radius, [pi,pj])
        bstar = np.delete(bstar, [pi,pj])
        sim.set_new_state(ic[:, 0], ic[:, 1], ic[:, 2], ic[:, 3],
                          ic[:, 4], ic[:, 5], radius, pars=[bstar])

    elif oc == csc.outcome.reentry:
        pi = sim.interrupt_info
        # We log the event to file
        days_elapsed = sim.time * pk.SEC2DAY
        # We log the event to screen
        print(f"{days_elapsed}: Reentry event: {pi}")
        # We remove the re-entered object and restart the simulation
        ic = np.delete(ic, pi, 0)
        radius = np.delete(radius, pi)
        bstar = np.delete(bstar, pi)
        sim.set_new_state(ic[:, 0], ic[:, 1], ic[:, 2], ic[:, 3],
                          ic[:, 4], ic[:, 5], radius, pars=[bstar])
    

# get the end time
et = time.time()
# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
