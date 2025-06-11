# Copyright 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the cascade library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

# This little helper returns the heyoka expression for the density using
# an exponential fit
import typing
import numpy as np
import heyoka as hy
from cascade.dynamics import kepler

def _compute_atmospheric_density(h):
    """
    Returns the heyoka expression for the atmosheric density in kg.m^3.
    Input is the altitude in m.
    """
    

    # This array is produced by fitting
    best_x = np.array(
        [
            1.01709935e-06,
            7.86443375e-01,
            7.50341883e-09,
            8.63934252e-14,
            4.63822910e-02,
            1.86080048e-01,
            2.48667176e-02,
            4.81080852e-03,
            5.01594516e00,
            2.28429809e01,
            4.27957829e00,
            1.56291673e-01,
        ]
    )
    p1 = np.array(best_x[:4])
    p2 = np.array(best_x[4:8]) / 1000
    p3 = np.array(best_x[8:]) * 1000
    retval = 0.0
    for alpha, beta, gamma in zip(p1, p2, p3):
        retval += alpha * hy.exp(-(h - gamma) * beta)
    return retval

def ECI2ECEF(days_since_J2000):
    """
    This function returns the rotation matrix from ECI to ECEF at a given number of days elapsed since J2000
    Args:
        - days_since_J2000 (`float`): days elapsed since J2000

    Returns:
        - `list`: Earth rotation matrix
    """
    era = (
        2
        * np.pi
        * (0.7790572732640 + 1.00273781191135448 * days_since_J2000)       # Earth rotation angle
    )
    R = [[hy.cos(era), hy.sin(era), 0], [-hy.sin(era), hy.cos(era), 0], [0, 0, 1]]
    return R


def _compute_density_thermonets(r, f107, f107a, ap):
    """
    Returns the heyoka expression for the atmosheric density in kg.m^3, computed through ThermoNets
    """

    # days elapsed since the reference epoch J2000 (1st Jan 2000 12:00)
    days_since_J2000 = hy.time / 86400

    doy = days_since_J2000
    
    
    xyz_ecef = np.matmul(ECI2ECEF(days_since_J2000), r)      # matrix multiplication   r_ECEF = R(ECI2ECEF) @ r_ECI

    h, lat, lon = hy.model.cart2geo([xyz_ecef[0], xyz_ecef[1], xyz_ecef[2]])     # compute geodetic cooordinates [h is in meters]

    density_nn = hy.model.nrlmsise00_tn(geodetic=[h / 1000, lat, lon], f107=f107, f107a=f107a, ap=ap, time_expr=doy)     # compute density [h must be in kilometers]

    return density_nn


def simple_earth(
    J2: bool = True,
    J3: bool = False,
    J4: bool = False,
    C22S22: bool = True,
    sun: bool = False,
    moon: bool = False,
    SRP: bool = False,
    drag: bool = True,
    thermonets: bool = False,
) -> typing.List[typing.Tuple[hy.expression, hy.expression]]:
    """Perturbed dynamics around the Earth.

    Returns heyoka expressions to be used as dynamics in :class:`~cascade.sim` and corresponding
    to the Earth orbital environment as perturbed by selectable term (all in SI units).

    The equations are taken from those used during the ESA Kelvins competition
    "Space Debris: the Origin" adding a drag term.

    The reference frame used is the EME2000 and thus a simulation time of zero will refer to the epoch
    2000 JAN 01 12:00:00, or JD 2451545.0

    .. note::
       The equations are largely derived from:
       Celletti, Alessandra, et al. "Dynamical models and the onset of chaos in space debris."
       International Journal of Non-Linear Mechanics 90 (2017): 147-163. (`arxiv <https://arxiv.org/pdf/1612.08849.pdf>`_)

    .. note::
       If *drag* is active, the BSTAR coefficient (SI units) of the object must be passed as a first
       simulation parameter in :class:`~cascade.sim`, if *SRP* is active, the term Cr*AOM (SI units) must be also passed as a
       simulation parameter (after BSTAR if present), being Cr the coefficient of reflectivity of the spacecraft and AOM the 
       area-to-mass-ratio.


    Args:
        J2 (bool, optional): adds the Earth J2 spherical harmonic (C20 Stokes' coefficient). Defaults to True.
        J3 (bool, optional): adds the Earth J3 spherical harmonic (C30 Stokes' coefficient). Defaults to False.
        J4 (bool, optional): adds the Earth J4 spherical harmonic (C40 Stokes' coefficient). Defaults to False.
        C22S22 (bool, optional): adds the Earth C22 and S22 Stokes' coefficients. Defaults to True.
        sun (bool, optional): adds the Sun gravity. Defaults to False.
        moon (bool, optional): adds the Moon gravity. Defaults to False.
        SRP (bool, optional): adds the solar radiation pressure. Defaults to False.
        drag (bool, optional): adds the drag acceleration (atmosphere is modelled via a fitted isotropic NRLMSISE00). Defaults to True.

    Returns:
        The dynamics in SI units. Can be used to instantiate a :class:`~cascade.sim`.
    """
    

    # constants (final underscore reminds us its not SI)
    GMe_ = 3.986004407799724e5  # [km^3/sec^2]
    GMo_ = 1.32712440018e11  # [km^3/sec^2]
    GMm_ = 4.9028e3  # [km^3/sec^2]
    Re_ = 6378.1363  # [km]
    
    
    
    theta_g = (np.pi / 180) * 280.4606  # [rad] # This value defines the rotation of the Earth fixed system at t0
    nu_e = (np.pi / 180) * (4.178074622024230e-3)  # [rad/sec] # This value represents the Earth spin angular velocity.
    nu_o = (np.pi / 180) * (1.1407410259335311e-5)  # [rad/sec]
    nu_ma = (np.pi / 180) * (1.512151961904581e-4)  # [rad/sec]
    nu_mp = (np.pi / 180) * (1.2893925235125941e-6)  # [rad/sec]
    nu_ms = (np.pi / 180) * (6.128913003523574e-7)  # [rad/sec]
    alpha_o_ = 1.49619e8  # 1 AU in [km]
    epsilon = (np.pi / 180) * 23.4392911  # [rad]
    phi_o = (np.pi / 180) * 357.5256  # [rad]
    Omega_plus_w = (np.pi / 180) * 282.94  # [rad]
    PSRP_ = 4.56e-3  # [kg/(km*sec^2)]

    # Dynamical variables.
    x, y, z, vx, vy, vz = hy.make_vars("x", "y", "z", "vx", "vy", "vz")

    # Create Keplerian dynamics in SI units.
    Re_SI = Re_ * 1000
    GMe_SI = GMe_ * 1e9
    dyn = kepler(mu=GMe_SI)

    # Define the radius squared
    r2 = x**2 + y**2 + z**2
    # Define the radius
    r = hy.sqrt(r2)

    
    if J2:
        J2_adim = -0.1082635854 * 1e-2             # from 'https://en.wikipedia.org/wiki/Geopotential_spherical_harmonic_model'
        J2 = - J2_adim * GMe_SI * Re_SI**2
        u_J2 = J2/(2*r**5)*(3*z**2 - r2)
        fJ2x = - hy.diff(u_J2,x)
        fJ2y = - hy.diff(u_J2,y)
        fJ2z = - hy.diff(u_J2,z)
        dyn[3] = (dyn[3][0], dyn[3][1] + fJ2x)
        dyn[4] = (dyn[4][0], dyn[4][1] + fJ2y)
        dyn[5] = (dyn[5][0], dyn[5][1] + fJ2z)

    if J3:
        J3_adim = 0.2532435346 * 1e-5             # from 'https://en.wikipedia.org/wiki/Geopotential_spherical_harmonic_model'
        J3 = - J3_adim * GMe_SI * Re_SI**3
        u_J3 = J3*z/(2*r**7)*(5*z**2-3*r2)
        fJ3x = - hy.diff(u_J3,x)
        fJ3y = - hy.diff(u_J3,y)
        fJ3z = - hy.diff(u_J3,z)
        dyn[3] = (dyn[3][0], dyn[3][1] + fJ3x)
        dyn[4] = (dyn[4][0], dyn[4][1] + fJ3y)
        dyn[5] = (dyn[5][0], dyn[5][1] + fJ3z)
    
    if J4:
        J4_adim = 0.1619331205*1e-5               # from 'https://en.wikipedia.org/wiki/Geopotential_spherical_harmonic_model'
        J4 = - J4_adim * GMe_SI * Re_SI**4
        u_J4 = J4/8*(35*z**4 - 30*r2*z**2 + 3*r2**2)/r**9
        fJ4x = - hy.diff(u_J4,x)
        fJ4y = - hy.diff(u_J4,y)
        fJ4z = - hy.diff(u_J4,z)  
        dyn[3] = (dyn[3][0], dyn[3][1] + fJ4x)
        dyn[4] = (dyn[4][0], dyn[4][1] + fJ4y)
        dyn[5] = (dyn[5][0], dyn[5][1] + fJ4z) 

    if C22S22:
        C22 = 2.43914352398e-6
        S22 = -1.40016683654e-6

        X = x * hy.cos(theta_g + nu_e * hy.time) + y * hy.sin(theta_g + nu_e * hy.time)
        Y = -x * hy.sin(theta_g + nu_e * hy.time) + y * hy.cos(theta_g + nu_e * hy.time)
        Z = z

        C22term1 = (
            5 * GMe_SI * (Re_SI**2) * np.sqrt(15) * C22 / (2 * r2 ** (7 / 2))
        )
        C22term2 = GMe_SI * (Re_SI**2) * np.sqrt(15) * C22 / (r2 ** (5 / 2))
        fC22X = C22term1 * X * (Y**2 - X**2) + C22term2 * X
        fC22Y = C22term1 * Y * (Y**2 - X**2) - C22term2 * Y
        fC22Z = C22term1 * Z * (Y**2 - X**2)

        S22term1 = 5 * GMe_SI * (Re_SI**2) * np.sqrt(15) * S22 / (r2 ** (7.0 / 2))
        S22term2 = GMe_SI * (Re_SI**2) * np.sqrt(15) * S22 / (r2 ** (5.0 / 2))
        fS22X = -S22term1 * (X**2) * Y + S22term2 * Y
        fS22Y = -S22term1 * X * (Y**2) + S22term2 * X
        fS22Z = -S22term1 * X * Y * Z

        fC22x = fC22X * hy.cos(theta_g + nu_e * hy.time) - fC22Y * hy.sin(
            theta_g + nu_e * hy.time
        )
        fC22y = fC22X * hy.sin(theta_g + nu_e * hy.time) + fC22Y * hy.cos(
            theta_g + nu_e * hy.time
        )
        fC22z = fC22Z

        fS22x = fS22X * hy.cos(theta_g + nu_e * hy.time) - fS22Y * hy.sin(
            theta_g + nu_e * hy.time
        )
        fS22y = fS22X * hy.sin(theta_g + nu_e * hy.time) + fS22Y * hy.cos(
            theta_g + nu_e * hy.time
        )
        fS22z = fS22Z

        dyn[3] = (dyn[3][0], dyn[3][1] + fC22x + fS22x)
        dyn[4] = (dyn[4][0], dyn[4][1] + fC22y + fS22y)
        dyn[5] = (dyn[5][0], dyn[5][1] + fC22z + fS22z)

    if sun or SRP:
        # We compute the Sun's position
        lo = phi_o + nu_o * hy.time
        lambda_o = (
            Omega_plus_w
            + lo
            + (np.pi / 180)
            * ((6892 / 3600) * hy.sin(lo) + (72 / 3600) * hy.sin(2 * lo))
        )
        ro = (149.619 - 2.499 * hy.cos(lo) - 0.021 * hy.cos(2 * lo)) * (10**9)  # [m]
        Xo = ro * hy.cos(lambda_o)
        Yo = ro * hy.sin(lambda_o) * np.cos(epsilon)
        Zo = ro * hy.sin(lambda_o) * np.sin(epsilon)
        magRo2 = Xo**2 + Yo**2 + Zo**2
        magRRo2 = (x - Xo) ** 2 + (y - Yo) ** 2 + (z - Zo) ** 2

    if sun:
        # We add Sun's gravity
        GMo_SI = GMo_ * 1e9
        fSunX = -GMo_SI * (
            (x - Xo) / (magRRo2 ** (3.0 / 2)) + Xo / (magRo2 ** (3.0 / 2))
        )
        fSunY = -GMo_SI * (
            (y - Yo) / (magRRo2 ** (3.0 / 2)) + Yo / (magRo2 ** (3.0 / 2))
        )
        fSunZ = -GMo_SI * (
            (z - Zo) / (magRRo2 ** (3.0 / 2)) + Zo / (magRo2 ** (3.0 / 2))
        )

        dyn[3] = (dyn[3][0], dyn[3][1] + fSunX)
        dyn[4] = (dyn[4][0], dyn[4][1] + fSunY)
        dyn[5] = (dyn[5][0], dyn[5][1] + fSunZ)

    if moon:
        # Moon's position
        phi_m = nu_o * hy.time
        phi_ma = nu_ma * hy.time
        phi_mp = nu_mp * hy.time
        phi_ms = nu_ms * hy.time
        L0 = phi_mp + phi_ma + (np.pi / 180) * 218.31617
        lm = phi_ma + (np.pi / 180) * 134.96292
        llm = phi_m + (np.pi / 180) * 357.5256
        Fm = phi_mp + phi_ma + phi_ms + (np.pi / 180) * 93.27283
        Dm = phi_mp + phi_ma - phi_m + (np.pi / 180) * 297.85027

        # Careful, here we multiply by 1000 as the units are meters and not km as in the original paper where the eqs were presented.  
        rm = (
            385000
            - 20905 * hy.cos(lm)
            - 3699 * hy.cos(2 * Dm - lm)
            - 2956 * hy.cos(2 * Dm)
            - 570 * hy.cos(2 * lm)
            + 246 * hy.cos(2 * lm - 2 * Dm)
            - 205 * hy.cos(llm - 2 * Dm)
            - 171 * hy.cos(lm + 2 * Dm)
            - 152 * hy.cos(lm + llm - 2 * Dm)
        ) * 1000

        lambda_m = L0 + (np.pi / 180) * (
            (22640 / 3600) * hy.sin(lm)
            + (769 / 3600) * hy.sin(2 * lm)
            - (4856 / 3600) * hy.sin(lm - 2 * Dm)
            + (2370 / 3600) * hy.sin(2 * Dm)
            - (668 / 3600) * hy.sin(llm)
            - (412 / 3600) * hy.sin(2 * Fm)
            - (212 / 3600) * hy.sin(2 * lm - 2 * Dm)
            - (206 / 3600) * hy.sin(lm + llm - 2 * Dm)
            + (192 / 3600) * hy.sin(lm + 2 * Dm)
            - (165 / 3600) * hy.sin(llm - 2 * Dm)
            + (148 / 3600) * hy.sin(lm - llm)
            - (125 / 3600) * hy.sin(Dm)
            - (110 / 3600) * hy.sin(lm + llm)
            - (55 / 3600) * hy.sin(2 * Fm - 2 * Dm)
        )

        Bm = (np.pi / 180) * (
            (18520 / 3600)
            * hy.sin(
                Fm
                + lambda_m
                - L0
                + (np.pi / 180)
                * ((412 / 3600) * hy.sin(2 * Fm) + (541 / 3600) * hy.sin(llm))
            )
            - (526 / 3600) * hy.sin(Fm - 2 * Dm)
            + (44 / 3600) * hy.sin(lm + Fm - 2 * Dm)
            - (31 / 3600) * hy.sin(-lm + Fm - 2 * Dm)
            - (25 / 3600) * hy.sin(-2 * lm + Fm)
            - (23 / 3600) * hy.sin(llm + Fm - 2 * Dm)
            + (21 / 3600) * hy.sin(-lm + Fm)
            + (11 / 3600) * hy.sin(-llm + Fm - 2 * Dm)
        )

        Xm = hy.cos(Bm) * hy.cos(lambda_m) * rm
        Ym = (
            -np.sin(epsilon) * hy.sin(Bm) * rm
            + np.cos(epsilon) * hy.cos(Bm) * hy.sin(lambda_m) * rm
        )
        Zm = (
            np.cos(epsilon) * hy.sin(Bm) * rm
            + hy.cos(Bm) * np.sin(epsilon) * hy.sin(lambda_m) * rm
        )

        # We add Moon's gravity
        GMm_SI = GMm_ * 1e9
        magRm2 = Xm**2 + Ym**2 + Zm**2
        magRRm2 = (x - Xm) ** 2 + (y - Ym) ** 2 + (z - Zm) ** 2
        fMoonX = -GMm_SI * (
            (x - Xm) / (magRRm2 ** (3.0 / 2)) + Xm / (magRm2 ** (3.0 / 2))
        )
        fMoonY = -GMm_SI * (
            (y - Ym) / (magRRm2 ** (3.0 / 2)) + Ym / (magRm2 ** (3.0 / 2))
        )
        fMoonZ = -GMm_SI * (
            (z - Zm) / (magRRm2 ** (3.0 / 2)) + Zm / (magRm2 ** (3.0 / 2))
        )

        dyn[3] = (dyn[3][0], dyn[3][1] + fMoonX)
        dyn[4] = (dyn[4][0], dyn[4][1] + fMoonY)
        dyn[5] = (dyn[5][0], dyn[5][1] + fMoonZ)

    if drag:
        # Adds the drag force.
        BSTAR = hy.par[0]
        magv2 = vx**2 + vy**2 + vz**2
        magv = hy.sqrt(magv2)

        if thermonets:
            f107 = hy.par[1]
            f107a = hy.par[2]
            ap = hy.par[3]
            density = _compute_density_thermonets(r = [x, y, z], f107 = f107, f107a = f107a, ap = ap)      # time must be seconds elapsed since J2000
        else:
            altitude = r - Re_SI   # Here we consider a spherical Earth ... would be easy to account for the oblateness effect.
            density = _compute_atmospheric_density(altitude)

        ref_density = 0.1570 / Re_SI
        fdrag = density / ref_density * BSTAR * magv
        fdragx = -fdrag * vx
        fdragy = -fdrag * vy
        fdragz = -fdrag * vz
        dyn[3] = (dyn[3][0], dyn[3][1] + fdragx)
        dyn[4] = (dyn[4][0], dyn[4][1] + fdragy)
        dyn[5] = (dyn[5][0], dyn[5][1] + fdragz)

    
    if SRP:
        if drag and thermonets:
            k = hy.par[4]                 # c_r * A / m    where c_r is the reflectivity, A is the area facing the Sun and m is the S/C mass (D.A. Vallado - Fundamentals of Astrodynamics and Applications - 4th Edition - Section 8.6.4)
        elif drag and not thermonets:
            k = hy.par[1]
        elif not drag:
            k = hy.par[0]
        
        PSRP_SI = PSRP_ / 1000.0  # [kg/(m*sec^2)]
        alpha_o_SI = alpha_o_ * 1000.0  # 1 AU in [m]
           
        SRPterm = (
            k * PSRP_SI * (alpha_o_SI**2) / (magRRo2 ** (3.0 / 2))
        )
        fSRPX = SRPterm * (x - Xo)
        fSRPY = SRPterm * (y - Yo)
        fSRPZ = SRPterm * (z - Zo)
        dyn[3] = (dyn[3][0], dyn[3][1] + fSRPX)
        dyn[4] = (dyn[4][0], dyn[4][1] + fSRPY)
        dyn[5] = (dyn[5][0], dyn[5][1] + fSRPZ)

    return dyn