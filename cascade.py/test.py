# Copyright 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the cascade.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class dynamics_test_case(_ut.TestCase):
    def runTest(self):
        self.test_simple_earth_api()
        self.test_kepler_equivalence()
        self.test_perturbation_magnitudes()

    def test_simple_earth_api(self):
        from .dynamics import simple_earth

        simple_earth(
            J2=True, J3=False, J4=False, C22S22=False, sun=False, moon=False, SRP=False, drag=False, thermonets=False
        )
        simple_earth(
            J2=True, J3=True, J4=False, C22S22=False, sun=False, moon=False, SRP=False, drag=False, thermonets=False
        )
        simple_earth(
            J2=True, J3=True, J4=True, C22S22=False, sun=False, moon=False, SRP=False, drag=False, thermonets=False
        )
        simple_earth(
            J2=True, J3=True, J4=True, C22S22=True, sun=False, moon=False, SRP=False, drag=False, thermonets=False
        )
        simple_earth(
            J2=True, J3=True, J4=True, C22S22=True, sun=True, moon=False, SRP=False, drag=False, thermonets=False
        )
        simple_earth(
            J2=True, J3=True, J4=True, C22S22=True, sun=True, moon=True, SRP=False, drag=False, thermonets=False
        )
        simple_earth(
            J2=True, J3=True, J4=True, C22S22=True, sun=True, moon=True, SRP=True, drag=False, thermonets=False
        )
        simple_earth(
            J2=True, J3=True, J4=True, C22S22=True, sun=True, moon=True, SRP=True, drag=True, thermonets=False
        )
        simple_earth(
            J2=True, J3=True, J4=True, C22S22=True, sun=True, moon=True, SRP=True, drag=True, thermonets=True
        )

    def test_kepler_equivalence(self):
        from .dynamics import simple_earth, kepler

        dyn1 = simple_earth(
            J2=False, C22S22=False, sun=False, moon=False, SRP=False, drag=False
        )
        dyn2 = kepler(mu=3.986004407799724e5 * 1e9)
        self.assertEqual(dyn1, dyn2)

    def test_perturbation_magnitudes(self):
        from .dynamics import simple_earth, kepler, _compute_density_thermonets
        import numpy as np
        from heyoka import cfunc, make_vars

        dynkep = simple_earth(
            J2=False,
            J3=False,
            J4=False,
            C22S22=False,
            sun=False,
            moon=False,
            SRP=False,
            drag=False,
            thermonets=False
        )
        dynJ2 = simple_earth(
            J2=True,
            J3=False,
            J4=False,
            C22S22=False,
            sun=False,
            moon=False,
            SRP=False,
            drag=False,
            thermonets=False
        )
        dynJ3 = simple_earth(
            J2=False,
            J3=True,
            J4=False,
            C22S22=False,
            sun=False,
            moon=False,
            SRP=False,
            drag=False,
            thermonets=False
        )
        dynJ4 = simple_earth(
            J2=False,
            J3=False,
            J4=True,
            C22S22=False,
            sun=False,
            moon=False,
            SRP=False,
            drag=False,
            thermonets=False
        )

        # Dynamical variables.
        x, y, z, f107, f107a, ap = make_vars("x", "y", "z", "f107","f107a","ap")
        dynkep_c = cfunc([dynkep[i][1] for i in [3, 4, 5]], vars=[x,y,z])
        dynJ2_c = cfunc([dynJ2[i][1] for i in [3, 4, 5]], vars=[x,y,z])
        dynJ3_c = cfunc([dynJ3[i][1] for i in [3, 4, 5]], vars=[x,y,z])
        dynJ4_c = cfunc([dynJ4[i][1] for i in [3, 4, 5]], vars=[x,y,z])

        # We compute the various acceleration magnitudes at 7000 km
        pos = np.array([7000000.0, 0.0, 0.0])
        acckep = dynkep_c(pos)
        accJ2 = dynJ2_c(pos) - dynkep_c(pos)
        accJ3 = dynJ3_c(pos) - dynkep_c(pos)
        accJ4 = dynJ4_c(pos) - dynkep_c(pos)

        # And check magnitudes
        self.assertTrue(np.linalg.norm(acckep) > 8)
        self.assertTrue(np.linalg.norm(accJ2) > 0.01)
        self.assertTrue(np.linalg.norm(accJ3) > 0.00001)
        self.assertTrue(np.linalg.norm(accJ4) > 0.00001)
        self.assertTrue(np.linalg.norm(acckep) < 10)
        self.assertTrue(np.linalg.norm(accJ2) < 0.1)
        self.assertTrue(np.linalg.norm(accJ3) < 0.0001)
        self.assertTrue(np.linalg.norm(accJ4) < 0.0001)

        # Check atmospheric density from ThermoNets
        # Space weather indices at J2000 (1st January 2000, 12:00:00 TT)
        f107_val = 129.9
        f107a_val = 166.2
        ap_val = 30

        density_func = _compute_density_thermonets(r=[x,y,z],f107=f107,f107a=f107a,ap=ap)
        density_c = cfunc([density_func], vars=[x,y,z,f107,f107a,ap])
        density = density_c([pos[0],pos[1],pos[2],f107_val,f107a_val,ap_val],time=0.)

        # Check magnitude
        self.assertTrue(density > 5e-14)
        self.assertTrue(density < 1e-13)


class sim_test_case(_ut.TestCase):
    def runTest(self):
        self.test_basic()
        self.test_remove_particles()
        self.test_set_new_state_pars()
        self.test_ct_api()
        self.test_conjunctions()

    def test_conjunctions(self):
        from . import sim
        import gc
        import numpy as np

        # NOTE: initial conditions corresponding to
        # 2 particles with polar conjunctions.
        r1 = (0.3342377271241684, 0.942488801930755, 0.0)
        v1 = (0.009424730938644767, -0.003342321565239028, 0.9999500004176654)
        r2 = (-0.15179985849820377, -0.988411251938142, 0.0)
        v2 = (6.052273379648475e-17, -9.295060541060909e-18, 1.000000000001)

        psize = 1.57e-8

        s = sim(
            [list(r1) + list(v1) + [psize], list(r2) + list(v2) + [psize]],
            0.23,
            conj_thresh=psize * 100000000,
        )

        cv0 = s.conjunctions
        self.assertEqual(len(cv0), 0)

        s.propagate_until(2.0)
        cv1 = s.conjunctions
        self.assertEqual(len(cv1), 1)

        s.propagate_until(20.0)
        cv2 = s.conjunctions
        self.assertEqual(len(cv2), 6)

        s.reset_conjunctions()
        self.assertEqual(len(s.conjunctions), 0)

        del s

        gc.collect()

        # "Use" the stored conjunction vectors
        # to test they are still valid.
        self.assertEqual(len(cv0), 0)

        self.assertEqual(len(cv1), 1)
        self.assertTrue(np.all(cv1["time"] == cv1["time"]))

        self.assertEqual(len(cv2), 6)
        self.assertTrue(np.all(cv2["dist"] == cv2["dist"]))

        # Make sure the conjunctions array are read-only.
        with self.assertRaises(ValueError) as cm:
            cv2["dist"][0] = 0

        with self.assertRaises(ValueError) as cm:
            cv2.resize((100,))

    def test_ct_api(self):
        from . import sim

        s = sim()

        s.ct = 1.1
        self.assertEqual(s.ct, 1.1)
        s.n_par_ct = 5
        self.assertEqual(s.n_par_ct, 5)

        with self.assertRaises(ValueError) as cm:
            s.ct = -1.1
        with self.assertRaises(ValueError) as cm:
            s.n_par_ct = 0

        self.assertEqual(s.ct, 1.1)
        self.assertEqual(s.n_par_ct, 5)

    def test_basic(self):
        from . import sim, dynamics, outcome
        import heyoka as hy
        import numpy as np
        import gc

        s = sim()
        self.assertEqual(s.state.shape, (0, 7))
        self.assertEqual(s.pars.shape, (0, 0))
        self.assertEqual(s.nparts, 0)
        self.assertEqual(s.time, 0.0)
        self.assertEqual(s.ct, 1.0)
        self.assertFalse(s.high_accuracy)
        self.assertFalse(s.compact_mode)
        self.assertEqual(s.npars, 0)
        self.assertEqual(s.reentry_radius, 0.0)
        self.assertEqual(s.exit_radius, 0.0)
        self.assertEqual(s.n_par_ct, 1)
        self.assertEqual(s.conj_thresh, 0)
        self.assertEqual(s.min_coll_radius, 0)
        self.assertEqual(len(s.coll_whitelist), 0)
        self.assertEqual(len(s.conj_whitelist), 0)

        with self.assertRaises(ValueError) as cm:
            s.conj_thresh = -1
        self.assertTrue(
            "The conjunction threshold value -1 is invalid: it must be finite and non-negative"
            in str(cm.exception)
        )

        dyn = dynamics.kepler()
        dyn[0] = (dyn[0][0], dyn[0][1] + hy.par[1])

        s = sim(
            ct=0.5,
            state=[[1.0, 0.001, 0.001, 0.001, 1.0, 0.001, 0.001]],
            dyn=dyn,
            pars=[[0.002, 0.001]],
            reentry_radius=[0.1, 0.2, 0.3],
            exit_radius=100.0,
            tol=1e-12,
            high_accuracy=True,
            compact_mode=True,
            n_par_ct=2,
            conj_thresh=0.1,
            min_coll_radius=0.2,
            coll_whitelist={1, 2},
            conj_whitelist={3, 4},
        )

        self.assertTrue(
            np.all(s.state == [[1.0, 0.001, 0.001, 0.001, 1.0, 0.001, 0.001]])
        )
        self.assertTrue(np.all(s.pars == [[0.002, 0.001]]))
        self.assertEqual(s.nparts, 1)
        self.assertEqual(s.time, 0.0)
        self.assertEqual(s.ct, 0.5)
        self.assertTrue(s.high_accuracy)
        self.assertTrue(s.compact_mode)
        self.assertEqual(s.npars, 2)
        self.assertEqual(s.reentry_radius, [0.1, 0.2, 0.3])
        self.assertEqual(s.exit_radius, 100.0)
        self.assertEqual(s.tol, 1e-12)
        self.assertEqual(s.n_par_ct, 2)
        self.assertEqual(s.conj_thresh, 0.1)
        self.assertEqual(s.min_coll_radius, 0.2)
        self.assertEqual(s.coll_whitelist, {1, 2})
        self.assertEqual(s.conj_whitelist, {3, 4})

        s.min_coll_radius = 4
        self.assertEqual(s.min_coll_radius, 4)

        s.min_coll_radius = float("inf")
        self.assertEqual(s.min_coll_radius, float("inf"))

        with self.assertRaises(ValueError) as cm:
            s.min_coll_radius = -1
        self.assertTrue(
            "The minimum collisional radius cannot be NaN or negative, but the invalid value -1 was provided"
            in str(cm.exception)
        )

        s.coll_whitelist = {10, 20}
        self.assertEqual(s.coll_whitelist, {10, 20})

        s.conj_whitelist = {30, 40}
        self.assertEqual(s.conj_whitelist, {30, 40})

        self.assertEqual(s.step(), outcome.success)

        self.assertFalse(
            np.all(s.state == [[1.0, 0.001, 0.001, 0.001, 1.0, 0.001, 0.001]])
        )
        self.assertTrue(np.all(s.pars == [[0.002, 0.001]]))
        self.assertGreater(s.time, 0.0)


        s = sim(
            ct=0.5,
            state=[[1.0, 0.001, 0.001, 0.001, 1.0, 0.001, 0.001]],
            dyn=dyn,
            pars=[[0.002, 0.001]],
            reentry_radius=0.1,
            exit_radius=100.0,
            tol=1e-12,
            high_accuracy=True,
            compact_mode=True
        )

        self.assertEqual(s.reentry_radius, 0.1)

        with self.assertRaises(ValueError) as cm:
            sim(state=[1.0, 2.0, 3.0], ct=1)
        self.assertTrue(
            "The input state must have 2 dimensions, but instead an array with 1 dimension(s) was provided"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            sim(state=[[1.0, 0.001, 0.001, 0.001, 1.0, 0.001]], ct=1)
        self.assertTrue(
            "An input state with 7 columns is expected, but the number of columns is instead 6"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            sim(
                state=[[1.0, 0.001, 0.001, 0.001, 1.0, 0.001, 0.0]],
                ct=1,
                pars=[0.002, 0.001],
                dyn=dyn,
            )
        self.assertTrue(
            "The input array of parameter values must have 2 dimensions, but instead an array with 1 dimension(s) was provided"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            sim(
                state=[[1.0, 0.001, 0.001, 0.001, 1.0, 0.001, 0.0]],
                ct=1,
                pars=[[0.002, 0.001], [0.002, 0.001]],
                dyn=dyn,
            )
        self.assertTrue(
            "An input array of parameter values with 1 row(s) is expected, but the number of rows is instead 2"
            in str(cm.exception)
        )

    def test_set_new_state_pars(self):
        from . import sim, dynamics
        import heyoka as hy
        import numpy as np

        dyn = dynamics.kepler()
        dyn[0] = (dyn[0][0], dyn[0][1] + hy.par[0])

        st = np.zeros((1, 7))
        st = np.vstack([st, np.ones((1, 7))])
        st = np.vstack([st, np.full((1, 7), 2.0)])
        pars = np.array([0.1, 0.2, 0.3]).reshape((3, 1))

        s = sim(st, 0.5, pars=pars, dyn=dyn)

        with self.assertRaises(ValueError) as cm:
            s.set_new_state_pars(new_state=[1.0, 2.0, 3.0])
        self.assertTrue(
            "The input state must have 2 dimensions, but instead an array with 1 dimension(s) was provided"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            s.set_new_state_pars(new_state=np.ones((1, 6)))
        self.assertTrue(
            "An input state with 7 columns is expected, but the number of columns is instead 6"
            in str(cm.exception)
        )

        # Test resetting of params to zero.
        s.set_new_state_pars(new_state=st)
        self.assertTrue(np.all(s.state == st))
        self.assertTrue(np.all(s.pars == 0.0))

        with self.assertRaises(ValueError) as cm:
            s.set_new_state_pars(new_pars=[1, 2, 3], new_state=st)
        self.assertTrue(
            "The input array of parameter values must have 2 dimensions, but instead an array with 1 dimension(s) was provided"
            in str(cm.exception)
        )

        with self.assertRaises(ValueError) as cm:
            s.set_new_state_pars(new_pars=np.ones((6, 6)), new_state=st)
        self.assertTrue(
            "An array of parameter values with 1 column(s) is expected, but the number of columns is instead 6"
            in str(cm.exception)
        )

        s.set_new_state_pars(new_pars=pars[::2], new_state=st[::2])
        self.assertTrue(np.all(s.state == st[::2]))
        self.assertTrue(np.all(s.pars == pars[::2]))

    def test_remove_particles(self):
        from . import sim, dynamics
        import heyoka as hy
        import numpy as np

        dyn = dynamics.kepler()
        dyn[0] = (dyn[0][0], dyn[0][1] + hy.par[0])

        st = np.zeros((1, 7))
        st = np.vstack([st, np.ones((1, 7))])
        st = np.vstack([st, np.full((1, 7), 2.0)])
        pars = np.array([0.1, 0.2, 0.3]).reshape((3, 1))

        s = sim(st, 0.5, pars=pars, dyn=dyn)
        s.remove_particles([2, 2, 2, 0])

        self.assertTrue(np.all(s.state == np.ones((1, 7))))
        self.assertTrue(np.all(s.pars == 0.2))


def run_test_suite():
    retval = 0

    suite = _ut.TestLoader().loadTestsFromTestCase(dynamics_test_case)
    suite.addTests(_ut.TestLoader().loadTestsFromTestCase(sim_test_case))

    test_result = _ut.TextTestRunner(verbosity=2).run(suite)

    if len(test_result.failures) > 0 or len(test_result.errors) > 0:
        retval = 1
    if retval != 0:
        raise RuntimeError("One or more tests failed.")
