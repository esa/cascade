# Copyright 2023 Francesco Biscani (bluescarni@gmail.com), Dario Izzo (dario.izzo@gmail.com)
#
# This file is part of the cascade.py library.
#
# This Source Code Form is subject to the terms of the Mozilla
# Public License v. 2.0. If a copy of the MPL was not distributed
# with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import unittest as _ut


class sim_test_case(_ut.TestCase):
    def runTest(self):
        self.test_basic()
        self.test_remove_particles()
        self.test_set_new_state_pars()
        self.test_ct_api()

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
        self.assertEqual(s.npars, 0)
        self.assertEqual(s.c_radius, 0.0)
        self.assertEqual(s.d_radius, 0.0)
        self.assertEqual(s.n_par_ct, 1)

        dyn = dynamics.kepler()
        dyn[0] = (dyn[0][0], dyn[0][1] + hy.par[1])

        s = sim(
            ct=0.5,
            state=[[1.0, 0.001, 0.001, 0.001, 1.0, 0.001, 0.001]],
            dyn=dyn,
            pars=[[0.002, 0.001]],
            c_radius=[0.1, 0.2, 0.3],
            d_radius=100.0,
            tol=1e-12,
            high_accuracy=True,
            n_par_ct=2,
        )

        self.assertTrue(
            np.all(s.state == [[1.0, 0.001, 0.001, 0.001, 1.0, 0.001, 0.001]])
        )
        self.assertTrue(np.all(s.pars == [[0.002, 0.001]]))
        self.assertEqual(s.nparts, 1)
        self.assertEqual(s.time, 0.0)
        self.assertEqual(s.ct, 0.5)
        self.assertTrue(s.high_accuracy)
        self.assertEqual(s.npars, 2)
        self.assertEqual(s.c_radius, [0.1, 0.2, 0.3])
        self.assertEqual(s.d_radius, 100.0)
        self.assertEqual(s.tol, 1e-12)
        self.assertEqual(s.n_par_ct, 2)

        self.assertEqual(s.step(), outcome.success)

        self.assertFalse(
            np.all(s.state == [[1.0, 0.001, 0.001, 0.001, 1.0, 0.001, 0.001]])
        )
        self.assertTrue(np.all(s.pars == [[0.002, 0.001]]))
        self.assertEqual(s.nparts, 1)
        self.assertGreater(s.time, 0.0)
        self.assertEqual(s.ct, 0.5)
        self.assertTrue(s.high_accuracy)
        self.assertEqual(s.npars, 2)
        self.assertEqual(s.c_radius, [0.1, 0.2, 0.3])
        self.assertEqual(s.d_radius, 100.0)
        self.assertEqual(s.tol, 1e-12)

        s = sim(
            ct=0.5,
            state=[[1.0, 0.001, 0.001, 0.001, 1.0, 0.001, 0.001]],
            dyn=dyn,
            pars=[[0.002, 0.001]],
            c_radius=0.1,
            d_radius=100.0,
            tol=1e-12,
            high_accuracy=True,
        )

        self.assertEqual(s.c_radius, 0.1)

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

    suite = _ut.TestLoader().loadTestsFromTestCase(sim_test_case)

    test_result = _ut.TextTestRunner(verbosity=2).run(suite)

    if len(test_result.failures) > 0 or len(test_result.errors) > 0:
        retval = 1
    if retval != 0:
        raise RuntimeError("One or more tests failed.")
