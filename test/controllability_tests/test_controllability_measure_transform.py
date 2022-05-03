import unittest
import os
import numpy as np
import pandas as pd

from photonai_graph.Controllability.controllability_measures import ControllabilityMeasureTransform


class ControllabilityTransformTests(unittest.TestCase):

    def setUp(self):
        b = np.random.randint(2, size=(20, 20))
        b_symm = (b + b.T) / 2
        b_symm[b_symm < 0.6] = 0
        b_symm = b_symm[np.newaxis, :, :, np.newaxis]
        b_symm = np.repeat(b_symm, 10, axis=0)
        self.X_sym = b_symm
        self.y = np.random.rand(20)
        self.rep_X_sym = np.load(os.path.dirname(__file__) + '/X_test.npz')['arr_0']

    def test_mod_control_shape(self):
        contr = ControllabilityMeasureTransform(mod_control=1, ave_control=0)
        contr.fit(self.X_sym, self.y)
        vec = contr.transform(self.X_sym)
        self.assertEqual(np.shape(vec), (10, 20))

    def test_ave_control_shape(self):
        # ensure that mean transform style with a 3d matrix returns the right shape
        contr = ControllabilityMeasureTransform(mod_control=0, ave_control=1)
        contr.fit(self.X_sym, self.y)
        vec = contr.transform(self.X_sym)
        self.assertEqual(np.shape(vec), (10, 20))

    def test_mod_ave_control_shape(self):
        contr = ControllabilityMeasureTransform(mod_control=1, ave_control=1)
        contr.fit(self.X_sym, self.y)
        vec = contr.transform(self.X_sym)
        self.assertEqual(np.shape(vec), (10, 40))

    def test_error_transform(self):
        # ensure that mean transform style with a 3d matrix returns the right shape
        with self.assertRaises(ValueError):
            ControllabilityMeasureTransform(mod_control=0, ave_control=0)

    def test_extract_measures_mod_control(self):
        path = "/tmp/test.csv"
        contr = ControllabilityMeasureTransform(mod_control=1, ave_control=0)
        contr.extract_measures(self.X_sym, path)
        os.remove(path)

    def test_extract_measures_ave_control(self):
        path = "/tmp/test.csv"
        contr = ControllabilityMeasureTransform(mod_control=0, ave_control=1)
        contr.extract_measures(self.X_sym, path)
        os.remove(path)

    def test_extract_measures_mod_ave_control(self):
        path = "/tmp/test.csv"
        contr = ControllabilityMeasureTransform(mod_control=1, ave_control=1)
        contr.extract_measures(self.rep_X_sym, path, ids=[i for i in range(10)], node_list=[f"node_{i}" for i in range(20)])
        df = pd.read_csv(path)
        path_ref = os.path.dirname(__file__) + "/test.csv"
        df_expected = pd.read_csv(path_ref)
        self.assertTrue(np.allclose(df.to_numpy(), df_expected.to_numpy()), "Generated measures are not as expected")
        os.remove(path)

    def test_node_list_error(self):
        path = "/tmp/test.csv"
        node_list = list(range(5))
        contr = ControllabilityMeasureTransform(mod_control=1, ave_control=0)
        with self.assertRaises(ValueError):
            contr.extract_measures(self.X_sym, path, node_list=node_list)

    def test_id_list_error(self):
        path = "/tmp/test.csv"
        id_list = list(range(3))
        contr = ControllabilityMeasureTransform(mod_control=1, ave_control=0)
        with self.assertRaises(ValueError):
            contr.extract_measures(self.X_sym, path, ids=id_list)
