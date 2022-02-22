import unittest
import numpy as np
from photonai_graph.Controllability.controllability_functions import modal_control


class ModalControllabilityTests(unittest.TestCase):

    def setUp(self):
        b = np.random.randint(2, size=(20, 20))
        b_symm = (b + b.T) / 2
        b_symm[b_symm < 0.6] = 0
        self.X_sym = b_symm
        self.X_ident = np.eye(20)

    def test_mod_control(self):
        mdr = modal_control(self.X_sym)
        self.assertTrue(np.all(mdr >= 0.75))

    def test_mod_control_shape(self):
        mdr = modal_control(self.X_sym)
        self.assertEqual(mdr.shape, (20,))

    def test_mod_control_mechanism(self):
        mdr = modal_control(self.X_ident)
        self.assertTrue(np.all(mdr == mdr[0]))
