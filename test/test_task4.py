from task4 import *
import unittest

class TestTask4(unittest.TestCase):
    oTask4 = Task4()
    
    def test_one(self):
        v = self.oTask4.vSuccess
        """Check if it fit the training data well"""
        self.assertTrue(v[0] > 0.95)
        """Check if it fit the cross-validation data well"""
        self.assertTrue(v[1] > 0.95)