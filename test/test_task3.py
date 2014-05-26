from task3 import *
import unittest

class TestTask3(unittest.TestCase):
    oTask3 = Task3()
    
    def test_one(self):
        v = self.oTask3.vbPredTrain[0:20]
        v[0:20] -= 1
        f = abs(v).mean()
        """Make sure it is at least fitting the training data"""
        self.assertTrue(f < 0.05)