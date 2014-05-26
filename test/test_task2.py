from task2 import *
import unittest

class TestTask2(unittest.TestCase):
    
    def test_one(self):
        oLSI = Task2().oLSIModel
        """Sanity check"""
        self.assertTrue(oLSI.num_topics > 2)
