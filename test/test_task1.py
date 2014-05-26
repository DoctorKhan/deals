from task1 import *
import unittest

class TestTask1(unittest.TestCase):
    """The answers are known, so simply compare them."""
    def test_one(self):
        b = Task1().most()
        self.assertEqual(b, 'and')

    def test_two(self):
        c = Task1().least()
        self.assertEqual(c, 'code:cjcele')

    def test_three(self):
        d = Task1().guitars()
        self.assertEqual(d, 32)
