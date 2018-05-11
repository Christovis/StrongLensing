# Run: python -m unittest test_lc_tools.py
import sys
import unittest
sys.path.insert(0, '..')
import lc_tools as LC

class TestLCTools(unittest.TestCase):
    def test_boxlength(self):
        result = LC.boxlength(10, 10)
        self.assertEqual(result, 15)

    def test_position_box_init():
        result = LC.position_box_init(10, 10)
        self.assertEqual(result, 15)

