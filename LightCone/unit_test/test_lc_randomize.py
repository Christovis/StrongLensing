import sys
import unittest
sys.path.insert(0, '..')
import lc_randomize as RND


class TestLCRandomize(unittest.TestCase):

    def test_translation_s(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def rotation_s(self):
        self.assertEqual('foo'.upper(), 'FOO')

if __name__ == '__main__':
    unittest.main()
