# Run: python -m unittest test_lc_tools.py
import sys
import unittest
sys.path.insert(0, '..')
import lc_tools as LC

class TestLCTools(unittest.TestCase):

    def __init__(self):
        hfdir = '/cosma5/data/dp004/dc-beck3/rockstar/full_physics/L62_N512_GR_kpc/'
        snap_dir = '/cosma6/data/dp004/dc-arno1/SZ_project/full_physics/L62_N512_GR_kpc/snapdir_%03d/snap_%03d'
        snapnum = 20
        LengthUnit = 'Mpc' 
        result = LC.__init__(self, hfdir, snap_dir, snapnum, LengthUnit)
        self.assertTrue(result['snapnum'])
        self.assertTrue(result['ID'])
        self.assertTrue(result['pos'])
        self.assertTrue(result['pos_b'])
        self.assertTrue(result['rs_b'])


    def test_subhalo_selection(self):
        result = LC.subhalo_selection(self, dd, [1e12, 1e10], dd, dd, dd)
        self.assertEqual(result[0], 1)


    def test_find_subhalos_in_lc(self):
        lc = None
        subpos = np.asarray([1, 0, 0], [1, 1, 0])
        alpha = 30  #[degrees]
        result = LC.find_subhalos_in_lc(self, lc, subpos, alpha)
        self.assertEqual(result[0], 0)


    def test_find_sub_in_CoDi(self):
        subpos = np.asarray([7, 0, 0], [9, 1, 0])
        codi_1 = 8
        codi_2 = 10
        direction = 0
        result = find_sub_in_CoDi(self, subpos, codi_1, codi_2, direction)
        self.assertEqual(result[1], 1)


    def test_position_box_init(self):
        result = LC.position_box_init(10, 10)
        self.assertEqual(result, 15)


if __name__ == '__main__':
    unittest.main()
