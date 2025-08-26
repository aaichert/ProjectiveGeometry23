import unittest
import ProjectiveGeometry23.utils as utils
import ProjectiveGeometry23.homography as homography
import ProjectiveGeometry23.pluecker as pluecker
import ProjectiveGeometry23.central_projection as central_projection

class TestUtils(unittest.TestCase):
    def test_homogenize(self):
        p = [1, 2]
        hp = utils.homogenize(p)
        self.assertEqual(hp[-1], 1)
        self.assertEqual(len(hp), 3)

class TestHomography(unittest.TestCase):
    def test_translation2d(self):
        T = homography.translation2d([1, 2])
        self.assertEqual(T.shape, (3, 3))

class TestPluecker(unittest.TestCase):
    def test_join_points(self):
        import numpy as np
        a = np.array([1, 0, 0, 1])
        b = np.array([0, 1, 0, 1])
        line = pluecker.join_points(a, b)
        self.assertEqual(len(line), 6)

class TestCentralProjection(unittest.TestCase):
    def test_projection_matrix(self):
        import numpy as np
        P = central_projection.ProjectionMatrix(np.eye(3,4))
        self.assertEqual(P.P.shape, (3, 4))

if __name__ == '__main__':
    unittest.main()
