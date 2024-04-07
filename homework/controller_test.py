import unittest

from homework.utils import PyTux
from homework.controller import *


class MyTestCase(unittest.TestCase):
    from homework.utils import PyTux
    from argparse import ArgumentParser

    def test_rollout_with_controller(self):
        import numpy as np
        pytux = PyTux()
        steps, how_far = pytux.rollout('zengarden', control, max_frames=1000)
        print(steps, how_far)
        pytux.close()

    def test_angle_between_vectors(self):

        #what is angle if vector is (0, -1)
        aim_point = (0, -1)
        straight_vector = (0, 1)
        theta = angle_between_vectors(straight_vector, aim_point)
        self.assertTrue(theta == 180)


if __name__ == '__main__':
    unittest.main()
