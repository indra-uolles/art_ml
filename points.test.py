# -*- coding: utf-8 -*-
# python points.test.py

import unittest
import points

class TestPoints(unittest.TestCase):

    def test_point_inside_parallelepipede():
        # p1, p2, p4, p5
        assert points.point_inside_parallelepipede([0.5,0.5,0.5], [0,0,0], [0,0,1], [0,1,0], [0,0,1]) == True, 'should be True'
        assert points.point_inside_parallelepipede([2,2,2], [0,0,0], [0,0,1], [0,1,0], [0,0,1]) == False, 'should be False' 

if __name__ == "__main__":
    TestPoints.test_point_inside_parallelepipede()