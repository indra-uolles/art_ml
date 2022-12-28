# -*- coding: utf-8 -*-
import numpy as np

# https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d
def point_inside_parallelepipede(x, p1, p2, p4, p5):
    u = np.subtract(p1, p2)
    v = np.subtract(p1, p4)
    w = np.subtract(p1, p5)

    if np.dot(u, x) >= np.dot(u, p2) and np.dot(u, x) <= np.dot(u, p1) \
    and np.dot(v, x) >= np.dot(v, p4) and np.dot(v, x) <= np.dot(v, p1) \
    and np.dot(w, x) >= np.dot(w, p5) and np.dot(w, x) <= np.dot(w, p1):
        return True
    else:
        return False