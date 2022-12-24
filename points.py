# -*- coding: utf-8 -*-

def dot_product(a, b):
    return sum([a[i] * b[i] for i in range(len(a))])

def vectors_subtraction(a, b):
    return [a[i] - b[i] for i in range(len(a))]    

# https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d
def point_inside_parallelepipede(x, p1, p2, p4, p5):
    u = vectors_subtraction(p1, p2)
    v = vectors_subtraction(p1, p4)
    w = vectors_subtraction(p1, p5)

    if dot_product(u, x) >= dot_product(u, p2) and dot_product(u, x) <= dot_product(u, p1) \
    and dot_product(v, x) >= dot_product(v, p4) and dot_product(v, x) <= dot_product(v, p1) \
    and dot_product(w, x) >= dot_product(w, p5) and dot_product(w, x) <= dot_product(w, p1):
        return True
    else:
        return False