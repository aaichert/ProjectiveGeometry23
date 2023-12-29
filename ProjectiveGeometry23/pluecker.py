"""
Representation of lines in 3D space in Plücker coordinates.
Author: André Aichert
Date: June 22, 2023
"""

import numpy as np
from .utils import dot, cvec, append
from .utils import homogenize, dehomogenize, RP3Point
from scipy.optimize import minimize


def join_points(A, B):
    """Join two points to form a 3D line."""
    A = A.flatten()
    B = B.flatten()
    return cvec([
            A[0] * B[1] - A[1] * B[0],
            A[0] * B[2] - A[2] * B[0],
            A[0] * B[3] - A[3] * B[0],
            A[1] * B[2] - A[2] * B[1],
            A[1] * B[3] - A[3] * B[1],
            A[2] * B[3] - A[3] * B[2]
        ])


def meet_planes(P, Q):
    """Intersect two planes to form a 3D line."""
    P = P.flatten()
    Q = Q.flatten()
    return cvec([
            P[2] * Q[3] - P[3] * Q[2],
            P[3] * Q[1] - P[1] * Q[3],
            P[1] * Q[2] - P[2] * Q[1],
            P[0] * Q[3] - P[3] * Q[0],
            P[2] * Q[0] - P[0] * Q[2],
            P[0] * Q[1] - P[1] * Q[0]
        ])
    

def join(L, X):
    """Join a line L and a point X to form a plane."""
    L = L.flatten()
    X = X.flatten()
    return cvec([
                          + X[1] * L[5] - X[2] * L[4] + X[3] * L[3],  # noqa
            - X[0] * L[5]               + X[2] * L[2] - X[3] * L[1],  # noqa
            + X[0] * L[4] - X[1] * L[2]               + X[3] * L[0],  # noqa
            - X[0] * L[3] + X[1] * L[1] - X[2] * L[0]                 # noqa
        ])
    

def meet(L, E):
    """Intersect a line L and a plane E to form a point."""
    L = L.flatten()
    E = E.flatten()
    return cvec([
                          - E[1] * L[0] - E[2] * L[1] - E[3] * L[2],  # noqa
            + E[0] * L[0]               - E[2] * L[3] - E[3] * L[4],  # noqa
            + E[0] * L[1] + E[1] * L[3]               - E[3] * L[5],  # noqa
            + E[0] * L[2] + E[1] * L[4] + E[2] * L[5]                 # noqa
        ])
    

def direction(L):
    """Direction of a 3D line in Plücker coordinates.
    Note that for dual Plücker coordinates, the functions direction and moment will be swapped."""
    return np.array([-L[2][0], -L[4][0], -L[5][0]]).reshape(-1, 1)
    

def moment(L):
    """Moment of a 3D line in Plücker coordinates.
    Note that for dual Plücker coordinates, the functions direction and moment will be swapped."""
    return np.array([L[3][0], -L[1][0], L[0][0]]).reshape(-1, 1)
    

def closest_point_to_origin(L):
    """Closest point on line L to the origin."""
    L = L.flatten()
    return np.array([
         L[4] * L[0] + L[1] * L[5],  # noqa
        -L[0] * L[2] + L[3] * L[5],  # noqa
        -L[1] * L[2] - L[3] * L[4],  # noqa
        -L[2] * L[2] - L[4] * L[4] - L[5] * L[5] # noqa
    ]).reshape(-1, 1)


def closest_to_point(L, X):
    """Compute the closest point on the line L to a given point X."""
    d = direction(L)
    plane_through_X_orthogonal_to_L = RP3Point(d[0][0], d[1][0], d[2][0], -dot(d, dehomogenize(X)))
    closest_point_to_X_on_L = meet(L, plane_through_X_orthogonal_to_L)
    return closest_point_to_X_on_L


def distance_to_point(L, X):
    """Compute the distance of a line L to a given point X."""
    I = closest_to_point(L, X)
    return np.linalg.norm(dehomogenize(I) - dehomogenize(X))


def point_closest_to_lines(lines, eps=1e-9):
    """Find point with minimal distance to a line bundle and return (distance, point).
    Returns intersection of two or more lines, if they do intersect."""
    def mean_distance(Xe):
        """Function to be minimized. Takes 3D *euclidian* coordinates."""
        return np.sum([distance_to_point(L, RP3Point(Xe[0], Xe[1] ,Xe[2], 1)) for L in lines]) / len(lines)
    result = minimize(mean_distance, [0, 0, 0], tol=eps)
    return result.fun, homogenize(result.x)


def distance_to_origin(L):
    """Distance of a line to the origin."""
    m = moment(L)
    d = direction(L)
    return np.linalg.norm(m) / np.linalg.norm(d)


def matrixDual(L):
    """Anti-symmetric matrix for the join operation using dual Plücker coordinates."""
    L = L.flatten()
    B = np.array([
        [    0,  L[5], -L[4],  L[3]],  # noqa
        [-L[5],     0,  L[2], -L[1]],  # noqa
        [ L[4], -L[2],     0,  L[0]],  # noqa
        [-L[3],  L[1], -L[0],     0]   # noqa
    ])
    return B


def matrix(L):
    """Anti-symmetric matrix for the meet operation using Plücker coordinates."""
    L = L.flatten()
    B = np.array([
        [   0, -L[0], -L[1], -L[2]],  # noqa
        [L[0],     0, -L[3], -L[4]],  # noqa
        [L[1],  L[3],     0, -L[5]],  # noqa
        [L[2],  L[4],  L[5],     0]   # noqa
    ])
    return B


def projection_matrix(P):
    """Compute Sturm-style projection matrix for central projection of Plücker lines. Projection from Plücker coordinates directly to 2D lines written in a single 3x6 matrix. P is a standard 3x4 projection matrix."""
    PL = np.array([
        [P[1, 0]*P[2, 1]-P[1, 1]*P[2, 0],  P[1, 0]*P[2, 2]-P[1, 2]*P[2, 0],  P[1, 0]*P[2, 3]-P[1, 3]*P[2, 0],  P[1, 1]*P[2, 2]-P[1, 2]*P[2, 1],  P[1, 1]*P[2, 3]-P[1, 3]*P[2, 1],  P[1, 2]*P[2, 3]-P[1, 3]*P[2, 2]],  # noqa
        [P[0, 1]*P[2, 0]-P[0, 0]*P[2, 1], -P[0, 0]*P[2, 2]+P[0, 2]*P[2, 0], -P[0, 0]*P[2, 3]+P[0, 3]*P[2, 0], -P[0, 1]*P[2, 2]+P[0, 2]*P[2, 1], -P[0, 1]*P[2, 3]+P[0, 3]*P[2, 1], -P[0, 2]*P[2, 3]+P[0, 3]*P[2, 2]],  # noqa
        [P[0, 0]*P[1, 1]-P[0, 1]*P[1, 0],  P[0, 0]*P[1, 2]-P[0, 2]*P[1, 0],  P[0, 0]*P[1, 3]-P[0, 3]*P[1, 0],  P[0, 1]*P[1, 2]-P[0, 2]*P[1, 1],  P[0, 1]*P[1, 3]-P[0, 3]*P[1, 1],  P[0, 2]*P[1, 3]-P[0, 3]*P[1, 2]]   # noqa
    ])
    return PL


def project(L, P):
    """Directly project 3D line in Plücker coordinates to 2D line.
    In Python, it may be faster to just do np.dot(projection_matrix(P), L)"""
    line = np.array([
        L[0]*(P[1, 0]*P[2, 1]-P[1, 1]*P[2, 0]) + L[1]*( P[1, 0]*P[2, 2]-P[1, 2]*P[2, 0]) + L[2]*( P[1, 0]*P[2, 3]-P[1, 3]*P[2, 0]) + L[3]*( P[1, 1]*P[2, 2]-P[1, 2]*P[2, 1]) + L[4]*( P[1, 1]*P[2, 3]-P[1, 3]*P[2, 1]) + L[5]*( P[1, 2]*P[2, 3]-P[1, 3]*P[2, 2]),  # noqa
        L[0]*(P[0, 1]*P[2, 0]-P[0, 0]*P[2, 1]) + L[1]*(-P[0, 0]*P[2, 2]+P[0, 2]*P[2, 0]) + L[2]*(-P[0, 0]*P[2, 3]+P[0, 3]*P[2, 0]) + L[3]*(-P[0, 1]*P[2, 2]+P[0, 2]*P[2, 1]) + L[4]*(-P[0, 1]*P[2, 3]+P[0, 3]*P[2, 1]) + L[5]*(-P[0, 2]*P[2, 3]+P[0, 3]*P[2, 2]),  # noqa
        L[0]*(P[0, 0]*P[1, 1]-P[0, 1]*P[1, 0]) + L[1]*( P[0, 0]*P[1, 2]-P[0, 2]*P[1, 0]) + L[2]*( P[0, 0]*P[1, 3]-P[0, 3]*P[1, 0]) + L[3]*( P[0, 1]*P[1, 2]-P[0, 2]*P[1, 1]) + L[4]*( P[0, 1]*P[1, 3]-P[0, 3]*P[1, 1]) + L[5]*( P[0, 2]*P[1, 3]-P[0, 3]*P[1, 2])   # noqa
    ])
    return line
