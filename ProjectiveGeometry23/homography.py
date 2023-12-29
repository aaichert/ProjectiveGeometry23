"""
Useful homographies on real projective two- and three-space.
Author: André Aichert
Date: June 22, 2023
"""

import numpy as np


def translation2d(t):
    """Homogeneous 2D translation."""
    T = np.array([
        [1, 0, t[0]],
        [0, 1, t[1]],
        [0, 0,    1]
    ])
    return T


def scale2d(s):
    """Homogeneous 2D scaling."""
    if isinstance(s, (int, float)):
        s = [s,s]        
    S = np.array([
        [s[0], 0,    0],
        [0,    s[1], 0],
        [0,    0,    1]
    ])
    return S


def rotation2d(alpha):
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    R = np.array([
        [ca, -sa, 0],
        [sa,  ca, 0],
        [ 0,   0, 1]
    ])
    return R


def rigid_transform(alpha, tu, tv):
    """Homogeneous 2D rigid transformation."""
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    R = np.array([
        [ca, -sa, tu],
        [sa,  ca, tv],
        [ 0,   0,  1]
    ])
    return R
    
    
def translation(t):
    """Homogeneous 3D translation."""
    T = np.array([
        [1, 0, 0, t[0]],
        [0, 1, 0, t[1]],
        [0, 0, 1, t[2]],
        [0, 0, 0, 1]
    ])
    return T
    

def scale(s):
    """Homogeneous 3D scaling."""
    if isinstance(s, (int, float)):
        s = [s,s,s]
    S = np.array([
        [s[0], 0,    0,    0],
        [0,    s[1], 0,    0],
        [0,    0,    s[2], 0],
        [0,    0,    0,    1]
    ])
    return S
    
    
def rotation_x(alpha):
    """Homogeneous rotation about X-axis."""
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    R = np.array([
        [1,  0,   0, 0],
        [0, ca, -sa, 0],
        [0, sa,  ca, 0],
        [0,  0,   0, 1]
    ])
    return R


def rotation_y(alpha):
    """Homogeneous rotation about Y-axis."""
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    R = np.array([
        [ca,  0, sa, 0],
        [ 0,  1,  0, 0],
        [-sa, 0, ca, 0],
        [ 0,  0,  0, 1]
    ])
    return R


def rotation_z(alpha):
    """Homogeneous rotation about Z-axis."""
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    R = np.array([
        [ca, -sa, 0, 0],
        [sa,  ca, 0, 0],
        [ 0,   0, 1, 0],
        [ 0,   0, 0, 1]
    ])
    return R


def euler_to_rotation_matrix(euler_angles):
    """Rotation from Euler angles. Assumes input is a 3-vector containing the Euler angles."""
    rotation_x_matrix = rotation_x(euler_angles[0])
    rotation_y_matrix = rotation_y(euler_angles[1])
    rotation_z_matrix = rotation_z(euler_angles[2])

    R = rotation_x_matrix @ rotation_y_matrix @ rotation_z_matrix
    return R


def euler_from_rotation_matrix(R):
    """Euler angles from rotation matrix.
    Assumes the top-left 3x3 submatrix of R is in fact an orthogonal matrix."""
    ry = np.arcsin(R[0, 2])
    rz = np.arccos(R[0, 0] / np.cos(ry))
    rx = np.arccos(R[2, 2] / np.cos(ry))
    euler_angles = np.array([rx, ry, rz])
    return euler_angles

