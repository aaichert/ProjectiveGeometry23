"""
Direct Linear Transformation (DLT)
Author: André Aichert
Date: June 22, 2023
"""


import numpy as np

def dlt_normalization(p2d, p3d, matches):
    """Normalization of 2D and 3D point clouds. Input assumed to have homogeneous coordinate 1.
    Purpose: De-mean and scale to +/- sqtr(2), sqrt(3) respectively. """
    # Compute mean
    mean2d = np.array([0, 0])
    mean3d = np.array([0, 0, 0])
    for i, j in matches:
        mean2d += p2d[i][:2]
        mean3d += p3d[j][:3]
    mean2d /= len(match)
    mean3d /= len(match)

    # Compute size
    s2d = 0
    s3d = 0
    for i, j in matches:
        s2d += np.linalg.norm(p2d[i][:2] - mean2d)
        s3d += np.linalg.norm(p3d[j][:3] - mean3d)
    s2d *= np.sqrt(2) / len(match)
    s3d *= np.sqrt(3) / len(match)

    # Compose normalization matrices
    normalization_2d = np.matmul(scale(s2d, s2d), translation(-mean2d[0], -mean2d[1]))
    normalization_3d = np.matmul(scale(s3d, s3d, s3d), translation(-mean3d[0], -mean3d[1], -mean3d[2]))

    return normalization_2d, normalization_3d


def dlt(x, X, match=None):
    """Direct Linear Transformation (DLT) for projection matrices."""
    # Optional match parameter
    if not match and len(X) == len(x):
        match = {i: i for i in range(len(X))}

    # Check for insufficient data
    if len(match) < 6:
        return np.zeros((3, 4))

    # Normalization of input data
    N_px, N_mm = dlt_normalization(x, X, match)

    # Build homogeneous system matrix from point matches
    A = np.zeros((2 * len(match), 12))
    for k, (key, value) in enumerate(match.items()):
        # Normalize input points
        x_norm = np.dot(N_px, x[key])
        X_norm = np.dot(N_mm, X[value])
        # Write two rows in A (we get two independent equations from one point match)
        A[2 * k, 4:8] = x_norm[2] * X_norm
        A[2 * k + 1, 0:4] = -x_norm[2] * X_norm
        A[2 * k, 8:12] = -x_norm[1] * X_norm
        A[2 * k + 1, 8:12] = x_norm[0] * X_norm

    # Solve and reshape
    _, _, V = np.linalg.svd(A)
    p = V[-1] / V[-1, -1]
    P_norm = p.reshape((3, 4))

    # Denormalize
    P = np.dot(np.dot(np.linalg.inv(N_px), P_norm), N_mm)
    P /= np.linalg.norm(P[:, :3], axis=0)
    return P
