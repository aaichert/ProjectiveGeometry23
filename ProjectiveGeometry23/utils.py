"""
Utility functions for geometry.

Note that these functions are imported directly to projective_geometry.

Example (compute line through point [2, 1] pointing right):
    import projective_geometry as pg
    pg.join(RP2Point(1, 2, 1), RP2Point(0, 1, 0))

Author: André Aichert
Date: June 22, 2023

FIXME (?) numpy is awkward with keeping track of row versus column vectors.
          Possibly simplify code to just use 1D arrays?

"""

import numpy as np
from numpy.linalg import pinv
import shlex


def dot(a,b):
    """Computes dot product for two vectors, no matter the actual shape of the
    numpy array (e.g. 2D row or 2D column or 1D) """
    return np.dot(a.ravel(), b.ravel())


def cvec(vector):
    """Column vector from 1D array or list of values."""
    vector = np.array(vector)
    if vector.ndim == 1:
        vector = vector.reshape(-1, 1)
    return vector


def nullspace(M):
    """ Solve M@X == 0 for |X| == 1.
    Returns solution closest to 0 if M is full rank"""
    _, _, V = np.linalg.svd(M)
    return V[-1, :]


def append(vector, last_coordinate):
    """"Takes a vector as a list of values or an np.array, interprets it as a
    column vector and appends the value last_coordinate to the end of it."""
    vector = cvec(vector)
    return np.vstack([vector, [last_coordinate]])


def homogenize(euclidean):
    """"Takes a vector as a list of values or an np.array, interprets it as a
    column vector and appends a one to the end of it."""
    return append(euclidean, 1.0)


def infinite(direction):
    """"Takes a vector as a list of values or an np.array, interprets it as a
    column vector and appends a zero to the end of it. Normalized to unit."""
    return append(direction, 0.0) / np.linalg.norm(direction)


def dehomogenize(vector):
    """"Divides column vector by last element and returns all but last element.
    """
    vector = cvec(vector)
    return vector[0:-1] / vector[-1]


def hessianNormalForm(vector):
    """Compute the Hessian normal form of a 2D line or 3D plane given as
    homogeneous three-, respectively four-vector."""
    return vector / np.linalg.norm(vector[0:-1])


def RP2Point(x, y, w=1):
    """Functions to improve code readability"""
    return cvec([x,y,w])


def RP2Line(l0, l1, l2):
    """Functions to improve code readability"""
    return cvec([l0, l1, l2])


def RP3Point(x, y, z, w=1):
    """Functions to improve code readability"""
    return cvec([x,y,z,w])


def RP3Plane(p0, p1, p2, p3):
    """Functions to improve code readability"""
    return cvec([p0, p1, p2, p3])


def join2(a,b):
    """Compute joining line from two homogeneous 2D points."""
    return cvec(np.cross(cvec(a)[:,0], cvec(b)[:,0])) 


def meet2(l,m):
    """Compute intersection from homogeneous coordinates of two 2D lines."""
    return cvec(np.cross(cvec(l)[:,0], cvec(m)[:,0]))


def join3(A, B, C):
    """
    Compute the common plane passing through three points or the point of
    intersection of three planes.

    Note: for computing the joining line from two 3D points, please use
          L = pluecker.join_points(A,B)

    Parameters:
        A, B, C: 1D arrays or column vectors representing three points
                 in homogeneous coordinates.
    """
    ABC = np.vstack((A, B, C)) if A.ndim == 1 else np.hstack((A, B, C))
    P = np.array([
        +np.linalg.det(ABC[[1, 2, 3], :]),
        -np.linalg.det(ABC[[0, 2, 3], :]),
        +np.linalg.det(ABC[[0, 1, 3], :]),
        -np.linalg.det(ABC[[0, 1, 2], :]),
    ])
    return cvec(P)


def meet3(P, Q, R):
    """
    Compute the point of intersection for three planes that meet at one point.
    
    Note: for computing the line of intersection from two 3D planes, please use
          L = pluecker.meet_planes(P,Q)

    Parameters:
        P, Q, R: 1D arrays or column vectors representing planes in homogeneous
                 coordinates.
    """
    PQR = np.vstack((P, Q, R)) if P.ndim == 1 else np.hstack((P, Q, R))
    X = np.array([
        +np.linalg.det(PQR[[1, 2, 3], :]),
        -np.linalg.det(PQR[[0, 2, 3], :]),
        +np.linalg.det(PQR[[0, 1, 3], :]),
        -np.linalg.det(PQR[[0, 1, 2], :]),
    ])
    return cvec(X)
    
    
def KRt(K, R, t):
    """Compose projection matrix from intrinsic and extrinsic parameters."""
    return K@np.column_stack((R,t))


def line2d_to_angle_intercept(l):
    """Convert homogeneous coordinates of a 2D line to ange and intercept.
    As is the convention in mathematics, the angle is measured with respect
    to the x-axis. alpha=t=0 corresponds to the line (0,1,0,0)."""
    alpha = np.arctan2(l[1], l[0]) - np.pi * 0.5  # angle
    alpha = alpha + 2.0 * np.pi if alpha < -np.pi else alpha
    t = -l[2] / np.sqrt(l[0] * l[0] + l[1] * l[1])  # intercept
    return np.array([alpha, t])


def line2d_from_angle_intercept(alpha, t):
    """Convert ange and intercept to homogeneous coordinates of a 2D line.
    As is the convention in mathematics, the angle is measured with respect
    to the x-axis. alpha=t=0 corresponds to the line (0,1,0,0)."""
    return RP2Line(
       np.cos(alpha + np.pi * 0.5),
       np.sin(alpha + np.pi * 0.5),
       -t)


def intersectLineWithRect(l, n_x: float, n_y: float ):
    """Intersects line with a rect. Use this for drawing lines.
    Arguments:
        l              a 2D line in homogeneous coordinates.
                       sequence of numbers or np.array.
    Returns:
        x1 y1 x2 y2    line entry (x1,y1) and exit points (x2,y2)."""

    eps = 1e-10
    l = cvec(l)
    # Find intersections with image boundaries
    intersection = [
        meet2(l,cvec([1,0,0]))[:,0],
        meet2(l,cvec([-1,0,n_x-1]))[:,0],
        meet2(l,cvec([0,1,0]))[:,0],
        meet2(l,cvec([0,-1,n_y-1]))[:,0]
    ]

    # Find intersections which are in bounds
    pto, pfrom = [-1,-1,-1], [-1,-1,-1]
    for i in range(4):
        if abs(intersection[i][2])>eps:
            intersection[i] = intersection[i] / intersection[i][2]        
        if  intersection[i][0]<=n_x+eps and intersection[i][1]<=n_y+eps and \
            intersection[i][0]+eps>=0 and intersection[i][1]+eps>=0:
            if pfrom[0]<0:
                pfrom = [intersection[i][0],intersection[i][1],0]
            elif pto[0]<0:
                pto = [intersection[i][0],intersection[i][1],0]
            else:
                # This may happen if a corner coincides with the line.
                # Then, we have to use two intersections, which are far apart to get the line.
                pto2=cvec([intersection[i][0],intersection[i][1],0])
                if np.linalg.norm(np.array(pfrom)-np.array(pto)) < np.linalg.norm(np.array(pfrom)-np.array(pto2)):
                    pto=pto2
 
    return (pfrom[0],pfrom[1],pto[0],pto[1])


def parse_ompl(ompl):
    """Load a text file with one matrix per line (*.ompl).
    Lines that so not start with '#' character (for comments) contain a colon
    seperated list of floating point values, each representing a matrix row.
    Lines starting with '#>' may contain additional meta info, such as
    #> spacing="0.1" detector_size_px="800 600"
    """
    comments = []
    meta = dict()
    matrices = []
    for index, line in enumerate(ompl.split('\n')):
        if line.startswith('#'):
            if line.startswith('#>'):
                # Expected string: key1:"value1" key2:"value2" ...
                kvps = [assignment.split('=') for assignment in shlex.split(line[2:])]
                meta.update({kvp[0]: kvp[1] for kvp in kvps})
            else:
                comments += (index, line[2:])
        else:
            if len(line) < 2:
                continue
            line = line.replace('[', '')
            line = line.replace(']', '')
            matrices += [[[float(value) for value in row.split()] for row in line.split(';')]]
    return matrices, meta, comments

