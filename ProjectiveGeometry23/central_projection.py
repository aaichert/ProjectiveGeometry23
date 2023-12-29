"""
Linear projection from real projective three-space to real projective two-space.
Representations of a pinhole camera as Projection matrix optionally with additional information of image size and pixelation.

Author: André Aichert
Date: June 22, 2023
"""

import numpy as np
import scipy


# recommended:
# from rich import print
# np.set_printoptions(suppress=True)

class ProjectionMatrix:
    def __init__(self, P, image_size=(400, 300), pixel_spacing=1.0):
        """Pinhole projection model as 3x4 projection matrix with optional
        information of detector size and pixel spacing [mm per px]."""
        self.P = np.array(P).reshape((3, 4))
        self.image_size = np.array(image_size)
        self.pixel_spacing = pixel_spacing


    @classmethod
    def perspective_look_at(cls, eye, center=np.array([0, 0, 0]), image_size=(400, 300), fovy_rad=0.5, pixel_spacing=1.0):
        """Create a ProjectionMatrix that looks at a center point from and eye point given field-of-view and image size.

        Extrinsic parameters as rotation matix R and translation vector t:
        - eye: Position [X,Y,Z] of the camera (camera center)
        - center: Point the camera is looking at (target position)
        
        Intrinsic parameters of a pinhole camera as a 3x3 matrix K.
        - fovy_rad: Field of view angle in radians (vertical)
        - image_size: Width and height of the image sensor

        Returns ProjectionMatrix P=K[R t] with given image_size
        
        TODO: optional pixel_spacing should probably affect intrinsics
        """
        # Up vector defining the camera's orientation
        up=np.array([0, 1, 0])
        
        # Intrinsics
        w, h = image_size
        tanfov2 = 2 * np.tan(0.5 * fovy_rad)
        fx, fy = h / tanfov2, h / tanfov2
        cx, cy = 0.5 * w, 0.5 * h

        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]])

        # Extrinsics
        fwd = (center - eye) / np.linalg.norm(center - eye)
        left = np.cross(up, fwd)
        left /= np.linalg.norm(left)
        new_up = np.cross(fwd, left)

        R = np.array([left, new_up, -fwd])
        t = -R.dot(eye)
        
        # Compose projection matrix
        P = np.zeros((3, 4))
        P[:, :3] = K @ R
        P[:, 3] = K @ t
        
        return cls(P, image_size, pixel_spacing)

    def to_dict(self):
        return {
            "P": self.P.tolist(),
            "image_size": self.image_size.to_dict(),
            "pixel_spacing": self.pixel_spacing
        }

    def to_ompl(self, with_geometry=True):
        def f2str(f):
            return np.format_float_positional(f, trim='-')
        s = "; ".join([" ".join([f2str(x) for x in row]) for row in self.P])
        if with_geometry:
            return f'#> spacing="{self.pixel_spacing}" detector_size_px="{self.image_size}"\n[{s}]'
        else:
            return f'[{s}]'
        
    @classmethod
    def from_dict(cls, data):
        return cls(
            data["P"],
            np.array(data["image_size"]),
            data["pixel_spacing"]
        )

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __repr__(self):
        return f"ProjectionMatrix:\n{self.P}\nImage Size: {self.image_size}\nPixel Spacing: {self.pixel_spacing}"

    def principal_ray(self):
        return self.P[2][0:3]

    def normalize(self):
        norm_m3 = np.linalg.norm(self.principal_ray())
        detM = np.linalg.det(self.P[0:3, 0:3])
        if detM < 0:
            norm_m3 *= -1;
        self.P /= norm_m3

    def getCenterOfProjection(self):
        _, _, V = np.linalg.svd(self.P)
        C = V[-1, :4]
        if C[3] < -1e-12 or C[3] > 1e-12:
            C = C / C[3]  # By definition: Camera centers are always positive points.
        return C.reshape(-1, 1)

    def getPrincipalPoint(self):
        pp = self.P[:3, :3] @ self.P[2, :3].T
        return pp / pp[2]

    def getPrincipalRay(self):
        m3 = self.P[2, :3]
        if np.linalg.norm(m3) > 1e-12:
            m3 = m3 / np.linalg.norm(m3)
        return m3.reshape(-1, 1)

    def getFocalLengthPx(self):
        """Compute the focal length in pixels (diagonal entries of K in P=K[R t] )."""
        self.normalize()
        
        m1 = self.P[0, :3]
        m2 = self.P[1, :3]
        m3 = self.P[2, :3]

        U = np.cross(m3, m2) / np.linalg.norm(np.cross(m3, m2))
        V = np.cross(m3, m1) / np.linalg.norm(np.cross(m3, m1))

        focal_length_u_px = np.dot(m1, np.cross(V, m3))
        # focal_length_v_px = np.dot(m2, np.cross(U, m3))  # should be identical

        return focal_length_u_px

    def getDetectorAxisDirections(self):
        """Compute the two three-points where the image u- and v-axes meet infinity. Scaled to world coordinates."""
        self.normalize()
        
        m1 = self.P[0, :3]
        m2 = self.P[1, :3]
        m3 = self.P[2, :3]

        U = np.append(np.cross(m3, m2) / np.linalg.norm(np.cross(m3, m2)), 0)
        V = np.append(np.cross(m3, m1) / np.linalg.norm(np.cross(m3, m1)), 0)

        return U, V

    def getDetectorAxisDirectionsPx(self):
        """Compute the two three-points where the image u- and v-axes
        meet infinity. Scaled to pixels."""
        U, V = getDetectorAxisDirections()
        return U * self.pixel_spacing, V * self.pixel_spacing

    def getDetectorPlane(self, imageVPointsUp=False):
        """ Decomposes the projection matrix to compute the equation of the
        image/detector plane. Assumes rectangular pixels. This is identical to
        the principal place shifted by the focal length. For left-handed
        coordinate systems, set imageVPointsUp to True.
        Returns the image detector plane in Hessian normal form and a boolean
        indicating whether imageVPointsUp appears to be set correctly. """
        # TODO: alternative (that won't know when handedness is off)
        # focal_length_u_mm = self.getFocalLength() * self.pixel_spacing
        K, R, t, appears_flipped = self.decomposition(imageVPointsUp)
        focal_length_u_mm = K[0, 0] * self.pixel_spacing
        principal_plane = self.P[2, :4].copy()
        principal_plane /= np.linalg.norm(principal_plane[:3])
        principal_plane[3] -= focal_length_u_mm
        return principal_plane, appears_flipped

    def decomposition(self, imageVPointsUp = False):
        """Decompose Projection Matrix into K[R|t] using RQ-decomposition.
        Returns K, R, t and a boolean indicating if R is left-handed.
        For right-handed world coordinates implies imageVPointsUp is wrong."""
        # Compute RQ decomposition of leftmost 3x3 sub-matrix of P
        R, Q = scipy.linalg.rq(self.P[:3, :3].copy())
        K = R
        R = Q
        # make diagonal of K positive
        S = np.eye(3)
        if K[0, 0] < 0:
            S[0, 0] = -1
        if K[1, 1] < 0:
            S[1, 1] = -1
        if imageVPointsUp:
            S[1, 1] *= -1
        if K[2, 2] < 0:
            S[2, 2] = -1
        K = K @ S
        R = S @ R
        # Force zero elements in K
        K[1, 0] = K[2, 0] = K[2, 1] = 0
        # Scale
        K *= 1.0 / K[2, 2]
        t = np.linalg.solve(K, self.P[:3, 3])
        # EXPLANATION of appears_flipped:
        # In oriented projective geometry, for a visible point x=P*X the
        # homogeneous coordinate will be positive. A negative homogeneous
        # coordinate implies "behind the camera". You can thus change the
        # direction in which a camera is facing by a multiplication of the
        # projection matrix with -1.
        # This is relevant for the decomposition, because in practice, the
        # coordinate system can be left-handed. In this case, the image
        # "rotation" matrix has negative determinant. This may actually be the
        # case, e.g. for the way pixels are stored in a BMP file, some window
        # coordinates or Siemens 'Leonardo' style raw images. In most cases,
        # though, that means you got the oriantation wrong.
        # You can check using sourceDetectorGeometry. Only if your pixel
        # spacing and imageVPointsUp values are correct, will the image plane
        # and detector origin be on the opposite side of the object w.r.t.
        # source position (as it should be... duh).
        appears_flipped = np.linalg.det(R) > 0
        return K, R, t, appears_flipped

    def pseudoinverse(self):
        return np.linalg.pinv(self.P)

    def backproject(self, x):
        """Compute Plücker Coordinates of a backprojection for a 2D image point x in homogeneous coordinates.
        if x is a list, make sure to convert o a column vector np.array(x).reshape(-1, 1)."""
        Pinv = self.pseudoinverse()
        return Pinv @ x

    def computeFundamentalMatrix(self, P1):
        """Compute fundamental matrix from two projection matrices. Pseudoinverse-based implementation"""
        if isinstance(P1, ProjectionMatrix):
            P1 = P1.P
        C0 = self.getCenterOfProjection()
        e1 = P1 @ C0
        e1 = e1.flatten() / np.linalg.norm(e1)
        e1x = np.array([[     0, +e1[2], -e1[1]],
                        [-e1[2],      0, +e1[0]],
                        [+e1[1], -e1[0],      0]])
        P0plus = self.pseudoinverse()

        return  np.dot(np.dot(e1x, P1), P0plus)
