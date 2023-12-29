"""
Computations on the detector in pixels and 3D millimeters of source-detector systems such as X-ray.

Author: André Aichert
Date: June 22, 2023
"""

import numpy as np
from numpy.linalg import norm
from .central_projection import ProjectionMatrix
from .utils import cvec, append, dehomogenize

class SourceDetectorGeometry:
    def __init__(self, projection: ProjectionMatrix):
        """From a 3x4 projection matrix and pixel spacing, compute physical location of detector.
        This allows for a full visualization of a source-detector geometry as a pyramid or frustum.
        In case you have a left-handed coordinate frame, use negative pixel spacing to swap the
        viewing direction of the system (i.e. negate pixel spacing if projection looks backwards."""
        projection.normalize()        
        
        C = projection.getCenterOfProjection()
        
        m1 = projection.P[0, :3]
        m2 = projection.P[1, :3]
        m3 = projection.P[2, :3]
        
        U = np.append(np.cross(m3, m2), 0)
        V = np.append(np.cross(m3, m1), 0)

        U *= projection.pixel_spacing / norm(U)
        V *= projection.pixel_spacing / norm(V)
        
        principal_plane = projection.P[2, :4]
        
        # Note how this conversion is symmetric for U and V.
        V_dir = V[:3] / projection.pixel_spacing
        f = np.dot(m1, np.cross(V_dir, m3))
        
        # However, the results will be the same ONLY in the case of rectangular pixels.
        # Assumption of rectangular pixels for a digital detector is pretty safe though.
        image_plane = principal_plane.copy()
        image_plane[3] -= f * projection.pixel_spacing
        
        # negative pixel spacings support flipping detector axes (left handed systems)
        # Note how mulpiplication of a projection matrix with -1 inverts viewing direction.
        # if projection.pixel_spacing < 0:
        #   V *= -1
        
        central_projection = SourceDetectorGeometry.centralProjectionToPlane(C, image_plane)
        
        source_detector_distance = np.dot(image_plane, C)[0]
        
        principal_point_3d = cvec(C) - append(m3, 0) * source_detector_distance
        
        principal_point = np.dot(projection.P, principal_point_3d)
        pp = principal_point / principal_point[2]
        
        # This is the corner of the detector where the pixel origin is located
        detector_origin = principal_point_3d - cvec(U) * pp[0] - cvec(V) * pp[1]

        # things that fully define the source-detector geometry:
        self.source_position = C
        self.detector_origin = detector_origin
        self.axis_direction_Upx = U
        self.axis_direction_Vpx = V
        # plus some useful extras
        self.image_plane = image_plane
        self.principal_point_3d = principal_point_3d
        self.source_detector_distance = source_detector_distance
        # and a projection matrix in 3D world coordinates (3D points -> 3D points on the detector).
        self.central_projection_3d = central_projection

    def __repr__(self):
        return "\n  ".join([
                "SourceDetectorGeometry:",
               f"Source Position: {self.source_position.flatten()}",
               f"Source Detector Distance: {self.source_detector_distance}",
               f"Detector Origin: {self.detector_origin[:,0].tolist()}",
               f"Principal Point 3D: {self.principal_point_3d[:,0].tolist()}",
               f"Axis Orientation:\n    U={self.axis_direction_Upx}\n    V={self.axis_direction_Vpx}"
            ])

    
    @classmethod
    def centralProjectionToPlane(cls, C, E):
        """A mapping T from a 3D point C to a plane E via central projection from C.
        Mapping is according to T*X=meet(join(C,X),E) written in matrix form."""
        C = C.flatten()
        E = E.flatten()
        T = [[+ C[1]*E[1] + C[2]*E[2] + C[3]*E[3] , - C[0]*E[1]                         , - C[0]*E[2]                         , - C[0]*E[3]                         ],  # noqa
             [- C[1]*E[0]                         , + C[0]*E[0] + C[2]*E[2] + C[3]*E[3] , - C[1]*E[2]                         , - C[1]*E[3]                         ],  # noqa
             [- C[2]*E[0]                         , - C[2]*E[1]                         , + C[0]*E[0] + C[3]*E[3] + C[1]*E[1] , - C[2]*E[3]                         ],  # noqa
             [- C[3]*E[0]                         , - C[3]*E[1]                         , - C[3]*E[2]                         , + C[0]*E[0] + C[1]*E[1] + C[2]*E[2] ]]  # noqa
        return T

    def detectorPixelIn3Dmm(self, u, v):
        """ Compute the 3D location of a pixel (u, v) in world coordinates (mm) . See also: ProjectionMatrix.sourceDetectorGeometry()
        This function mostly serves as documentation for how to interpret the source detector geometry."""
        return self.detector_origin + self.axis_direction_Upx * u + self.axis_direction_Vpx * v 

    def projectToDetector3Dmm(self, X):
        return self.central_projection_3d @ X

