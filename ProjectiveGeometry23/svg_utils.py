"""
Interactive drawing of points, lines in 2d and 3D, as well as vidualizing X-ray source-detector geometries.

Usage:
    !pip install svg_vis

    from svg.Jupyter import CanvasWithOverlay
    from pg.homography import rotation_x, rotation_z, scale
    from svg.Renderer import RenderSVG

    target = ProjectionMatrix([...], detector_size_px, spacing)
    
    vis = CanvasWithOverlay(target.image_size[0], target.image_size[1])
    
    # World transformation
    ax, az = 0.5, 0.5
    s = 0.2
    
    def handle_draw(vis):
        global ax
        global az
        x,y = vis.mouse_state.pos()
        if vis.mouse_state.clicked:
            az += vis.mouse_state.dx * 0.01
            ax += vis.mouse_state.dy * 0.01

        svg = RenderSVG((vis.w, vis.h))
    
        svg.add(svg_world_geometry)
        svg.add(svg_source_detector, projection=target,
                draw_on_detector=svg_world_geometry,
                label_source='C0', label_detector='I0(u,v)')
                
        T = scale(s) @ rotation_x(ax) @ rotation_z(az)
        raw_svg_code = svg.render(P=target.P@T)
        vis.html_overlay.value = raw_svg_code

    vis.handle_draw = handle_draw
    
    vis.display()

Author: André Aichert
Date: Dec 11th, 2023
"""


from svg_snip.Composer import Composer
import svg_snip.Elements as e2d
import svg_snip.Elements3D as e3d

import ProjectiveGeometry23.utils as pgu
from ProjectiveGeometry23 import pluecker

from ProjectiveGeometry23.central_projection import ProjectionMatrix
from ProjectiveGeometry23.source_detector_geometry import SourceDetectorGeometry


def svg_coordinate_frame(P, size=100, **kwargs):
    """Draw a coordinate system of default size 100."""
    el = [
        '<g>',
        # Coordinate frame
        e3d.line(P=P, X1=[0,0,0,1], X2=[size,0,0,1], stroke='red', **kwargs),
        e3d.line(P=P, X1=[0,0,0,1], X2=[0,size,0,1], stroke='green', **kwargs),
        e3d.line(P=P, X1=[0,0,0,1], X2=[0,0,size,1], stroke='blue', **kwargs)
    ]
    return '\n  '.join(el) + '\n</g>\n'


def svg_world_geometry(P, **kwargs):
    """Draw a coordinate system of size 100 and a wire cube
    of the same size centered in the origin."""
    el = [
        '<g>',
        # Coordinate frame
        e3d.line(P=P, X1=[0,0,0,1], X2=[100,0,0,1], stroke='red', **kwargs),
        e3d.line(P=P, X1=[0,0,0,1], X2=[0,100,0,1], stroke='green', **kwargs),
        e3d.line(P=P, X1=[0,0,0,1], X2=[0,0,100,1], stroke='blue', **kwargs),
        # a cube with size 100
        e3d.wire_cube(P=P, min=[-50,-50,-50], max=[50,50,50], stroke='black', **kwargs)
    ]
    return '\n  '.join(el) + '\n</g>\n'


def svg_source_detector(P, projection: ProjectionMatrix, draw_on_detector=None, **kwargs):
    """Draw X-ray source-detector geometry. 
    Define draw_on_detector as any SVG drawing function to project
    additional 3D geometry to the detector plane."""
    sdg = SourceDetectorGeometry(projection)
    C = pgu.cvec(sdg.source_position)
    O = sdg.detector_origin
    U = pgu.cvec(sdg.axis_direction_Upx) * projection.image_size[0]
    V = pgu.cvec(sdg.axis_direction_Vpx) * projection.image_size[1]

    el = [
        '<g>\n',
        # Source position
        e3d.point(P=P, X=C, r=1, fill="black", **kwargs),
        # Detector frame
        e3d.polygon(P=P, Xs=[O, O+U, O+V+U ,O+V],
                    fill="#00000020", stroke="#00000040", **kwargs),
        e3d.line(P=P, X1=O, X2=O + U, stroke="magenta", **kwargs),
        e3d.line(P=P, X1=O, X2=O + V, stroke="cyan", **kwargs),
        # Frustum
        e3d.line(P=P, X1=C, X2=O, stroke="#00000020", **kwargs),
        e3d.line(P=P, X1=C, X2=O+V, stroke="#00000020", **kwargs),
        e3d.line(P=P, X1=C, X2=O+U, stroke="#00000020", **kwargs),
        e3d.line(P=P, X1=C, X2=O+V+U, stroke="#00000020", **kwargs)
    ]

    if 'label_source' in kwargs:
        el += [e3d.text(P=P, X=C, content=kwargs['label_source'], **kwargs)]
    if 'label_detector' in kwargs:
        el += [e3d.text(P=P, X=O, content=kwargs['label_detector'], **kwargs)]

    if draw_on_detector is not None:
        T_detector = sdg.central_projection_3d
        detector = draw_on_detector(P=P@T_detector, **kwargs)
    else:
        detector = ""
    
    return '\n<!-->Source Detector Geometry<-->\n' + detector + '\n  '.join(el) + '\n</g>\n'


def svg_homogeneous_line(l, composer: Composer, stroke="yellow", **kwargs):
    """Draw a 2D line given in homogeneous coordinates.
        Note: composer is passed in automatically via svg.Renderer.
    """
    w, h = composer.image_size
    l = pgu.cvec(l)
    x1, y1, x2, y2 = pgu.intersectLineWithRect(l, w, h)
    if not all(isinstance(v, float) for v in [x1, y1, x2, y2]):
        return ""
    return e2d.line(x1=x1, y1=y1, x2=x2, y2=y2, composer=composer, stroke=stroke, **kwargs)


def svg_pluecker_line(P, L, **kwargs):
    """Draw a 3D line given in plucker coordinates."""
    l = pluecker.project(L, P)
    return svg_homogeneous_line(l, **kwargs)
    
