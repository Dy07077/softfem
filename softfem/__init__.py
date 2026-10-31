"""softfem: a compact implicit FEM simulator for hyperelastic soft bodies."""
from .contact import Plane, Sphere, barrier
from .fem import Elasticity, element_to_vertex, lumped_mass, nodal_areas
from .materials import LinearElastic, StableNeoHookean, lame
from .mesh import TetMesh, ball, ball_octant, box, ellipsoid, merge
from .solver import Simulator, StepInfo

__version__ = "0.1.0"

__all__ = [
    "TetMesh",
    "box",
    "ball",
    "ball_octant",
    "ellipsoid",
    "merge",
    "LinearElastic",
    "StableNeoHookean",
    "lame",
    "Elasticity",
    "lumped_mass",
    "nodal_areas",
    "element_to_vertex",
    "Plane",
    "Sphere",
    "barrier",
    "Simulator",
    "StepInfo",
]
