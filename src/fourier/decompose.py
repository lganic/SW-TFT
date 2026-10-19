import math
import numpy as np
import trimesh
from typing import List, Tuple

def decompose(points: List, size: int):

    assert size % 2 == 1
    assert len(points) > (size ** 2)

    size_offset = size // 2

    # I am going to be lazy, and construct the matrix in python, then transfer to numpy. Might be a cleaner way to construct the matrix.

    a_real = []
    a_imag = []
    z = []

    for p in points:

        real_row = []
        real_col = []

        z.append(p[2])

        for t in range(size ** 2):

            x_c = (t % size) - size_offset
            y_c = (t // size) - size_offset

            angle = 2 * math.pi * (x_c * p[0] + y_c * p[1])

            real_row.append(math.cos(angle))
            real_col.append(math.sin(angle))

        a_real.append(real_row.copy())
        a_imag.append(real_col.copy())

    A = np.matrix(a_real) + 1j * np.matrix(a_imag)
    Z = np.array(z)

    c, *_ = np.linalg.lstsq(A, Z, rcond=None)

    return c

def scale_and_decompose(mesh: trimesh.Trimesh, size: int) -> np.ndarray:

    verts = []

    # Stormworks meshes are Y-up: X/Z is the ground plane and Y is the height.
    # Z is negated so (x, -z, y) stays right-handed; using +z mirrors the terrain left/right.
    for x, y, z in mesh.vertices:

        mapped_x = (500 - x) / 1000
        mapped_y = (z + 500) / 1000
        mapped_z = max(0, y / 1000)

        verts.append((mapped_x, mapped_y, mapped_z))

    # The terrain only covers part of the tile. Without data in the rest of the tile the fit is unconstrained
    # there and the coefficients blow up, so pin every empty cell to the lowest height of the terrain.
    cells = 2 * size
    occupied = np.zeros((cells, cells), dtype=bool)

    for x, y, _ in verts:
        occupied[min(int(y * cells), cells - 1), min(int(x * cells), cells - 1)] = True

    base_height = min(p[2] for p in verts)

    for iy, ix in zip(*np.nonzero(~occupied)):
        verts.append(((ix + 0.5) / cells, (iy + 0.5) / cells, base_height))

    return decompose(verts, size)