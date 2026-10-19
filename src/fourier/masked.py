from dataclasses import dataclass
from typing import Tuple
import numpy as np
import trimesh


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def _half_terms(size: int) -> Tuple[np.ndarray, np.ndarray]:
    # The (kx, ky) of the centre term followed by every term after it in decompose()'s order (t % size -> kx,
    # t // size -> ky). The terms before the centre are the conjugate partners of these, mirrored through the centre.
    offset = size // 2
    t = np.arange((size ** 2 - 1) // 2, size ** 2)
    return (t % size) - offset, (t // size) - offset


def _half_basis(u: np.ndarray, v: np.ndarray, size: int) -> np.ndarray:
    """
    Real basis for a real-valued series. Heights are real, so c(-k) = conj(c(k)) and each pair sums to
    2 * Re(c(k) * e^(i theta)) = 2a * cos(theta) - 2b * sin(theta), with c(k) = a + ib. Columns are
    [1, 2cos(theta_1) .. 2cos(theta_m), -2sin(theta_1) .. -2sin(theta_m)], matching [c0, a_1 .. a_m, b_1 .. b_m].
    """
    kx, ky = _half_terms(size)
    theta = 2 * np.pi * (np.outer(u, kx[1:]) + np.outer(v, ky[1:]))
    return np.hstack([np.ones((len(u), 1)), 2 * np.cos(theta), -2 * np.sin(theta)])


@dataclass
class MaskedFourier:
    """
    Terrain height h(x, z) = F(u, v) * M(x, z), in world coordinates (Stormworks is Y-up, so X/Z is the ground).

    F is a size x size Fourier series over the square [origin, origin + extent]. M is a convex polygon mask made of
    `len(offsets)` half-planes with evenly spaced normals: it is 1 at least `falloff` inside every edge, 0 outside the
    polygon, and smoothstep in between.

    Only half of the series' coefficients are stored: the other half are their complex conjugates (see _half_basis).
    Stored parameters: coefficients (size^2 reals, as the centre's imaginary part is always 0), origin (2), extent (1),
    offsets (N), falloff (1).
    """

    coefficients: np.ndarray  # ((size^2 + 1) / 2,) complex: the centre term, then the terms after it in decompose() order
    origin: np.ndarray        # (2,) world (x, z) of the series' u = v = 0 corner
    extent: float             # side length of the series' square, in metres
    offsets: np.ndarray       # (N,) half-plane offsets along the normals below
    falloff: float            # width of the mask's soft edge, in metres

    @property
    def size(self) -> int:
        return int(round(np.sqrt(2 * len(self.coefficients) - 1)))

    @property
    def full_coefficients(self) -> np.ndarray:
        """The full (size, size) complex matrix, indexed [ky, kx], with the conjugate half filled back in."""
        return np.concatenate([np.conj(self.coefficients[:0:-1]), self.coefficients]).reshape(self.size, self.size)

    @property
    def float_counts(self) -> dict:
        """How many floats each part of the model needs to encode. The mask's normals are implied by len(offsets)."""
        return {
            'coefficients': 1 + 2 * (len(self.coefficients) - 1),  # the centre term is real
            'origin': 2,
            'extent': 1,
            'offsets': len(self.offsets),
            'falloff': 1,
        }

    @property
    def normals(self) -> np.ndarray:
        angles = 2 * np.pi * np.arange(len(self.offsets)) / len(self.offsets)
        return np.stack([np.cos(angles), np.sin(angles)], axis=1)

    @property
    def polygon(self) -> np.ndarray:
        """Polygon vertices in world (x, z), found by intersecting neighbouring half-plane edges."""
        n = self.normals
        n_next = np.roll(n, -1, axis=0)
        o_next = np.roll(self.offsets, -1)
        det = n[:, 0] * n_next[:, 1] - n[:, 1] * n_next[:, 0]
        x = (self.offsets * n_next[:, 1] - o_next * n[:, 1]) / det
        z = (n[:, 0] * o_next - n_next[:, 0] * self.offsets) / det
        return np.stack([x, z], axis=1)

    def inside_distance(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Distance to the nearest polygon edge: positive inside, negative outside."""
        p = np.stack([np.ravel(x), np.ravel(z)], axis=1)
        return np.min(self.offsets - p @ self.normals.T, axis=1).reshape(np.shape(x))

    def mask(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        return _smoothstep(self.inside_distance(x, z) / self.falloff)

    def series(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        u = (np.ravel(x) - self.origin[0]) / self.extent
        v = (np.ravel(z) - self.origin[1]) / self.extent
        c = self.coefficients
        params = np.concatenate([c[:1].real, c[1:].real, c[1:].imag])
        return (_half_basis(u, v, self.size) @ params).reshape(np.shape(x))

    def evaluate(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        z = np.asarray(z, dtype=float)
        h = np.zeros(x.shape)
        inside = self.inside_distance(x, z) > 0
        # Only evaluate the series where the mask is non-zero
        h[inside] = self.series(x[inside], z[inside]) * self.mask(x[inside], z[inside])
        return h


def fit_masked(
    mesh: trimesh.Trimesh,
    size: int,
    sides: int = 12,
    buffer: float = 20.0,
    falloff: float = 15.0,
    margin: float = 0.1,
    ridge: float = 1e-4,
    samples_per_term: int = 4,
    sea_level: float = 0.0,
) -> MaskedFourier:

    assert size % 2 == 1
    assert falloff <= buffer, 'falloff must fit inside the buffer, or the mask will cut into the terrain'

    v = mesh.vertices
    ground = v[:, [0, 2]]

    # Polygon: N half-planes with evenly spaced normals, each pushed `buffer` past the furthest point above sea level.
    # Every land point then sits at least `buffer` inside every edge, so the mask is exactly 1 on all of it.
    land = ground[v[:, 1] > sea_level]
    angles = 2 * np.pi * np.arange(sides) / sides
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    offsets = (land @ normals.T).max(axis=0) + buffer

    model = MaskedFourier(np.zeros((size ** 2 + 1) // 2, complex), np.zeros(2), 1.0, offsets, falloff)

    # Series square: the polygon's bounding box, made square, plus a margin so the periodic wrap stays outside the mask
    poly = model.polygon
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    extent = (hi - lo).max() * (1 + 2 * margin)
    model.origin = (lo + hi) / 2 - extent / 2
    model.extent = extent

    # Fit points: a regular grid over the polygon, with heights from casting rays straight down onto the mesh (a miss is
    # sea level). Mesh vertices alone leave big gaps under large triangles, and the series oscillates wildly in them.
    spacing = extent / (samples_per_term * size)
    axis = np.arange(spacing / 2, extent, spacing)
    gx, gz = np.meshgrid(model.origin[0] + axis, model.origin[1] + axis)
    gx, gz = gx.ravel(), gz.ravel()
    keep = model.inside_distance(gx, gz) > 0
    gx, gz = gx[keep], gz[keep]

    origins = np.column_stack([gx, np.full(len(gx), mesh.bounds[1, 1] + 10), gz])
    directions = np.tile([0.0, -1.0, 0.0], (len(gx), 1))
    hit_points, ray_index, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=False)

    grid_heights = np.zeros(len(gx))
    grid_heights[ray_index] = np.maximum(sea_level, hit_points[:, 1]) - sea_level
    points = [np.column_stack([gx, gz, grid_heights])]

    # The polygon's edge is pinned to 0 so the series is already near 0 where the mask fades out (otherwise you get a
    # cliff at the edge).
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        count = max(2, int(np.ceil(np.linalg.norm(b - a) / spacing)))
        t = np.linspace(0, 1, count, endpoint=False)[:, None]
        edge = a + t * (b - a)
        points.append(np.column_stack([edge, np.zeros(len(edge))]))

    points = np.vstack(points)
    assert len(points) > size ** 2

    u = (points[:, 0] - model.origin[0]) / extent
    w = (points[:, 1] - model.origin[1]) / extent
    A = _half_basis(u, w, size)
    Z = points[:, 2]

    # Nothing constrains the series outside the polygon, so without a penalty the coefficients grow huge and cancel
    # inside it. The ridge term keeps them small; it is scaled by the point count so it means the same at any density.
    # Each a/b stands for a conjugate pair, so it is weighted 2 to penalise the same sum of |c|^2 as the full matrix.
    if ridge > 0:
        weights = np.full(size ** 2, 2.0)
        weights[0] = 1.0
        A = np.vstack([A, np.sqrt(ridge * len(points)) * np.diag(np.sqrt(weights))])
        Z = np.concatenate([Z, np.zeros(size ** 2)])

    params, *_ = np.linalg.lstsq(A, Z, rcond=None)

    m = (size ** 2 - 1) // 2
    model.coefficients = np.concatenate([params[:1], params[1:m + 1] + 1j * params[m + 1:]])

    print(model.coefficients)

    return model
