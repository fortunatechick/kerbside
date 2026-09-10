"""Perspective scaling: how big a thing of known real size looks at a given
place in the image.

This module exists because of traps 3 and 4. Under perspective a single global
minimum-area threshold cannot separate noise from vehicles - set it high enough
to reject speckle in the near field and every distant vehicle disappears; set it
low and near-field noise becomes traffic. The same applies to structuring
elements: one SE size either merges adjacent lanes in the near field or leaves
distant vehicles hollow.

The fix in both cases is to derive the expected size from the homography at the
blob's own road position, which is what the geometry is for.

Syllabus context: Unit 1 (spatial resolution, image as a matrix) supplying the
scale that Unit 4 (morphology) and Unit 6 (segmentation) then consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from kerbside.core.types import BBox, FloatArray, Footprint
from kerbside.geometry.homography import apply_h, apply_h_single, invert_h, normalise_h

__all__ = [
    "REFERENCE_FOOTPRINT",
    "PerspectiveModel",
    "VerticalModel",
    "homography_jacobian",
    "pixels_per_metre",
    "footprint_quad_px",
    "footprint_area_px",
    "estimate_focal_px_from_homography",
    "vertical_model_from_homography",
    "vertical_vanishing_point_px",
]


REFERENCE_FOOTPRINT = Footprint(length_m=4.0, width_m=1.8, height_m=1.5)
"""Reference car used to set the perspective size band. A nominal mid-size
saloon; the tolerance band around it is what admits everything from a scooter to
a bus, and the band is a documented parameter rather than a magic number."""


# --------------------------------------------------------------------------
# Local scale
# --------------------------------------------------------------------------


def homography_jacobian(h: FloatArray, x_m: float, y_m: float) -> FloatArray:
    """Analytic Jacobian ``d(u_px, v_px) / d(x_m, y_m)`` of a homography.

    For ``u = (h00 x + h01 y + h02) / D`` with ``D = h20 x + h21 y + h22``,
    differentiating the quotient gives ``du/dx = (h00 - u h20) / D`` and likewise
    for the other three partials. The Jacobian is the local linear approximation
    of the perspective map, so its column norms are the local pixels-per-metre
    scales and its determinant is the local area scale.
    """
    h = np.asarray(h, dtype=np.float64)
    d = h[2, 0] * x_m + h[2, 1] * y_m + h[2, 2]
    if abs(d) < 1e-12:
        return np.full((2, 2), np.inf, dtype=np.float64)
    u = (h[0, 0] * x_m + h[0, 1] * y_m + h[0, 2]) / d
    v = (h[1, 0] * x_m + h[1, 1] * y_m + h[1, 2]) / d
    return np.array(
        [
            [(h[0, 0] - u * h[2, 0]) / d, (h[0, 1] - u * h[2, 1]) / d],
            [(h[1, 0] - v * h[2, 0]) / d, (h[1, 1] - v * h[2, 1]) / d],
        ],
        dtype=np.float64,
    )


def pixels_per_metre(h: FloatArray, x_m: float, y_m: float) -> tuple[float, float]:
    """Local scale at a road position: ``(across_px_per_m, along_px_per_m)``.

    The two differ substantially on an oblique view - a metre along the road
    foreshortens far more with distance than a metre across it - which is why the
    structuring elements below are anisotropic rather than square.
    """
    j = homography_jacobian(h, x_m, y_m)
    if not np.all(np.isfinite(j)):
        return (0.0, 0.0)
    across = float(np.hypot(j[0, 0], j[1, 0]))
    along = float(np.hypot(j[0, 1], j[1, 1]))
    return (across, along)


def footprint_quad_px(
    h: FloatArray,
    x_m: float,
    y_m: float,
    footprint: Footprint = REFERENCE_FOOTPRINT,
    heading_deg: float = 0.0,
) -> FloatArray:
    """Project a footprint rectangle centred at ``(x_m, y_m)`` into the image.

    Returns the four corners as a ``(4, 2)`` array in pixels. This is the exact
    projected shape of that much road surface - no small-angle approximation - so
    the area it encloses is the right expectation to compare a blob against.
    """
    half_l = footprint.length_m / 2.0
    half_w = footprint.width_m / 2.0
    corners = np.array(
        [[-half_w, -half_l], [half_w, -half_l], [half_w, half_l], [-half_w, half_l]],
        dtype=np.float64,
    )
    if heading_deg:
        a = np.deg2rad(heading_deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float64)
        corners = corners @ rot.T
    corners = corners + np.array([x_m, y_m], dtype=np.float64)
    return apply_h(h, corners)


def polygon_area_px(poly_px: FloatArray) -> float:
    """Shoelace area of a simple polygon, in square pixels."""
    poly = np.asarray(poly_px, dtype=np.float64)
    if poly.shape[0] < 3 or not np.all(np.isfinite(poly)):
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def footprint_area_px(
    h: FloatArray,
    x_m: float,
    y_m: float,
    footprint: Footprint = REFERENCE_FOOTPRINT,
    heading_deg: float = 0.0,
) -> float:
    """Projected area in square pixels of a footprint at a road position."""
    return polygon_area_px(footprint_quad_px(h, x_m, y_m, footprint, heading_deg))


# --------------------------------------------------------------------------
# Recovering the vertical direction from the plane homography
# --------------------------------------------------------------------------


def estimate_focal_px_from_homography(
    h: FloatArray, width_px: int, height_px: int
) -> float | None:
    """Recover the focal length in pixels from a single-plane homography.

    Writing ``H = s K [r1 r2 t]`` and imposing that the rotation columns are
    orthonormal - ``r1 . r2 = 0`` and ``||r1|| = ||r2||`` - gives two independent
    equations for ``f`` once the principal point is fixed at the image centre and
    pixels are assumed square:

        f^2 = -(a1 a2 + b1 b2) / (h20 h21)
        f^2 = ((a1^2 + b1^2) - (a2^2 + b2^2)) / (h21^2 - h20^2)

    with ``a_i = h[0, i] - cx h[2, i]`` and ``b_i = h[1, i] - cy h[2, i]``.
    Either denominator can vanish for particular camera poses, so both are tried
    and the well-conditioned ones averaged. Returns ``None`` when neither is
    usable, which is the honest answer for a near fronto-parallel view where the
    focal length genuinely is not observable from a single plane.
    """
    h = normalise_h(np.asarray(h, dtype=np.float64))
    cx, cy = width_px / 2.0, height_px / 2.0
    a1 = h[0, 0] - cx * h[2, 0]
    b1 = h[1, 0] - cy * h[2, 0]
    a2 = h[0, 1] - cx * h[2, 1]
    b2 = h[1, 1] - cy * h[2, 1]

    estimates: list[float] = []
    denom_orth = h[2, 0] * h[2, 1]
    if abs(denom_orth) > 1e-10:
        f_sq = -(a1 * a2 + b1 * b2) / denom_orth
        if f_sq > 0:
            estimates.append(float(np.sqrt(f_sq)))
    denom_norm = h[2, 1] ** 2 - h[2, 0] ** 2
    if abs(denom_norm) > 1e-10:
        f_sq = ((a1**2 + b1**2) - (a2**2 + b2**2)) / denom_norm
        if f_sq > 0:
            estimates.append(float(np.sqrt(f_sq)))
    if not estimates:
        return None
    return float(np.mean(estimates))


@dataclass(frozen=True, slots=True)
class VerticalModel:
    """Everything needed to project points *off* the road plane from ``H`` alone.

    With ``H = s K [r1 r2 t]``, a world point ``(x, y, z)`` images to

        homogeneous = H (x, y, 1)^T + s z (K r3),    r3 = r1 x r2

    so once ``s`` and ``K r3`` are recovered from the calibration, the top of a
    vehicle of known height can be placed in the image without ever knowing the
    camera pose explicitly. This is what separates a vehicle's *footprint* (the
    road it covers) from its *silhouette* (the blob the pipeline sees), and the
    gap between the two is the ground-point bias of trap 1.
    """

    focal_px: float
    scale_s: float
    k_r3: FloatArray

    @property
    def vanishing_point_px(self) -> tuple[float, float] | None:
        """Image of the world vertical direction."""
        if abs(self.k_r3[2]) < 1e-9:
            return None
        return (float(self.k_r3[0] / self.k_r3[2]), float(self.k_r3[1] / self.k_r3[2]))


def vertical_model_from_homography(
    h: FloatArray, width_px: int, height_px: int, focal_px: float | None = None
) -> VerticalModel | None:
    """Recover the out-of-plane projection terms from a road-plane homography."""
    h = normalise_h(np.asarray(h, dtype=np.float64))
    f = focal_px
    if f is None:
        f = estimate_focal_px_from_homography(h, width_px, height_px)
    if f is None or not np.isfinite(f) or f <= 0:
        return None
    cx, cy = width_px / 2.0, height_px / 2.0
    k = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    k_inv = np.array(
        [[1.0 / f, 0.0, -cx / f], [0.0, 1.0 / f, -cy / f], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    m = k_inv @ h
    n1 = float(np.linalg.norm(m[:, 0]))
    n2 = float(np.linalg.norm(m[:, 1]))
    if n1 < 1e-12 or n2 < 1e-12:
        return None
    # m = mu [r1 r2 t]; the column norms give |mu| but not its sign, and the
    # cross product r1 x r2 is invariant to that sign, so r3 is unambiguous.
    mu_mag = float(np.sqrt(n1 * n2))
    r3 = np.cross(m[:, 0] / n1, m[:, 1] / n2)
    k_r3 = np.asarray(k @ r3, dtype=np.float64)

    # Fix the sign of mu physically rather than algebraically: a camera above
    # the road plane images the top of a vertical *above* its base, so the
    # candidate sign that moves a 1 m vertical upwards in the image is the
    # right one. This is well defined for any roadside mounting.
    probe = _probe_ground_point(h, width_px, height_px)
    if probe is None:
        return None
    base_homo = h @ np.array([probe[0], probe[1], 1.0], dtype=np.float64)
    if abs(base_homo[2]) < 1e-12:
        return None
    base_y = base_homo[1] / base_homo[2]
    for mu in (mu_mag, -mu_mag):
        top = base_homo + mu * 1.0 * k_r3
        if abs(top[2]) < 1e-12:
            continue
        if top[1] / top[2] < base_y:
            return VerticalModel(focal_px=float(f), scale_s=float(mu), k_r3=k_r3)
    return None


def _probe_ground_point(
    h: FloatArray, width_px: int, height_px: int
) -> tuple[float, float] | None:
    """A road-plane point known to be in view, used to disambiguate signs."""
    h_inv = invert_h(h)
    u = width_px / 2.0
    for frac in (0.95, 0.85, 0.75, 0.65):
        wx, wy = apply_h_single(h_inv, u, height_px * frac)
        if np.isfinite(wx) and np.isfinite(wy):
            back = apply_h_single(h, wx, wy)
            if np.isfinite(back[0]) and 0 <= back[1] <= height_px * 1.5:
                return (wx, wy)
    return None


def vertical_vanishing_point_px(
    h: FloatArray, width_px: int, height_px: int, focal_px: float | None = None
) -> tuple[float, float] | None:
    """Image of the world vertical direction, or ``None`` if not recoverable."""
    model = vertical_model_from_homography(h, width_px, height_px, focal_px)
    return None if model is None else model.vanishing_point_px


# --------------------------------------------------------------------------
# Row-indexed model
# --------------------------------------------------------------------------


@dataclass(eq=False, slots=True)
class PerspectiveModel:
    """Row-indexed perspective tables built once from a calibration.

    Every per-blob and per-pixel query in S4 and S6 needs the local scale, and
    recomputing a Jacobian per pixel would dominate the frame budget. On a
    road-plane view the scale varies almost entirely with image row, so the
    tables are built per row once and read back by index.
    """

    h_world_to_img: FloatArray
    width_px: int
    height_px: int
    footprint: Footprint = REFERENCE_FOOTPRINT
    focal_px_hint: float | None = None

    _h_img_to_world: FloatArray = field(init=False, repr=False)
    _horizon_row_px: float = field(init=False, repr=False, default=0.0)
    _table: FloatArray = field(init=False, repr=False)
    _vertical: VerticalModel | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self.h_world_to_img = normalise_h(np.asarray(self.h_world_to_img, dtype=np.float64))
        self._h_img_to_world = invert_h(self.h_world_to_img)
        self._horizon_row_px = self._compute_horizon_row_px()
        self._vertical = vertical_model_from_homography(
            self.h_world_to_img, self.width_px, self.height_px, self.focal_px_hint
        )
        self._table = self._build_row_table()

    # -- basic mapping -----------------------------------------------------

    @property
    def h_img_to_world(self) -> FloatArray:
        return self._h_img_to_world

    @property
    def horizon_row_px(self) -> float:
        """Rows at or above this are not on the road plane in front of the camera."""
        return self._horizon_row_px

    @property
    def vertical(self) -> VerticalModel | None:
        return self._vertical

    def _compute_horizon_row_px(self) -> float:
        # The horizon is where the inverse map's homogeneous weight vanishes:
        # w(u, v) = h[2,0] u + h[2,1] v + h[2,2] = 0, evaluated at the centre
        # column (the line is horizontal for an unrolled camera).
        h_inv = self._h_img_to_world
        u = self.width_px / 2.0
        a, b, c = h_inv[2, 0], h_inv[2, 1], h_inv[2, 2]
        if abs(b) < 1e-12:
            return float("-inf")
        return float(-(a * u + c) / b)

    def image_to_world_m(self, x_px: float, y_px: float) -> tuple[float, float] | None:
        """Road-plane position of an image point, or ``None`` above the horizon.

        Returning ``None`` rather than a huge number matters: a ground point a
        few pixels above the horizon maps to kilometres away, and silently
        accepting that turns an otherwise healthy track into a speed of several
        thousand km/h.
        """
        if y_px <= self._horizon_row_px + 1.0:
            return None
        wx, wy = apply_h_single(self._h_img_to_world, x_px, y_px)
        if not (np.isfinite(wx) and np.isfinite(wy)):
            return None
        return (wx, wy)

    def world_to_image_px(self, x_m: float, y_m: float) -> tuple[float, float]:
        return apply_h_single(self.h_world_to_img, x_m, y_m)

    def image_point_at_height_px(
        self, x_m: float, y_m: float, height_m: float
    ) -> tuple[float, float] | None:
        """Image of the world point ``(x_m, y_m, height_m)``, from ``H`` alone.

        See :class:`VerticalModel` for the derivation. Returns ``None`` when the
        vertical direction is not recoverable from this calibration.
        """
        vm = self._vertical
        if vm is None:
            return None
        homo = self.h_world_to_img @ np.array([x_m, y_m, 1.0], dtype=np.float64)
        homo = homo + vm.scale_s * height_m * vm.k_r3
        if abs(homo[2]) < 1e-12:
            return None
        return (float(homo[0] / homo[2]), float(homo[1] / homo[2]))

    # -- row tables --------------------------------------------------------

    def _build_row_table(self) -> FloatArray:
        """Per-row ``(expected_area_px, across_px_per_m, along_px_per_m)``.

        Computed at the image centre column; across a single row the variation
        is second order compared with the row-to-row change, and these numbers
        set thresholds rather than take measurements.
        """
        table = np.zeros((self.height_px, 3), dtype=np.float64)
        u = self.width_px / 2.0
        for row in range(self.height_px):
            if row <= self._horizon_row_px + 1.0:
                continue
            wx, wy = apply_h_single(self._h_img_to_world, u, float(row))
            if not (np.isfinite(wx) and np.isfinite(wy)):
                continue
            table[row, 0] = footprint_area_px(self.h_world_to_img, wx, wy, self.footprint)
            across, along = pixels_per_metre(self.h_world_to_img, wx, wy)
            table[row, 1] = across
            table[row, 2] = along
        return table

    @property
    def row_table(self) -> FloatArray:
        """``(height_px, 3)`` array of expected area and the two local scales."""
        return self._table

    def _row_index(self, y_px: float) -> int:
        if not np.isfinite(y_px):
            return self.height_px - 1
        return int(np.clip(round(y_px), 0, self.height_px - 1))

    def expected_area_px(self, ground_y_px: float) -> float:
        """Projected area of the reference footprint at that image row."""
        return float(self._table[self._row_index(ground_y_px), 0])

    def scale_px_per_m(self, ground_y_px: float) -> tuple[float, float]:
        row = self._table[self._row_index(ground_y_px)]
        return (float(row[1]), float(row[2]))

    def area_band_px(
        self, ground_y_px: float, lower: float = 0.25, upper: float = 6.0
    ) -> tuple[float, float]:
        """Accept band for blob area at that row, as multiples of the reference.

        The default band spans a scooter (roughly a quarter of a car's footprint)
        to a bus (six times it). Widening it admits shadow-merged pairs;
        narrowing it drops motorcycles. It is a UI parameter precisely because
        that trade-off is the interesting part.
        """
        expected = self.expected_area_px(ground_y_px)
        return (expected * lower, expected * upper)

    def accepts_area(
        self, bbox_px: BBox, area_px: float, lower: float = 0.25, upper: float = 6.0
    ) -> bool:
        """Perspective-aware size filter for one blob (trap 3)."""
        lo, hi = self.area_band_px(bbox_px.ground_point_px.y, lower, upper)
        if hi <= 0.0:
            return False
        return lo <= area_px <= hi

    # -- structuring elements ---------------------------------------------

    def se_size_px(
        self,
        ground_y_px: float,
        size_m: float,
        min_px: int = 3,
        max_px: int = 41,
        vertical: str = "along",
    ) -> tuple[int, int]:
        """Structuring-element size in pixels for a real-world size.

        Returns ``(width_px, height_px)``, both odd and at least ``min_px``.

        ``vertical`` chooses what the element's height is measuring, and getting
        it wrong is a subtle and expensive mistake.

        ``"along"``
            the height follows the along-road scale, which is right when the
            element is being applied to something lying *on the road plane* - a
            footprint, a lane marking, a puddle.
        ``"height"``
            the height follows the across-road scale, which is right when the
            element is being applied to a vehicle's *silhouette*. A silhouette
            extends upwards from its ground point by the vehicle's height, and
            a metre of world height projects to almost exactly the same number
            of pixels as a metre across the road - both are perpendicular to the
            viewing direction - while a metre *along* the road is foreshortened
            severely.

        The difference is not marginal. On the reference camera at image row
        300, the across-road scale is about 25 px/m and the along-road scale
        about 4 px/m. A closing element meant to bridge a half-metre window band
        is 12 px tall under the first reading and 2 px tall under the second, so
        with the wrong choice every vehicle splits into a roof blob and a bumper
        blob with the window band as a gap between them.
        """
        across, along = self.scale_px_per_m(ground_y_px)
        vertical_scale = across if vertical == "height" else along
        w = int(np.clip(int(round(size_m * across)), min_px, max_px))
        h = int(np.clip(int(round(size_m * vertical_scale)), min_px, max_px))
        if w % 2 == 0:
            w += 1
        if h % 2 == 0:
            h += 1
        return (w, h)

    def se_size_table(
        self, size_m: float, min_px: int = 3, max_px: int = 41
    ) -> list[tuple[int, int]]:
        """Per-row SE sizes, precomputed for the banded morphology in S4."""
        return [
            self.se_size_px(float(r), size_m, min_px, max_px) for r in range(self.height_px)
        ]
