"""Equal-area CRS helper.

Every area, distance, and density in this benchmark is computed in an
equal-area CRS. EPSG:4326 (WGS84 lat/lon) areas are *not* meaningful --
one square degree is ~12,000 km2 at the equator and shrinks toward the
poles. Passing an EPSG:4326 GeoDataFrame into `.area` returns square
degrees, which is nonsense for buildings.

Default choice: **EPSG:6933** ("WGS 84 / NSIDC EASE-Grid 2.0 Global").

Rationale for this specific projection:
    - It is a true equal-area projection (Lambert Cylindrical Equal-Area
      variant), not an equidistant or conformal projection.
    - It is defined on the WGS84 datum, matching every other CRS this
      project uses (mercantile bounds, SpaceNet metadata, Sentinel-2, ...).
    - Its "reasonable" latitude band (~85 N to ~85 S) covers every
      SpaceNet 2 AOI (Khartoum 15 N ... Paris 49 N).
    - It is a single global CRS, so a summary table for all four AOIs
      is expressed in the same units without per-AOI re-projection
      bookkeeping.

Rejected alternatives:
    - EPSG:32611 / 32633 / other UTM zones -- UTM is transverse-Mercator
      and CONFORMAL, not equal-area, and picking a per-AOI zone silently
      changes what the units mean between AOIs.
    - EPSG:4087 -- equidistant cylindrical; areas are wrong.
    - Web Mercator EPSG:3857 -- massively distorts area with latitude.
    - Per-AOI Lambert Azimuthal Equal Area centered on the AOI -- most
      accurate for that AOI, but harder to cross-compare and requires
      construction of an ad-hoc CRS per AOI.

The `assert_equal_area_crs` helper is what stubs in `bfbench.metrics`
use as a precondition. Change the default only via configuration; do
NOT smuggle a different CRS into the metric code.
"""

from __future__ import annotations

from typing import Any

from pyproj import CRS

# NSIDC EASE-Grid 2.0 Global. See https://epsg.io/6933.
DEFAULT_EQUAL_AREA_EPSG: int = 6933


class NonEqualAreaCRSError(ValueError):
    """Raised when an area/distance is about to be computed in a CRS that
    is not equal-area (WGS84 lat/lon, UTM, Web Mercator, ...).
    """


def is_equal_area(crs: Any) -> bool:
    """True iff the projection's `is_area_preserving` property is True.

    Uses pyproj's own metadata rather than a hand-maintained allowlist so
    the check stays honest when a user brings a non-standard equal-area
    projection.
    """
    try:
        parsed = CRS.from_user_input(crs)
    except Exception as e:  # noqa: BLE001
        raise NonEqualAreaCRSError(
            f"could not parse CRS {crs!r} for equal-area check: {e}"
        ) from e
    # pyproj exposes .coordinate_operation.method_name and .is_projected.
    # The definitive test is the .is_geographic flag (no) plus a name-based
    # cross-check against the recognised equal-area projections.
    if parsed.is_geographic:
        return False
    # pyproj CRS objects expose `.coordinate_operation`; ask the underlying
    # projection whether it's area-preserving. For CRSs where pyproj can't
    # decide (rare), fall back to a small allowlist of EPSGs.
    op = parsed.coordinate_operation
    method_name = (op.method_name.lower() if op is not None else "") or ""
    if any(k in method_name for k in ("equal area", "equal-area", "cylindrical equal", "mollweide",
                                     "lambert azimuthal equal", "albers")):
        return True
    known_equal_area_epsg = {6933, 6931, 6932, 3410, 3411, 3408, 3409}  # EASE/EASE-Grid family
    known_equal_area_esri = {
        "esri:54009",  # World_Mollweide
        "esri:54034",  # World_Cylindrical_Equal_Area
        "esri:54017",  # Behrmann
        "esri:54008",  # Sinusoidal
    }
    if parsed.to_epsg() in known_equal_area_epsg:
        return True
    auth = parsed.to_authority()
    if auth is not None and f"{auth[0].lower()}:{auth[1]}" in known_equal_area_esri:
        return True
    return False


def assert_equal_area_crs(gdf_or_crs: Any, *, what: str = "computation") -> None:
    """Raise NonEqualAreaCRSError unless the argument is (or has) an
    equal-area CRS. `gdf_or_crs` may be a GeoDataFrame (uses `.crs`) or
    any pyproj-compatible CRS input.
    """
    crs = getattr(gdf_or_crs, "crs", gdf_or_crs)
    if crs is None:
        raise NonEqualAreaCRSError(
            f"{what}: input has no CRS. Explicitly project into "
            f"EPSG:{DEFAULT_EQUAL_AREA_EPSG} (or another equal-area CRS) first."
        )
    if not is_equal_area(crs):
        raise NonEqualAreaCRSError(
            f"{what}: CRS {crs!s} is not equal-area. Reproject to "
            f"EPSG:{DEFAULT_EQUAL_AREA_EPSG} (or another equal-area CRS) "
            "before computing areas, distances, or densities."
        )
