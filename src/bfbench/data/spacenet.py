"""SpaceNet 2 (SN2_buildings) fetch from the public AWS Open Data bucket.

Bucket layout, verified 2026-09-10 via anonymous list_objects_v2:

    s3://spacenet-dataset/spacenet/SN2_buildings/
        train/
            AOI_2_Vegas/{MS,PAN,PS-MS,PS-RGB,geojson_buildings}/...
            AOI_3_Paris/...
            AOI_4_Shanghai/...
            AOI_5_Khartoum/...
            SN2_buildings_train_AOI_<n>_<City>_solutions.csv
        test_public/...
        tarballs/       # multi-GB archives; NOT used by this fetcher

Access model: anonymous (unsigned) requests. No AWS credentials required.
If we ever hit a 403/AccessDenied we raise a clear error so the caller
knows to escalate rather than substituting a different dataset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from botocore.exceptions import ClientError, NoCredentialsError

_LOG = logging.getLogger(__name__)

SPACENET_BUCKET = "spacenet-dataset"
SPACENET_SN2_PREFIX = "spacenet/SN2_buildings"

_AOI_CITIES: dict[int, str] = {
    2: "Vegas",
    3: "Paris",
    4: "Shanghai",
    5: "Khartoum",
}

_VALID_SPLITS: frozenset[str] = frozenset({"train", "test_public"})
_VALID_IMAGE_KINDS: frozenset[str] = frozenset({"PS-RGB", "PS-MS", "MS", "PAN"})


class SpaceNetAccessError(RuntimeError):
    """Raised when the public SpaceNet bucket cannot be read anonymously."""


@dataclass(frozen=True)
class SpaceNetAOI:
    """Identifies one SpaceNet 2 area-of-interest split combination."""

    aoi_id: int
    split: str = "train"

    def __post_init__(self) -> None:
        if self.aoi_id not in _AOI_CITIES:
            raise ValueError(
                f"unknown SpaceNet 2 AOI id {self.aoi_id!r}; "
                f"valid ids are {sorted(_AOI_CITIES)}"
            )
        if self.split not in _VALID_SPLITS:
            raise ValueError(
                f"unknown split {self.split!r}; valid splits are {sorted(_VALID_SPLITS)}"
            )

    @property
    def city(self) -> str:
        return _AOI_CITIES[self.aoi_id]

    @property
    def aoi_dir(self) -> str:
        """S3 key prefix (with trailing '/') under the SpaceNet bucket."""
        return f"{SPACENET_SN2_PREFIX}/{self.split}/AOI_{self.aoi_id}_{self.city}/"

    def s3_uri(self) -> str:
        return f"s3://{SPACENET_BUCKET}/{self.aoi_dir}"


class SpaceNetFetcher:
    """Anonymous read-only client for SpaceNet 2 on AWS Open Data.

    The client refuses to silently fall back to signed access; if anonymous
    listing fails, we raise SpaceNetAccessError so the caller has to make an
    explicit decision rather than a benchmark drifting to a different dataset.
    """

    def __init__(self, region_name: str = "us-east-1") -> None:
        self._s3 = boto3.client(
            "s3",
            config=Config(signature_version=UNSIGNED, region_name=region_name),
        )
        self._region = region_name

    # -- introspection ---------------------------------------------------

    def verify_access(self, aoi: SpaceNetAOI) -> str:
        """Confirm we can list the AOI. Returns the exact bucket path used.

        Raises SpaceNetAccessError with an actionable message on failure.
        """
        try:
            resp = self._s3.list_objects_v2(
                Bucket=SPACENET_BUCKET, Prefix=aoi.aoi_dir, MaxKeys=1
            )
        except NoCredentialsError as e:  # pragma: no cover - defensive
            raise SpaceNetAccessError(
                "boto3 tried to sign the request; anonymous fetcher should not "
                "have done that. Check that UNSIGNED signature_version is honored."
            ) from e
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "?")
            raise SpaceNetAccessError(
                f"failed to list {aoi.s3_uri()} anonymously (code={code}). "
                "SpaceNet is normally publicly readable; if this persists the "
                "bucket may have moved or the caller's network is blocking S3. "
                "Do NOT substitute a different dataset."
            ) from e

        n = resp.get("KeyCount", 0)
        if n == 0:
            raise SpaceNetAccessError(
                f"{aoi.s3_uri()} listed successfully but is empty. "
                "This is unexpected for a public SpaceNet AOI; stopping."
            )
        _LOG.info("SpaceNet access ok: %s (KeyCount>=%d)", aoi.s3_uri(), n)
        return aoi.s3_uri()

    def list_keys(self, prefix: str) -> Iterable[str]:
        """Yield every object key under `prefix` via paginated listing."""
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=SPACENET_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                yield obj["Key"]

    # -- imagery + reference polygons ------------------------------------

    def list_image_ids(self, aoi: SpaceNetAOI, image_kind: str = "PS-RGB") -> list[str]:
        """Return the sorted set of image identifiers (e.g. 'img101') present
        in both the image directory AND the buildings-geojson directory.

        Returning only the intersection keeps downstream code from ever
        looking at an image without a reference polygon file (or vice versa).
        """
        if image_kind not in _VALID_IMAGE_KINDS:
            raise ValueError(
                f"unknown image_kind {image_kind!r}; "
                f"valid are {sorted(_VALID_IMAGE_KINDS)}"
            )

        img_prefix = f"{aoi.aoi_dir}{image_kind}/"
        geo_prefix = f"{aoi.aoi_dir}geojson_buildings/"

        img_ids: set[str] = set()
        for key in self.list_keys(img_prefix):
            name = key.rsplit("/", 1)[-1]
            if not name.lower().endswith(".tif"):
                continue
            # Format: SN2_buildings_train_AOI_2_Vegas_PS-RGB_img101.tif
            stem = name.rsplit(".", 1)[0]
            iid = stem.rsplit("_", 1)[-1]  # "img101"
            if iid.startswith("img"):
                img_ids.add(iid)

        geo_ids: set[str] = set()
        for key in self.list_keys(geo_prefix):
            name = key.rsplit("/", 1)[-1]
            if not name.lower().endswith(".geojson"):
                continue
            # Format: SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img101.geojson
            stem = name.rsplit(".", 1)[0]
            iid = stem.rsplit("_", 1)[-1]
            if iid.startswith("img"):
                geo_ids.add(iid)

        both = sorted(
            img_ids & geo_ids,
            key=lambda s: int(s.removeprefix("img")) if s.removeprefix("img").isdigit() else -1,
        )
        return both

    def download_pair(
        self,
        aoi: SpaceNetAOI,
        image_id: str,
        out_dir: Path,
        image_kind: str = "PS-RGB",
        overwrite: bool = False,
    ) -> tuple[Path, Path]:
        """Download one (image, buildings-geojson) pair. Returns local paths."""
        out_dir = Path(out_dir)
        img_dir = out_dir / image_kind
        geo_dir = out_dir / "geojson_buildings"
        img_dir.mkdir(parents=True, exist_ok=True)
        geo_dir.mkdir(parents=True, exist_ok=True)

        img_name = f"SN2_buildings_{aoi.split}_AOI_{aoi.aoi_id}_{aoi.city}_{image_kind}_{image_id}.tif"
        geo_name = f"SN2_buildings_{aoi.split}_AOI_{aoi.aoi_id}_{aoi.city}_geojson_buildings_{image_id}.geojson"

        img_key = f"{aoi.aoi_dir}{image_kind}/{img_name}"
        geo_key = f"{aoi.aoi_dir}geojson_buildings/{geo_name}"

        img_path = img_dir / img_name
        geo_path = geo_dir / geo_name

        self._download_one(img_key, img_path, overwrite=overwrite)
        self._download_one(geo_key, geo_path, overwrite=overwrite)
        return img_path, geo_path

    def _download_one(self, key: str, dest: Path, overwrite: bool) -> None:
        if dest.exists() and not overwrite:
            _LOG.info("skip (exists): %s", dest)
            return
        _LOG.info("fetching s3://%s/%s -> %s", SPACENET_BUCKET, key, dest)
        try:
            self._s3.download_file(SPACENET_BUCKET, key, str(dest))
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "?")
            raise SpaceNetAccessError(
                f"failed to download s3://{SPACENET_BUCKET}/{key} (code={code}); "
                "not substituting a different source. Investigate before retrying."
            ) from e
