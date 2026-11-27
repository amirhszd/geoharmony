"""Portable settings shared by the command line and browser interface.

All band indices are zero-based; every ``*_band_range`` includes both endpoints.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class SplashConfig:
    vnir_path: str = ""
    swir_path: str = ""
    mica_path: str = ""
    mica_dsm_path: str | None = None
    out_folder: str = ""
    coreg_vnir_band_indices: tuple = (0, 1, 2)
    coreg_swir_band_indices: tuple = (0, 1, 2)
    coreg_mica_band_indices: tuple = (0, 1, 2)
    ss_vnir_band_range: tuple = (0, 1)
    ss_swir_band_range: tuple = (0, 1)
    ss_vnir_mica_band_index: int = 0
    ss_swir_mica_band_index: int = 0
    pixel_shift_vnir: int = 3
    pixel_shift_swir: int = 3
    kernel_size: int = 3
    ereg_vnir_band_range: tuple = (0, 1)
    ereg_swir_band_range: tuple = (0, 1)
    elastic_kernel: int = 20
    elastic_sigma_color: float = 15
    elastic_sigma_space: float = 20
    elastic_flow_sigma: float = 3
    num_threads: int = 2
    elastic_workers: int = 2
    itk_threads: int = 2
    random_seed: int = 42
    save_to_memory: bool = False
    use_available_homography: bool = False

    @classmethod
    def from_dict(cls, data, base_dir=None):
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(
                "Unknown configuration fields: " + ", ".join(sorted(unknown))
            )
        cfg = cls(**data)
        base = Path(base_dir or Path.cwd())
        for name in (
            "vnir_path",
            "swir_path",
            "mica_path",
            "mica_dsm_path",
            "out_folder",
        ):
            value = getattr(cfg, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"{name} must be a path string"
                    + (" or null" if name == "mica_dsm_path" else "")
                )
            if value is None and name != "mica_dsm_path":
                raise ValueError(f"{name} must be a path string")
            if value:
                path = Path(value).expanduser()
                setattr(
                    cfg,
                    name,
                    str(
                        (base / path).resolve()
                        if not path.is_absolute()
                        else path.resolve()
                    ),
                )
        cfg.validate_settings()
        return cfg

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        with path.open() as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object")
        return cls.from_dict(data, path.parent)

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    def validate_settings(self):
        for name in (
            "pixel_shift_vnir",
            "pixel_shift_swir",
            "elastic_kernel",
            "num_threads",
            "elastic_workers",
            "itk_threads",
            "kernel_size",
            "random_seed",
        ):
            value = getattr(self, name)
            minimum = (
                0 if name.startswith("pixel_shift") or name == "random_seed" else 1
            )
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.random_seed >= 2**32:
            raise ValueError("random_seed must be less than 2**32")
        if self.kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd (1 disables median filtering)")
        for name in (
            "elastic_sigma_color",
            "elastic_sigma_space",
            "elastic_flow_sigma",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < value < float("inf")
            ):
                raise ValueError(f"{name} must be finite and positive")
        for name in ("save_to_memory", "use_available_homography"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false")
        for name in ("ss_vnir_mica_band_index", "ss_swir_mica_band_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative, zero-based band index")
        for name in (
            "ss_vnir_band_range",
            "ss_swir_band_range",
            "ereg_vnir_band_range",
            "ereg_swir_band_range",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or any(
                    isinstance(i, bool) or not isinstance(i, int) or i < 0
                    for i in value
                )
                or value[0] > value[1]
            ):
                raise ValueError(
                    f"{name} must be [start, end] with 0 <= start <= end; endpoints are inclusive"
                )
        for name in (
            "coreg_vnir_band_indices",
            "coreg_swir_band_indices",
            "coreg_mica_band_indices",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 3
                or any(
                    isinstance(i, bool) or not isinstance(i, int) or i < 0
                    for i in value
                )
            ):
                raise ValueError(
                    f"{name} must contain three nonnegative band indices in display channel order"
                )

    def validate_inputs(self):
        """Validate raster metadata and row identifiers without altering inputs."""
        import numpy as np
        from osgeo import gdal, osr

        self.validate_settings()
        datasets = {}
        for sensor, name in (
            ("vnir", "vnir_path"),
            ("swir", "swir_path"),
            ("mica", "mica_path"),
        ):
            path = getattr(self, name)
            if not path or not Path(path).is_file():
                raise ValueError(
                    f"{name}: select an existing raster binary or GeoTIFF (not its .hdr file)"
                )
            ds = gdal.Open(path)
            if ds is None or not ds.GetProjection():
                raise ValueError(f"{name}: raster must be readable and have a CRS")
            srs = osr.SpatialReference(wkt=ds.GetProjection())
            if srs.Validate() != 0 or ds.GetGeoTransform(can_return_null=True) is None:
                raise ValueError(f"{name}: valid georeferencing is required")
            gt = ds.GetGeoTransform()
            if gt[2] != 0 or gt[4] != 0 or gt[1] <= 0 or gt[5] >= 0:
                raise ValueError(
                    f"{name}: use a north-up orthorectified grid before SPLASH"
                )
            minimum = 4 if sensor != "mica" else 3
            if ds.RasterCount < minimum:
                raise ValueError(
                    f"{name}: needs at least {minimum} bands"
                    + (
                        " including a final acquisition-row index band"
                        if sensor != "mica"
                        else ""
                    )
                )
            count = ds.RasterCount - (sensor != "mica")
            display = getattr(self, f"coreg_{sensor}_band_indices")
            if max(display) >= count:
                raise ValueError(
                    f"coreg_{sensor}_band_indices: selected band exceeds the {count} available spectral bands"
                )
            if sensor != "mica":
                for prefix in ("ss", "ereg"):
                    if getattr(self, f"{prefix}_{sensor}_band_range")[1] >= count:
                        raise ValueError(
                            f"{prefix}_{sensor}_band_range: excludes the last acquisition-row band; maximum spectral index is {count - 1}"
                        )
                band = ds.GetRasterBand(ds.RasterCount)
                positive = 0
                has_wide_row = False
                row_counts = {}
                for y in range(0, ds.RasterYSize, 256):
                    rows = band.ReadAsArray(
                        0, y, ds.RasterXSize, min(256, ds.RasterYSize - y)
                    )
                    if (
                        not np.all(np.isfinite(rows))
                        or np.any(rows < 0)
                        or np.any(rows != np.rint(rows))
                    ):
                        raise ValueError(
                            f"{name}: last band must contain integer acquisition-row IDs (0 for background), preserved with nearest-neighbor resampling"
                        )
                    ids, counts = np.unique(rows[rows > 0], return_counts=True)
                    positive += int(counts.sum())
                    for key, n in zip(ids, counts):
                        row_counts[int(key)] = row_counts.get(int(key), 0) + int(n)
                        has_wide_row |= row_counts[int(key)] > 10
                if not positive or not has_wide_row:
                    raise ValueError(
                        f"{name}: last band has no acquisition row with at least 11 pixels; verify row-index preparation"
                    )
            datasets[sensor] = ds

        # Raster extents are compared in the target CRS; the MSI may use a
        # different projection. This catches wrong-scene inputs early.
        def bounds(ds):
            gt = ds.GetGeoTransform()
            return (
                gt[0],
                gt[3] + ds.RasterYSize * gt[5],
                gt[0] + ds.RasterXSize * gt[1],
                gt[3],
            )

        for sensor in ("vnir", "swir"):
            target = osr.SpatialReference(wkt=datasets[sensor].GetProjection())
            reference = osr.SpatialReference(wkt=datasets["mica"].GetProjection())
            target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            reference.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            extent = bounds(datasets["mica"])
            if not reference.IsSame(target):
                extent = osr.CoordinateTransformation(
                    reference, target
                ).TransformBounds(*extent, 21)
            image_extent = bounds(datasets[sensor])
            if max(extent[0], image_extent[0]) >= min(
                extent[2], image_extent[2]
            ) or max(extent[1], image_extent[1]) >= min(extent[3], image_extent[3]):
                raise ValueError(
                    f"{sensor}_path: raster extent does not overlap the MSI reference; verify the scene and CRS"
                )
        for name in ("ss_vnir_mica_band_index", "ss_swir_mica_band_index"):
            if getattr(self, name) >= datasets["mica"].RasterCount:
                raise ValueError(f"{name}: selected MSI band does not exist")
        if self.mica_dsm_path:
            ds = gdal.Open(self.mica_dsm_path)
            if ds is None or ds.RasterCount != 1 or not ds.GetProjection():
                raise ValueError(
                    "mica_dsm_path: select a georeferenced, single-band DSM with heights in metres"
                )
        if not self.out_folder:
            raise ValueError("out_folder: choose a separate writable output directory")
        out = Path(self.out_folder).resolve()
        for name in ("vnir_path", "swir_path", "mica_path", "mica_dsm_path"):
            value = getattr(self, name)
            if value and out in Path(value).resolve().parents:
                raise ValueError(
                    "Use a separate output directory, outside the directories containing input rasters"
                )
        parent = out
        while not parent.exists():
            parent = parent.parent
        import os

        if not parent.is_dir() or not os.access(parent, os.W_OK):
            raise ValueError("out_folder: directory is not writable")
        return {
            name: {
                "bands": ds.RasterCount,
                "width": ds.RasterXSize,
                "height": ds.RasterYSize,
            }
            for name, ds in datasets.items()
        }
