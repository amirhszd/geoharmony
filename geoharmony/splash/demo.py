"""A generated synthetic scene for installation and end-to-end verification."""

import json
from pathlib import Path


def create_demo(directory):
    import numpy as np
    from osgeo import gdal, osr

    from geoharmony.rigid_coregisteration.launchers import homography_metadata
    from geoharmony.splash.config import SplashConfig
    from geoharmony.splash.pipeline import manual_reference_identity
    from geoharmony.splash.provenance import file_sha256, raster_identity

    directory = Path(directory).resolve()
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("Choose a new or empty demo output directory")
    data = directory / "data"
    data.mkdir(parents=True, exist_ok=True)
    size = 48
    yy, xx = np.mgrid[:size, :size]
    rng = np.random.default_rng(42)
    signal = (
        0.25
        + 0.15 * np.sin(xx / 3)
        + 0.1 * np.cos(yy / 4)
        + 0.03 * rng.random((size, size))
    )
    signal += 0.3 * np.exp(-((xx - 30) ** 2 + (yy - 18) ** 2) / 55)
    footprint = (xx >= 3) & (xx < size - 3) & (yy >= 2) & (yy < size - 2)
    rows = np.where(footprint, yy + 1, 0).astype(np.float32)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32618)
    paths = {}
    for sensor in ("vnir", "swir", "mica"):
        spectral = np.stack(
            [np.where(footprint, signal * (0.8 + i * 0.04), 0) for i in range(5)]
        ).astype(np.float32)
        # A valid zero measurement, distinct from the exterior footprint.
        spectral[0, 20, 20] = 0
        cube = np.concatenate([spectral, rows[None]]) if sensor != "mica" else spectral
        path = data / f"{sensor}.tif"
        ds = gdal.GetDriverByName("GTiff").Create(
            str(path), size, size, cube.shape[0], gdal.GDT_Float32
        )
        ds.SetProjection(srs.ExportToWkt())
        ds.SetGeoTransform((500000, 0.2, 0, 4700000, 0, -0.2))
        for index, values in enumerate(cube):
            band = ds.GetRasterBand(index + 1)
            band.WriteArray(values)
            band.SetDescription(
                "Acquisition row index"
                if index == 5
                else f"Synthetic spectral band {index + 1}"
            )
        ds.FlushCache()
        ds = None
        paths[sensor] = path
    settings = SplashConfig(
        vnir_path="data/vnir.tif",
        swir_path="data/swir.tif",
        mica_path="data/mica.tif",
        out_folder="outputs",
        ss_vnir_band_range=(0, 2),
        ss_swir_band_range=(0, 2),
        ereg_vnir_band_range=(0, 2),
        ereg_swir_band_range=(0, 2),
        ss_vnir_mica_band_index=1,
        ss_swir_mica_band_index=1,
        use_available_homography=True,
        num_threads=1,
        elastic_workers=1,
        itk_threads=1,
        elastic_kernel=3,
    )
    config_path = directory / "config.json"
    settings.save(config_path)
    config = SplashConfig.load(config_path)
    identities = {sensor: raster_identity(path) for sensor, path in paths.items()}
    transformations = directory / "outputs" / "transformations"
    transformations.mkdir(parents=True)
    for sensor in ("vnir", "swir"):
        path = transformations / f"homography_{sensor}.npy"
        np.save(path, np.eye(3))
        metadata = homography_metadata(
            identities[sensor], manual_reference_identity(identities, sensor)
        )
        metadata["homography_sha256"] = file_sha256(path)
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        corners = [[3, 2], [44, 2], [44, 45], [3, 45]]
        (transformations / f"points_{sensor}.json").write_text(
            json.dumps(
                {
                    "source_points": corners,
                    "reference_points": corners,
                    "description": "Synthetic identity correspondences, not manually measured landmarks",
                },
                indent=2,
            )
            + "\n"
        )
    return config


def run_demo(directory):
    from geoharmony.splash.pipeline import run_splash

    config = create_demo(directory)
    outputs = run_splash(config, no_gui=True, open_browser=False)
    import numpy as np
    from osgeo import gdal

    for sensor in ("vnir", "swir"):
        result = gdal.Open(outputs[sensor])
        if result.RasterCount != 5 or not np.all(np.isfinite(result.ReadAsArray())):
            raise RuntimeError("Demo output verification failed")
    print(
        f"Synthetic SPLASH demo passed. Inputs, configuration and outputs: {Path(directory).resolve()}"
    )
    return outputs
