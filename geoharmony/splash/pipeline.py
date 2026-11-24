"""The three complete SPLASH spatial-registration stages."""
import copy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import uuid

from geoharmony.splash.config import SplashConfig
from geoharmony.splash.provenance import file_sha256, raster_identity


def manual_reference_identity(identities, sensor):
    grid = {key: identities[sensor][key] for key in ("width", "height", "geotransform", "projection")}
    return {"msi": identities["mica"], "grid": grid, "preprocessing_version": 1}


def run_splash(config: SplashConfig, *, port=8050, open_browser=True, no_gui=False):
    config.validate_inputs()
    # Set before the first import of ANTs/ITK, including in spawned processes.
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(config.itk_threads)
    os.environ["ANTS_RANDOM_SEED"] = str(config.random_seed)
    import numpy as np
    from osgeo import gdal
    from geoharmony.rigid_coregisteration.launchers import coregister_gui, homography_metadata, validate_saved_homography
    from geoharmony.splash.scripts.shape_shifter import shape_shift_mpp_gimg
    from geoharmony.splash.scripts.antspy_registration_onraster import antspy_vnir_swir_gimg
    from geoharmony.splash.scripts.nonground_mask_creator import get_mask_gimg
    from geoharmony.utils.gdalimage import GdalImage
    from geoharmony.utils.gdalwriter import to_envi, to_envi_scale, warp_extent_res, set_zeros_base_ref

    gdal.UseExceptions()
    out = Path(config.out_folder)
    out.mkdir(parents=True, exist_ok=True)
    lock = out / ".splash-running"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError("This output directory is already in use. If an earlier process was killed, remove .splash-running only after confirming it has stopped.") from exc
    os.close(descriptor)
    started = time.perf_counter()
    metadata = {"schema_version": 1, "software_version": "0.1.0", "status": "running",
                "started_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.platform(),
                "python": platform.python_version(), "configuration": config.to_dict(), "stage_seconds": {}}
    memory_root = f"/vsimem/splash_{uuid.uuid4().hex}"
    manifest = out / "run_metadata.json"

    def record():
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")

    def stage_directory(name):
        path = f"{memory_root}/{name}" if config.save_to_memory else str(out / "intermediates" / name)
        if config.save_to_memory:
            gdal.Mkdir(path, 0o755)
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
        return path

    def clean_input(path, sensor, directory):
        image = GdalImage(path)
        values = image.read().astype(np.float32)
        for index in range(values.shape[0] - (sensor != "mica")):
            band = image.ds.GetRasterBand(index+1)
            nodata = band.GetNoDataValue()
            invalid = ~np.isfinite(values[index])
            # Within a known acquisition footprint, spectral zero is a valid
            # measurement. Honor independent masks and nonzero sentinels.
            flags = band.GetMaskFlags()
            implicit_zero_mask = sensor != "mica" and nodata == 0 and flags & gdal.GMF_NODATA
            if not implicit_zero_mask:
                invalid |= band.GetMaskBand().ReadAsArray() == 0
            if nodata is not None and not (sensor != "mica" and nodata == 0):
                invalid |= values[index] == nodata
            values[index, invalid] = np.nan
        if sensor != "mica":
            valid = (values[-1] > 0) & np.any(np.isfinite(values[:-1]), axis=0)
            values[-1, ~valid] = 0
            values[:-1, ~valid] = 0
        info = copy.copy(image)
        info.band_nodata = [None] * info.bands
        return to_envi(values, info, f"{sensor}_input", dtype=np.float32, output_folder=directory)

    try:
        if no_gui and not config.use_available_homography:
            raise ValueError("--no-gui requires use_available_homography=true and matching saved manual homographies")
        metadata["versions"] = {name: importlib.metadata.version(name) for name in
            ("numpy", "scipy", "GDAL", "rasterio", "opencv-python-headless", "scikit-learn", "scikit-image", "antspyx", "dash")}
        source_root = Path(__file__).resolve().parents[1]
        metadata["source_sha256"] = {str(path.relative_to(source_root)): file_sha256(path) for path in sorted(source_root.rglob("*.py"))}
        print("Validating input identities and recording configuration...", flush=True)
        identities = {sensor: raster_identity(getattr(config, f"{sensor}_path")) for sensor in ("vnir", "swir", "mica")}
        if config.mica_dsm_path:
            identities["dsm"] = raster_identity(config.mica_dsm_path)
        metadata["inputs"] = identities
        transformations = out / "transformations"
        transformations.mkdir(exist_ok=True)
        if config.use_available_homography:
            for sensor in ("vnir", "swir"):
                expected = homography_metadata(identities[sensor], manual_reference_identity(identities, sensor))
                validate_saved_homography(transformations / f"homography_{sensor}.npy", expected)
        config.save(out / "resolved_config.json")
        record()
        if config.save_to_memory:
            gdal.Mkdir(memory_root, 0o755)
        folders = {sensor: stage_directory(sensor) for sensor in ("vnir", "swir", "mica", "elastic")}
        source = {sensor: clean_input(getattr(config, f"{sensor}_path"), sensor, folders[sensor]) for sensor in ("vnir", "swir", "mica")}
        reference, warped = {}, {}
        t0 = time.perf_counter()
        print("Stage 1/3: manual coregistration and homography warping", flush=True)
        for sensor in ("vnir", "swir"):
            reference[sensor] = warp_extent_res(source["mica"], source[sensor], f"{sensor}res", output_folder=folders[sensor])
            warped[sensor] = coregister_gui(reference[sensor], config.coreg_mica_band_indices,
                source[sensor], getattr(config, f"coreg_{sensor}_band_indices"), folders[sensor], str(transformations),
                use_available_homography=config.use_available_homography, name=sensor,
                source_identity=identities[sensor], reference_identity=manual_reference_identity(identities, sensor),
                port=port, open_browser=open_browser, no_gui=no_gui, random_seed=config.random_seed)
            reference[sensor] = set_zeros_base_ref(warped[sensor], reference[sensor], output_folder=folders[sensor])
        metadata["stage_seconds"]["manual"] = time.perf_counter()-t0
        masks = {"vnir": None, "swir": None}
        if config.mica_dsm_path:
            mask = get_mask_gimg(source["mica"], GdalImage(config.mica_dsm_path), output_folder=folders["mica"])
            for sensor in masks:
                masks[sensor] = warp_extent_res(mask, warped[sensor], f"{sensor}res", output_folder=folders[sensor])
        t0 = time.perf_counter()
        print("Stage 2/3: VNIR and SWIR ShapeShifter", flush=True)
        shifted, diagnostics = {}, {}
        for sensor in ("vnir", "swir"):
            lower, upper = getattr(config, f"ss_{sensor}_band_range")
            shifted[sensor], diagnostics[sensor] = shape_shift_mpp_gimg(warped[sensor], reference[sensor],
                hs_bands=list(range(lower, upper+1)), mica_band=getattr(config, f"ss_{sensor}_mica_band_index"),
                pixel_shift=getattr(config, f"pixel_shift_{sensor}"), kernel_size=config.kernel_size,
                num_threads=config.num_threads, mica_mask_gimg=masks[sensor], output_folder=folders[sensor],
                name=sensor, random_seed=config.random_seed, model_folder=str(out / "models" / sensor))
        metadata["stage_seconds"]["shapeshifter"] = time.perf_counter()-t0
        t0 = time.perf_counter()
        print("Stage 3/3: SWIR-to-VNIR elastic registration", flush=True)
        ranges = {sensor: list(range(getattr(config, f"ereg_{sensor}_band_range")[0], getattr(config, f"ereg_{sensor}_band_range")[1]+1)) for sensor in ("vnir", "swir")}
        final_swir, deformation, elastic_qa, _ = antspy_vnir_swir_gimg(shifted["vnir"], shifted["swir"],
            vnir_overlap_band_indices=ranges["vnir"], swir_overlap_band_indices=ranges["swir"],
            output_folder=folders["elastic"], kernel=config.elastic_kernel, sigma_color=config.elastic_sigma_color,
            sigma_space=config.elastic_sigma_space, flow_sigma=config.elastic_flow_sigma,
            num_threads=config.elastic_workers, random_seed=config.random_seed, model_folder=str(out / "models" / "elastic"))
        metadata["stage_seconds"]["elastic"] = time.perf_counter()-t0
        outputs = {}
        for sensor, image in (("vnir", shifted["vnir"]), ("swir", final_swir)):
            path = out / f"{sensor}_splash"
            to_envi_scale(image, str(path), coefficients=None, dtype=np.float32, drop_last_band=True, output_folder=str(out))
            outputs[sensor] = str(path)
            mask_info = copy.copy(image)
            mask_info.bands = 1
            mask_info.band_description = ["Valid acquisition footprint (1 valid, 0 outside)"]
            mask_info.band_nodata = [None]
            footprint = (image.read(image.bands) > 0).astype(np.uint8)
            mask_path = out / f"{sensor}_valid_mask"
            to_envi(footprint[None], mask_info, str(mask_path), dtype=np.uint8, output_folder=str(out))
            outputs[f"{sensor}_valid_mask"] = str(mask_path)
        for name, image in (("vnir_shape_difference", diagnostics["vnir"]), ("swir_shape_difference", diagnostics["swir"]),
                            ("swir_elastic_difference", elastic_qa), ("elastic_deformation_grid", deformation)):
            to_envi(image.read()[None] if image.read().ndim == 2 else image.read(), image, name, dtype=np.uint8, output_folder=str(out))
            outputs[name] = str(out / name)
        output_identities = {name: raster_identity(path) for name, path in outputs.items()}
        metadata.update(status="complete", outputs=outputs, elapsed_seconds=time.perf_counter()-started,
                        finished_utc=datetime.now(timezone.utc).isoformat(),
                        output_identities=output_identities,
                        artifact_sha256={str(path.relative_to(out)): file_sha256(path)
                            for directory in (out / "models", transformations)
                            for path in sorted(directory.rglob("*")) if path.is_file()})
        record()
        return outputs
    except BaseException as exc:
        metadata.update(status="cancelled" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.perf_counter()-started)
        record()
        raise
    finally:
        if config.save_to_memory:
            gdal.RmdirRecursive(memory_root)
        lock.unlink(missing_ok=True)
