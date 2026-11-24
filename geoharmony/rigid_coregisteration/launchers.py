"""Managed local browser workflows and validated manual homographies."""
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import socket

import cv2
import numpy as np

from geoharmony.splash.provenance import file_sha256


def collect_from_app(target, args=(), port=8050, open_browser=True):
    """Wait for a browser result; stop children on cancellation or startup failure."""
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError as exc:
            raise RuntimeError(f"Port {port} is busy. Choose another port with --port.") from exc
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=target, args=(*args, queue, port, open_browser))
    process.start()
    print(f"Open http://127.0.0.1:{port} in your browser. Press Ctrl+C in this terminal to cancel.", flush=True)
    try:
        while True:
            try:
                result = queue.get(timeout=0.25)
                if isinstance(result, dict) and result.get("cancelled"):
                    raise KeyboardInterrupt
                return result
            except Empty:
                if not process.is_alive():
                    raise RuntimeError(f"Browser interface exited before submitting a result (exit {process.exitcode})")
    finally:
        process.join(timeout=0.3)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        queue.close()


def calc_homography(ref_points, target_points, random_seed=42):
    """Estimate a source-to-destination projective transform from pixel coordinates."""
    source, destination = np.asarray(ref_points, dtype=float), np.asarray(target_points, dtype=float)
    if source.ndim != 2 or source.shape[1:] != (2,) or source.shape != destination.shape or len(source) < 4:
        raise ValueError("Select at least four matching point pairs with equal counts")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(destination)):
        raise ValueError("Control-point coordinates must be finite")
    for points in (source, destination):
        if len(np.unique(points, axis=0)) < 4 or np.linalg.matrix_rank(points - points.mean(axis=0)) < 2:
            raise ValueError("Choose distinct, non-collinear control points spread across the image")
    cv2.setRNGSeed(int(random_seed) % (2**31 - 1))
    matrix, inliers = cv2.findHomography(source, destination, cv2.RANSAC, 5.0,
                                        maxIters=2000, confidence=0.995)
    if matrix is None or not np.all(np.isfinite(matrix)) or np.linalg.matrix_rank(matrix) < 3 or inliers.sum() < 4:
        raise ValueError("Homography estimation failed; choose better distributed corresponding points")
    return matrix


def homography_metadata(source_identity, reference_identity):
    return {"schema_version": 1, "source": source_identity, "reference": reference_identity}


def validate_saved_homography(path, expected):
    path = Path(path)
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Missing saved transformation or input-identity metadata: {path}. Run manual registration first.")
    saved = json.loads(metadata_path.read_text())
    if any(saved.get(key) != value for key, value in expected.items()) or saved.get("homography_sha256") != file_sha256(path):
        raise ValueError(f"Saved transformation does not match these input rasters and grids: {path}. Use a new output directory or repeat manual registration.")
    matrix = np.load(path, allow_pickle=False)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or np.linalg.matrix_rank(matrix) < 3:
        raise ValueError(f"Invalid homography matrix: {path}")
    return matrix


def coregister_gui(ref_gdalimage, ref_bands, target_gdalimage, target_bands,
                   outfolder_mem, outfolder, use_available_homography=False, name="",
                   source_identity=None, reference_identity=None, port=8050,
                   open_browser=True, no_gui=False, random_seed=42):
    from geoharmony.utils.gdalwriter import save_crop_warp
    from geoharmony.rigid_coregisteration.get_points_app import run_get_points_app
    from geoharmony.rigid_coregisteration.show_warped_app import run_show_warped_app

    folder = Path(outfolder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"homography_{name}.npy"
    expected = homography_metadata(source_identity, reference_identity)
    if use_available_homography:
        matrix = validate_saved_homography(path, expected)
    else:
        if no_gui:
            raise ValueError("--no-gui requires use_available_homography=true and matching saved transformations")
        ref = np.moveaxis(ref_gdalimage.read_display_image(list(ref_bands)), 0, 2)
        target = np.moveaxis(target_gdalimage.read_display_image(list(target_bands)), 0, 2)
        while True:
            ref_points, target_points = collect_from_app(run_get_points_app, (ref, target, name), port, open_browser)
            source = [[point["x"], point["y"]] for point in target_points]
            destination = [[point["x"], point["y"]] for point in ref_points]
            try:
                matrix = calc_homography(source, destination, random_seed)
            except ValueError as exc:
                print(f"{exc}. Reopening point selection.", flush=True)
                continue
            warped = cv2.warpPerspective(target, matrix, (ref.shape[1], ref.shape[0]))
            if collect_from_app(run_show_warped_app, (ref, warped), port, open_browser):
                break
        np.save(path, matrix)
        expected["homography_sha256"] = file_sha256(path)
        path.with_suffix(".json").write_text(json.dumps(expected, indent=2) + "\n")
        (folder / f"points_{name}.json").write_text(json.dumps({"coordinate_system": "zero-based raster pixel (column, row)",
                     "source_points": source, "reference_points": destination}, indent=2) + "\n")
    return save_crop_warp(target_gdalImage=target_gdalimage, ref_gdalImage=ref_gdalimage,
                          homography_matrix=matrix, outfolder=outfolder_mem)


def get_splash_inputs(port=8050, initial=None, open_browser=True):
    from geoharmony.rigid_coregisteration.splash_app import run_splash_app
    return collect_from_app(run_splash_app, (initial,), port, open_browser)
