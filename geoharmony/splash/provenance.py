"""Input and transformation identities used to prevent accidental cache reuse."""

import hashlib
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raster_identity(path):
    from osgeo import gdal

    path = Path(path).resolve()
    ds = gdal.Open(str(path))
    if ds is None:
        raise ValueError(f"Cannot open raster: {path}")
    files = []
    for filename in sorted(ds.GetFileList() or [str(path)]):
        item = Path(filename)
        stat = item.stat()
        files.append(
            {"name": item.name, "size": stat.st_size, "sha256": file_sha256(item)}
        )
    return {
        "files": files,
        "width": ds.RasterXSize,
        "height": ds.RasterYSize,
        "bands": ds.RasterCount,
        "geotransform": list(ds.GetGeoTransform()),
        "projection": ds.GetProjection(),
    }
