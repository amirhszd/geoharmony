"""Append categorical scanline IDs before orthorectifying a native waterfall."""
from pathlib import Path


def prepare_rows(input_path, output_path):
    import numpy as np
    from osgeo import gdal

    source, output = Path(input_path).resolve(), Path(output_path).resolve()
    if source == output or output.exists():
        raise ValueError("Choose a new output filename; input rasters are never overwritten")
    ds = gdal.Open(str(source))
    if ds is None:
        raise ValueError(f"Cannot read waterfall raster: {source}")
    if ds.GetRasterBand(ds.RasterCount).GetMetadataItem("SPLASH_ROLE") == "acquisition_row":
        raise ValueError("The input already has an acquisition-row band")
    output.parent.mkdir(parents=True, exist_ok=True)
    out = gdal.GetDriverByName("GTiff").Create(str(output), ds.RasterXSize, ds.RasterYSize,
                  ds.RasterCount+1, gdal.GDT_Float32, options=["COMPRESS=DEFLATE", "BIGTIFF=IF_SAFER"])
    transform = ds.GetGeoTransform(can_return_null=True)
    if transform is not None:
        out.SetGeoTransform(transform)
    if ds.GetProjection():
        out.SetProjection(ds.GetProjection())
    out.SetMetadata(ds.GetMetadata())
    valid = np.zeros((ds.RasterYSize, ds.RasterXSize), dtype=bool)
    for i in range(1, ds.RasterCount+1):
        band = ds.GetRasterBand(i)
        data = band.ReadAsArray()
        support = np.isfinite(data) & (band.GetMaskBand().ReadAsArray() != 0)
        nodata = band.GetNoDataValue()
        if nodata is not None:
            support &= data != nodata
        valid |= support
        target = out.GetRasterBand(i)
        target.WriteArray(data)
        target.SetDescription(band.GetDescription() or f"Spectral band {i}")
        if nodata is not None:
            target.SetNoDataValue(nodata)
    rows = np.broadcast_to(np.arange(1, ds.RasterYSize+1, dtype=np.float32)[:, None], valid.shape).copy()
    rows[~valid] = 0
    row_band = out.GetRasterBand(ds.RasterCount+1)
    row_band.WriteArray(rows)
    row_band.SetDescription("Acquisition row index (categorical; nearest-neighbor resampling only)")
    row_band.SetMetadataItem("SPLASH_ROLE", "acquisition_row")
    out.FlushCache()
    out = None
    return str(output)
