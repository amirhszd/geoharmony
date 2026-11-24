# Preparing input data

The processing workflow takes three raster inputs and an optional DSM:

| Input | Contents |
|---|---|
| VNIR | Georeferenced VNIR spectral bands followed by an acquisition-row ID band |
| SWIR | Georeferenced SWIR spectral bands followed by an acquisition-row ID band |
| Multispectral reference | A georeferenced reference mosaic with bands suitable for the two hyperspectral sensors |
| DSM, optional | Elevation data covering the reference footprint, used to derive a ground mask |

Use GDAL-readable rasters such as GeoTIFF or ENVI. For ENVI, provide the binary
file path and keep the corresponding header beside it. Retain wavelength and
band-description metadata when available.

## Acquisition-row IDs are required

ShapeShifter groups pixels by the row in which they were acquired in the native
pushbroom image. Orthorectification can turn a straight acquisition row into a
curved or sloping set of pixels in map coordinates. The auxiliary band carries
those original row IDs through the geometric processing.

The last band of each hyperspectral cube must contain:

- Positive integer IDs for acquisition rows, beginning at `1`.
- The same ID at every pixel originating from the same acquisition row.
- `0` outside valid acquisition support.

The row band is categorical geometry metadata, not a spectral measurement.
Exclude it from all display, ShapeShifter spectral and elastic overlap band
selections. SPLASH preserves it during row correction and removes it from the
final spectral outputs.

### Before orthorectification

Append row IDs to a native waterfall image:

```bash
geoharmony-splash prepare-rows native_vnir.tif --output native_vnir_with_rows.tif
geoharmony-splash prepare-rows native_swir.tif --output native_swir_with_rows.tif
```

Then carry these augmented cubes through the same external georeferencing or
orthorectification workflow as the spectral data. Preserve row IDs with
nearest-neighbor resampling and keep the acquisition-row band last. If the
orthorectification software uses separate resampling rules for spectral and
categorical bands, ensure both products still share exactly the same final
grid and acquisition support.

The row-preparation command does not orthorectify a waterfall image. The main
SPLASH workflow expects the resulting georeferenced/orthorectified inputs.

### Already orthorectified images

Use the acquisition-row band supplied by your preprocessing workflow. If it
was not retained, recover it through that workflow or reprocess the native
data. Filling a new band with the current orthorectified raster's row numbers
does not recover the original acquisition rows.

## Geometry and coverage

Each processing input needs a valid coordinate reference system and raster
transform on a north-up orthorectified grid, without rotation/shear terms in
the raster transform. The datasets must cover a common area and depict the same scene.
They can have different spatial resolutions; SPLASH resamples reference data
onto the working sensor grids and ultimately places SWIR on the VNIR grid.

Input georeferencing should put corresponding scene content within the
available raster footprints. Manual homographies provide the initial global
alignment; ShapeShifter's small pixel-shift search is a subsequent refinement.

## Spectral selections

All configuration band indices are **zero-based**. A range such as `[3, 8]`
includes indices `3, 4, 5, 6, 7, 8`.

Choose bands using your sensor wavelengths and data quality:

1. Select three representative bands for viewing each sensor during manual
   point picking. They need not be literal visible RGB bands for SWIR.
2. For each ShapeShifter correction, select a hyperspectral band range and a
   reference band depicting comparable structures.
3. For elastic registration, select VNIR and SWIR ranges in a usable shared
   spectral region.

The first three bands in `examples/config.json` are illustrative placeholders,
not recommended wavelengths for every sensor. Avoid blank, saturated or
otherwise unusable bands. Check valid-data support as well as the wavelength
range; averaging sparse bands can produce spatial patterns when different
pixels average different sets of measurements.

## No-data and radiometry

Retain the raster's no-data metadata and distinguish acquisition support from
spectral values. The acquisition-row band uses `0` for unsupported pixels.
Inspect the declared no-data values and band coverage before running the
workflow. Use consistent radiometric processing for each sensor's input cube;
do not mix physical bands and display-scaled images in one input.

Within a positive acquisition-row footprint, a finite hyperspectral zero is
treated as a valid measurement. The final valid-mask rasters describe
acquisition support, rather than per-band measurement quality; also check
finite values and your sensor's quality information when measuring outputs.

Provide at least three spectral bands plus the row-ID band for each
hyperspectral sensor, and at least three bands for the reference image. At
least one acquisition row must have 11 or more supported pixels for the row
correction stage; short or unsupported rows can be skipped.

Registration is based on image structure and intensity relationships. It does
not convert DN to reflectance, calibrate sensor gain/offset, or guarantee equal
VNIR/SWIR radiometry.

## Optional DSM

The optional DSM is used to derive a ground mask for ShapeShifter. Its vertical
units must be compatible with the documented height thresholds, which are
intended for metres. The current mask uses a `0.5` lower height threshold and a
`5` height cap after background estimation. See [parameters](parameters.md).

If the DSM's units or registration are not known, leave `mica_dsm_path` as
`null` while preparing a verified input set.

## Validate before processing

```bash
geoharmony-splash --config path/to/config.json --validate-only
```

Validation checks the configured input structure and selections. It does not
certify sensor calibration or registration quality. Inspect the point-picking
previews and final outputs as part of processing a new acquisition.
