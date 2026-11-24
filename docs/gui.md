# Interface and processing workflow

## Start the local interface

```bash
geoharmony-splash
```

Use `--config path/to/config.json` to supply a prepared configuration, `--port`
to change the setup port, or `--no-browser` to open the interface manually.
The browser displays a local application. File paths refer to the machine
running the Python process.

## 1. Inputs and resources

Choose the VNIR, SWIR and multispectral reference rasters, an output directory,
and optionally a reference DSM. Each hyperspectral raster must have the
acquisition-row ID band described in [input preparation](input-data.md).

Start with conservative worker counts and disk-backed intermediates. Use a
separate output directory for a different dataset or substantially changed
configuration.

## 2. Manual global coregistration

Choose three representative display bands for each image. SPLASH resamples the
reference to each sensor grid and asks for corresponding locations in the
reference and hyperspectral image.

- Pick distinct, stable features visible in both images.
- Use at least four corresponding pairs; the point-picker recommends 8–20.
- Spread points across the overlap instead of concentrating them in one small
  region or along a line.
- Pick each pair in the same order in the two views.
- Review the warped preview. Correct correspondences if the global alignment
  is unsuitable before continuing.

This stage estimates a projective homography, rather than restricting the
transformation to translation/rotation. VNIR and SWIR each receive a separate
homography relative to the reference. The selected control points and saved
transformations belong to those input files.

## 3. ShapeShifter

Choose a spectral range for each hyperspectral sensor and one reference band
for comparison. The range is averaged to create the registration image.

`pixel_shift_vnir` and `pixel_shift_swir` control the independent integer search
radii. A value of `3` searches shifts from `-3` through `+3`, inclusive. The
spectral data for an accepted acquisition-row shift are moved together.

`kernel_size` sets the bandwise median-filter size after row correction. This
filter is part of the ShapeShifter block, so changing it changes the output
even when accepted row shifts are small.

## 4. Elastic refinement

Select VNIR and SWIR overlap-band ranges. SPLASH forms comparison images,
applies bilateral preprocessing, and estimates an ANTsPy SyN registration with
VNIR fixed and SWIR moving.

The elastic controls govern preprocessing and deformation regularization.
See [parameters](parameters.md) for their meanings. The final SWIR raster is
written on the corrected VNIR grid.

## 5. Inspect and record the run

Keep the final ENVI binaries and headers together. Read `run_metadata.json`
and `resolved_config.json` in the output directory to identify the input
files, chosen bands and processing settings.

The `vnir_valid_mask` and `swir_valid_mask` rasters identify each final
acquisition footprint (1 valid, 0 outside). Apply the relevant mask when
displaying or measuring the output; a spectral value of zero can be valid
inside that footprint.

Check alignment at multiple features across the common footprint. Diagnostic
differences and deformation images help inspect processing behavior, but they
are not positional RMSE or calibrated uncertainty.

## Reuse homographies and run headlessly

For repeated processing of the same inputs, set:

```json
"use_available_homography": true
```

Then run:

```bash
geoharmony-splash --config path/to/config.json --no-gui
```

The configured output directory must contain:

```text
transformations/
  homography_vnir.npy
  homography_vnir.json
  homography_swir.npy
  homography_swir.json
```

The input metadata must match the current files. Copying a matrix from another
site, sensor geometry, crop or resized image does not make it valid for the new
inputs. A headless run validates the saved transformations rather than opening
the point picker.

## Portable configuration

JSON paths are resolved relative to the configuration file's directory, not
the shell's working directory. All band indices are zero-based and range
endpoints are inclusive. The [example configuration](../examples/config.json)
shows every public control used in the guided workflow.

Band labels may also show a one-based `Band N` for matching external raster
tools. The explicit index displayed in the interface and saved JSON remains
zero-based.
