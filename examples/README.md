# Examples

## Generated synthetic run

After installing the package in the `splash` environment:

```bash
geoharmony-splash demo --output-dir /tmp/splash-demo
```

Use a new directory. The command generates small raster inputs, auxiliary
acquisition-row bands, a processing configuration and known homographies for
those generated inputs. It then runs the three stages headlessly.

Inspect the generated configuration and output run metadata to locate the
inputs and final rasters. This example exercises installation and execution.
Its constructed scene and supplied homographies do not measure real-world
registration accuracy.

## Configure your own data

`config.json` is a portable template. It refers to files you supply under a
`data/` directory next to the checkout's `examples/` directory:

```text
data/
  vnir_with_rows.tif
  swir_with_rows.tif
  msi.tif
examples/
  config.json
outputs/
  example/
```

The data files are not included. You may move the configuration and use another
directory layout; relative paths are resolved from the configuration's own
directory.

Before running:

1. Set the three input paths and an output directory.
2. Ensure both hyperspectral cubes have valid acquisition-row IDs in the last
   band; see [input preparation](../docs/input-data.md).
3. Replace the illustrative first-three-band selections with bands appropriate
   for your sensors and their usable spectral overlap.
4. Keep the optional DSM as `null` unless you have verified its alignment and
   metre-based vertical units.

Validate, then start processing:

```bash
geoharmony-splash --config examples/config.json --validate-only
geoharmony-splash --config examples/config.json
```

The first processing run uses the manual correspondence and preview interfaces.
To reuse the resulting transformations for the same inputs, set
`use_available_homography` to `true` and run with `--no-gui`.

All band indices are zero-based. `[0, 2]` is an inclusive range selecting bands
0, 1 and 2. Do not select the auxiliary acquisition-row band as a spectral band.
