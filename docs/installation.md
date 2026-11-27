# Installation

SPLASH uses GDAL for raster I/O, OpenCV and scikit-learn for registration
components, ANTsPy for elastic registration, and Dash for its local browser
interface. Use the pinned Python 3.11 Conda environment below for the default
installation. Python 3.10, 3.11 and 3.12 have each passed Linux wheel-install,
dependency, test-suite and generated-workflow checks; see the
[development guide](development.md).

Windows and macOS installation and interface behavior have not yet been
validated.

## From a source checkout

```bash
git clone --branch splash-public https://github.com/amirhszd/geoharmony.git
cd geoharmony
conda env create -f environment.yml
conda activate splash
python -m pip install --no-deps -e .
```

`--no-deps` uses the dependencies installed by the environment file rather than
requesting a second installation of the GDAL/ANTs stack through pip.

The resolved Linux x86_64 package builds used for release checks are recorded
in `environment-linux-64.lock.yml`. To use that specification, replace the
environment-creation command above with:

```bash
conda env create -f environment-linux-64.lock.yml
```

Activate `splash` before running the CLI or Python commands so GDAL can locate
the environment's PROJ database and other runtime resources.

Check the installation:

```bash
geoharmony-splash --help
python -c "from osgeo import gdal; import ants, cv2; print(gdal.VersionInfo()); print(ants.__version__); print(cv2.__version__)"
```

For a small end-to-end check:

```bash
geoharmony-splash demo --output-dir /tmp/splash-demo
```

Use a fresh example directory. The generated images and known homographies are
for software verification; they do not establish performance on a real sensor
or site.

The compatibility environments are generated from the same dependency
specification with only the Python version changed. The production
`environment.yml` and Linux lock specification retain Python 3.11.

## Browser interface

```bash
geoharmony-splash
```

The interface runs on the same machine as the Python process. Paths entered in
the interface refer to that machine's filesystem. Use a different setup port if
the default port is occupied:

```bash
geoharmony-splash --port 8051
```

To suppress automatically opening a browser:

```bash
geoharmony-splash --no-browser
```

## Without an interactive display

Validate a configuration without starting the interface or processing:

```bash
geoharmony-splash --config path/to/config.json --validate-only
```

For processing on a server, first obtain the manual homographies for the same
inputs, then reuse them:

```bash
geoharmony-splash --config path/to/config.json --no-gui
```

The [interface guide](gui.md) describes the required transformation files and
input metadata. A fresh real-data registration cannot replace manual
correspondences with the synthetic example's identity transformations.

## Resource settings

Start with `num_threads`, `elastic_workers`, and `itk_threads` set to `2` as in
the example configuration. Worker processes and raster arrays use memory;
increasing workers can increase memory demand. Keep `save_to_memory` set to
`false` for a first full-sized run so intermediate rasters are written to disk.

Large hyperspectral cubes can require substantial RAM and temporary storage.
Begin with a small representative dataset and record the configuration that
completes successfully on your system.

## Troubleshooting

- **`geoharmony-splash` is not found:** activate the `splash` environment and
  rerun `python -m pip install --no-deps -e .` from the checkout.
- **GDAL cannot open an ENVI input:** pass the binary raster path, and keep its
  `.hdr` sidecar alongside it. A header alone is not the data cube.
- **The interface cannot locate a file:** paths are interpreted on the server
  running Python, not on a separate browser machine.
- **A headless run rejects a homography:** check that its metadata matches the
  current input files. Generate new correspondences if the inputs changed.
- **Memory use is too high:** reduce worker counts, keep intermediates on disk,
  and test a smaller dataset before processing the full acquisition.
