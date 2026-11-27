# Development and verification

## Python compatibility

SPLASH supports Python 3.10, 3.11 and 3.12 on Linux. Python 3.11 remains the
default installation, with its dependencies pinned in `environment.yml` and
the resolved Linux builds recorded in `environment-linux-64.lock.yml`.

The compatibility matrix uses the production dependency pins and changes only
the Python version. The following Linux results were verified on 2026-11-27
using independently built and installed wheels, with checks run outside the
checkout. All 17 directly pinned dependency versions were identical across
the three environments.

| Python | Dependency check | Test suite | Three-stage CLI demo |
|---|---|---|---|
| 3.10.22 | Passed | 40 passed | Passed |
| 3.11.13 | Passed | 40 passed | Passed |
| 3.12.16 | Passed | 40 passed | Passed |

Windows and macOS have not been validated.

The `SPLASH smoke tests` workflow has a job for each interpreter. Each job
builds and installs a wheel, then runs the checks outside the checkout so they
exercise the installed package, including its CLI and packaged interface assets.

To create a compatibility environment for Python 3.12 from the repository root:

```bash
python scripts/ci_environment.py --python-version 3.12 --output /tmp/splash-environment-3.12.yml
conda env create --name splash-py312 --file /tmp/splash-environment-3.12.yml
conda activate splash-py312
python -m pip install --no-deps -e .
```

Use `3.10` and a distinct environment name to check Python 3.10. Requesting
`3.11` retains the production patch-version pin. The generator writes the
selected output file; it leaves `environment.yml` unchanged. Then run all
checks below from the checkout root.

## Lint and formatting

Install the development tools in the activated environment:

```bash
python -m pip install -r requirements-dev.txt
python -m ruff check geoharmony tests scripts
python -m ruff format --check geoharmony tests scripts
```

`requirements-dev.txt` pins Ruff to version `0.17.6`, and the repository
configuration requires that version. Ruff checks selected Python syntax,
error, import-order and unused-code rules, and verifies formatting. The
configuration uses Python 3.10 as its minimum target, alongside runtime tests
for each interpreter. These checks do not replace tests or scientific validation.

For intentional formatting changes, run:

```bash
python -m ruff format geoharmony tests scripts
```

Review the resulting diff, then rerun both quality checks. The `Code quality`
workflow runs the same commands in its `Ruff lint and format` job.

## Tests and generated workflow

Activate the chosen Conda environment and install the checkout using the
[installation instructions](installation.md). From the repository root, run:

```bash
python -m pip check
geoharmony-splash --help
python -m geoharmony --help
python -m unittest discover -s tests -v
geoharmony-splash demo --output-dir /tmp/splash-development-demo
```

Choose a fresh output directory for each demo. Set `MPLBACKEND=Agg` when
running in a headless environment.

The suite currently contains 40 tests. It checks configuration types and band
selections, raster and acquisition-row handling, command errors, browser
callbacks, saved-transform validation, and pipeline behavior. The separate demo
runs the complete manual-homography, ShapeShifter and elastic stages using
generated inputs and known starting homographies. Its successful execution
does not measure positional accuracy on real acquisitions.

## Contributing changes

Keep the production environment pins intact when checking an additional Python
version. Record the interpreter version and dependency-check results with test
results. If a dependency combination fails, resolve that failure before
claiming support for the combination.

Run the repository's configured code-quality checks as well as the relevant
tests before submitting a change. The test and quality workflow badges report
the current workflow results; they are not measures of registration accuracy.
