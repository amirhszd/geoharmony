# Changelog

## Unreleased

- Add a Linux compatibility matrix for Python 3.10, 3.11 and 3.12, running the
  complete test suite and the generated three-stage CLI example for each
  interpreter. Keep the pinned Python 3.11 production environment as the default.
- Verify wheel installation, dependency consistency, all 40 tests and the full
  synthetic CLI workflow on Linux with Python 3.10.22, 3.11.13 and 3.12.16.
- Add reproducible development commands and automated code-quality checks.

## 0.1.0 — release candidate

- Provide SPLASH processing for new user-supplied VNIR, SWIR and multispectral
  inputs through a local setup interface and portable JSON configuration.
- Expose independent sensor shift limits, resource controls and a processing
  seed.
- Add input validation, acquisition-row preparation and a generated synthetic
  example.
- Save resolved configuration, run provenance, selected correspondences and
  input-associated homographies for repeatable processing.
- Retain floating-point spectral data through elastic resampling and provide
  corrected output-band and no-data handling, explicit final footprint masks,
  and checked acquisition-row shift bounds.
- Document input preparation, parameters, output interpretation and installation.

This release candidate is for processing new data. The generated example is a
software execution check, not an accuracy evaluation. Numerical changes from
legacy scripts mean historical outputs are not guaranteed to match this version.
