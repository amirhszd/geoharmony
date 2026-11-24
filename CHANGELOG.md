# Changelog

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
