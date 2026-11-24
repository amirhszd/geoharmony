# Configuration and hyperparameters

The configuration is a flat JSON object. Paths are resolved relative to the
configuration file. All public band indices are zero-based. Band ranges have
inclusive endpoints; `[2, 5]` selects four bands.

The values below describe this software interface. Default band indices are
placeholders, and must be adapted to the sensor wavelengths and data quality.
Every JSON key listed here is exposed in the setup interface. Fixed internal
settings and backend defaults are identified separately; they have no setup
control in this release. These are software defaults, not settings recovered
from a published experiment.

## Inputs and execution

| Key | Default | Meaning |
|---|---|---|
| `vnir_path` | Required | VNIR spectral cube with acquisition-row IDs in the last band |
| `swir_path` | Required | SWIR spectral cube with acquisition-row IDs in the last band |
| `mica_path` | Required | Multispectral reference mosaic |
| `mica_dsm_path` | `null` | Optional DSM for a ground mask |
| `out_folder` | Required | Separate output directory for this input set and run |
| `num_threads` | `2` | ShapeShifter worker count |
| `elastic_workers` | `2` | Workers applying the elastic transformation to bands |
| `itk_threads` | `2` | ITK/ANTs threading control |
| `random_seed` | `42` | Stochastic component seed |
| `save_to_memory` | `false` | Use memory-backed rather than disk intermediates |
| `use_available_homography` | `false` | Reuse saved, input-validated manual homographies |

Worker counts affect resource use. A fixed seed helps reproducibility; retain
the environment and thread settings with it when comparing runs.

## Manual global alignment

| Key or setting | Default/value | Meaning |
|---|---|---|
| `coreg_vnir_band_indices` | `[0, 1, 2]` | Three VNIR point-picker display bands |
| `coreg_swir_band_indices` | `[0, 1, 2]` | Three SWIR point-picker display bands |
| `coreg_mica_band_indices` | `[0, 1, 2]` | Three reference point-picker display bands |
| Correspondences | At least 4; GUI recommends 8–20 | Matched locations selected interactively |
| Homography estimator | OpenCV RANSAC | Robust projective alignment |
| Reprojection threshold | 5 pixels | RANSAC inlier threshold; fixed by the implementation |
| RANSAC maximum iterations / confidence | 2000 / 0.995 | Fixed internal settings |
| Manual hyperspectral warp | Nearest-neighbor | Applies to spectral bands and categorical acquisition-row IDs |

The homographies are saved in `transformations/` with metadata for their
inputs. `use_available_homography` is a reuse control, not a substitute for
acquiring correspondences for a new dataset.

## ShapeShifter

| Key or setting | Default/value | Meaning |
|---|---|---|
| `ss_vnir_band_range` | `[0, 1]` | VNIR bands averaged for the row-alignment image |
| `ss_swir_band_range` | `[0, 1]` | SWIR bands averaged for the row-alignment image |
| `ss_vnir_mica_band_index` | `0` | Reference comparison band for VNIR |
| `ss_swir_mica_band_index` | `0` | Reference comparison band for SWIR |
| `pixel_shift_vnir` | 3 pixels | Integer VNIR search from −3 to +3 |
| `pixel_shift_swir` | 3 pixels | Integer SWIR search from −3 to +3 |
| `kernel_size` | 3 | Odd, positive bandwise median-filter window; 1 disables it |
| Acquisition-row band | Last hyperspectral band | Positive integer acquisition IDs; 0 outside support |
| Line model | Linear regression with RANSAC | Estimates each acquisition row's orientation |
| RANSAC maximum trials | 20 | Fixed internal setting |
| RANSAC residual threshold | 30 pixels | Fixed internal setting |
| RANSAC stop probability | 0.99 | scikit-learn backend default |
| Minimum fitting/inlier samples | 5 | Rows with insufficient support are skipped |
| Minimum samples when applying a shift | More than 10 | Fixed internal shift support check |
| Objective | scikit-image normalized mutual information (NMI) | `(H(X) + H(Y)) / H(X,Y)`; 100 histogram bins per dimension |
| Objective window | Acquisition row's bounding rectangle | All finite paired samples in that rectangle, restricted to the optional ground mask when enabled |
| Objective intensity preprocessing | Independent uint8 min–max scaling | Applied to each paired comparison vector; constant or insufficient comparisons are skipped |
| Accepted relative MI improvement | Greater than 0.0005 | A shift must improve the registration objective by more than 0.05% |

Exclude the auxiliary row band from spectral selections. The search radius is
measured in each sensor's working pixels, so the same value can represent
different ground distances for VNIR and SWIR. This block includes its median
filter as well as the row shifts.

The integer search includes zero, and each candidate is compared with the
unshifted baseline using `(NMI_after - NMI_before) / NMI_before`. The best finite
candidate is accepted only when that ratio is strictly greater than `0.0005`.
When no candidate is accepted, that row stays unshifted; the median filter
still runs.
The objective is not calculated only on the row's own pixels: it uses the
bounding rectangle described above. The paired finite-sample mask can vary
between candidates.

## Elastic registration

| Key or setting | Default/value | Meaning |
|---|---|---|
| `ereg_vnir_band_range` | `[0, 1]` | VNIR comparison-image bands |
| `ereg_swir_band_range` | `[0, 1]` | SWIR comparison-image bands |
| `elastic_kernel` | 20 | OpenCV bilateral-filter neighborhood diameter |
| `elastic_sigma_color` | 15 | Bilateral intensity-domain sigma |
| `elastic_sigma_space` | 20 | Bilateral spatial-domain sigma |
| `elastic_flow_sigma` | 3 | ANTs SyN update-field smoothing parameter |
| Fixed/moving images | VNIR / SWIR | SWIR is refined onto the corrected VNIR grid |
| Transform family | ANTsPy SyN | Global and deformable registration components |
| SWIR grid matching | Bilinear | Resample SWIR to the VNIR working grid |
| Registration comparison images | Mean, uint8 min–max scaling, bilateral filtering | Fixed preprocessing; the fitted transform is applied to floating-point spectral data |
| Final spectral / row-band interpolation | Linear / nearest-neighbor | Floating-point spectral resampling with categorical acquisition-row IDs handled separately |

Use valid, comparable spectral ranges for the two mean images. The physical
meaning of a band is sensor-specific; indices from an unrelated instrument
are not universal defaults. Band quality matters as well as wavelength.

The following fixed or inherited settings describe the effective `SyN` branch
in the pinned ANTsPy 0.5.4 backend. They are not public JSON controls. ANTsPy's
generic function-signature defaults for affine schedules differ from the
effective schedules used by this branch.

| Backend setting | Effective value |
|---|---|
| Initial transform | `Identity` |
| Affine and deformable similarity metrics | Mattes mutual information, 32 bins |
| Affine sampling | Regular, fraction 0.2 |
| Affine step | 0.25 |
| Affine iterations / shrink factors / smoothing sigmas | `2100/1200/1200/0`; `4/2/2/1`; `3/2/1/0` |
| Deformable iterations / shrink factors / smoothing sigmas | `40/20/0`; `4/2/1`; `2/1/0` |
| Deformable step / total-field smoothing | 0.2 / 0 |
| Deformable convergence threshold / window | `1e-7` / 8 |
| Registration seed | Public `random_seed` value |

ANTs images are constructed from arrays without physical spacing metadata, so
these registration scales operate in the working pixel grid. Keep the pinned
environment when relying on inherited backend settings.

## Optional DSM-derived ground mask

Supplying `mica_dsm_path` enables the ground-mask branch. Its height thresholds
are intended for a DSM in metres. Current internal processing uses:

| Setting | Value |
|---|---|
| Initial DSM median filter | 29 × 29 |
| Background minimum filter | 30 × 30, reflect boundary |
| Background median filter | 5 × 5 |
| Lower relative-height cutoff | 0.5 |
| Relative-height cap | 5 |
| Binary-mask median filter | Size 9 |
| Mask dilation | 5 iterations |
| Final mask convention | Ground = 1; nonground = 0 |

These internal values are not JSON controls in this release. Verify the DSM's
units and alignment before enabling the branch.

## Implementation references

The [homography launcher](../geoharmony/rigid_coregisteration/launchers.py),
[ShapeShifter implementation](../geoharmony/splash/scripts/shape_shifter.py),
[elastic implementation](../geoharmony/splash/scripts/antspy_registration_onraster.py),
and [environment specification](../environment.yml) define these settings.
Backend behavior is version-specific: scikit-image 0.25.2 supplies NMI,
scikit-learn 1.7.0 supplies row RANSAC, OpenCV 4.11.0 supplies homography and
bilateral filtering, and ANTsPy 0.5.4 supplies the elastic registration branch.

## Recorded settings and diagnostics

The output configuration and run metadata are the source of truth for a
particular execution. Keep them with the input identifiers, transformations,
and final rasters. Diagnostic image differences measure changes to sampled
intensities; the deformation visualization shows a warped reference grid.
Neither is calibrated ground-position uncertainty.
