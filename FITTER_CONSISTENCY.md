# Least-squares fitter compatibility and review notes

The three classes in `toloof/fitbeam.py` are:

- `fit_beam_with_glob_tilt_M2offsets_leastsquares` (offsets).
- `fit_beam_with_glob_tilt_M2tilts_leastsquares` (tilts).
- `fit_beam_with_pointing_tilt_offsets_leastsquares` (pointing).

## Shared reporting interface

Each class provides `save_results`, `save_uncertainties`, `surface_plot`,
`plot_fit_results`, `combine_result_pngs`, `save_zernike_dat`, and
`save_subref_dat`. Reporting filenames accept strings or path-like objects.
`save_subref_dat(filename, reference_m2z=0.0)` accepts the same arguments in all
three; the reference coordinate affects only the offsets fitter's X/Y export.

After fitting, the same reporting calls work with any of these classes:

```python
fitter.run_fitter(newx0=None)
fitter.save_results('fit.json')       # JSON and table PNG
fitter.save_uncertainties('uncert')   # CSV, JSON and covariance heatmap PNG
fitter.save_zernike_dat('zernike.dat')
fitter.save_subref_dat('subref.dat')
```

`save_uncertainties` works directly after fitting, without `save_results`.
`save_zernike_dat` still requires `save_results` first in all three classes.
The composite plot requires the data/model/residual, result-table, and surface
plots to have been saved first.

All three use one shared implementation for uncertainty estimation and exports:

- CSV contains the covariance matrix in raw `results.x` order, with named rows
  and columns.
- JSON contains **variances, not standard deviations**, in squared native
  units. Its keys and order match that fitter's `save_results` JSON, including
  derived focus targets, moves, and Strehl. Take a square root for a standard
  deviation. A reported zero variance for `M2.Z_middle` treats the recorded
  coordinate as exact; it is not a telescope metrology estimate.
- PNG shows the covariance matrix on a symmetric logarithmic color scale.
- Attributes are `covariance_matrix`, `uncertaintiesdict`, and
  `uncertaintytable_fname`.

Covariance is `(J.T J)^-1 * sum(residual**2)/(N-p)` using column-scaled SVD.
Rank-deficient Jacobians, nonfinite residuals/Jacobians, and `N <= p` are errors.
Derived variances include parameter cross-covariances; Strehl uses a numerical
local gradient. This assumes locally linear behavior, inactive bounds, and
independent equal-variance residuals. It does not model correlated map noise,
coordinate errors, or the sampling distribution of a bound-constrained fit.

Focus reporting is shared: model defocus is recorded M2.Z plus fitted offset;
target is minus that offset; move is target minus the middle recorded position.
The middle map follows dictionary iteration order (the later middle map for even
counts). JSON stores axial positions in meters, tables show mm, and `subref.dat`
exports the absolute focus target in microns. Strehl excludes source amplitude
and pointing Zernikes, and retains defocus and the relevant fitted M2 aberrations
at the middle map.

Pointing coefficients now use the actual map names and iteration order in both
results and covariance labels. For reordered maps, this corrects the previous
association with numerically generated map names. Tables use four decimals
across these classes. Pointing coefficients are Zernike OPD coefficients in
microns, despite their historical `Az_Off_...` / `El_Off_...` names.

## Physical and parameter differences

| Property | Offsets | Tilts | Pointing |
| --- | --- | --- | --- |
| M2 terms | X/Y translations linear in recorded Z | X/Y angles linear in recorded Z | None |
| M2 coefficient units | intercept m, slope m/m | intercept degrees, slope degrees/m | N/A |
| Pointing terms | Two global Zernike tilts | Two global Zernike tilts | Two independent Zernike tilts per map |
| Coma Zernikes (`c[7:9]`) | Excluded | Excluded | Fitted |
| Parameter count for K Zernikes, M maps | K + 2 | K + 2 | K + 2M - 2 |
| Lateral/angle targets and per-map results | `M2.X_*`, `M2.Y_*` | `M2.alphX_*`, `M2.alphY_*` | `Az_Off_*`, `El_Off_*` |
| `subref.dat` X/Y | Fitted translations at `reference_m2z`, flag 1 | Zero, flag 0 | Zero, flag 0 |

Raw fit vectors and covariance matrices cannot be exchanged across classes:
parameter meanings, counts, and units differ. Use named result fields. Angular
M2 terms are preserved in JSON; `subref.dat` has no angular rows.

## Substantial existing behavior for review

The following fitting behavior was preserved while aligning uncertainty and
reporting methods:

1. **Different bound enforcement.** Pointing passes bounds to `least_squares`:
   amplitude is nonnegative and `x[tilt_offset_end_index + 8]` is limited to
   ±250 (this is `c[12]`, SPH, when present). Other coefficients are unbounded.
   Offsets and tilts calculate bounds but do not pass them to the solver, so
   even their amplitude constraints are inactive. `tilt_plusminus` and
   `zern_plusminus` do not currently constrain any of these fits: the pointing
   and offsets implementations ignore their requested limits; tilts builds
   the requested bounds but leaves enforcement disabled. This can change
   fitted solutions and covariance interpretation.
2. **Early stopping is ineffective.** Pointing's `stop_delta_cost` and `patience`
   arguments configure a local callback that is never passed to the optimizer.
   Global fitters do not expose these arguments. Solver defaults determine
   convergence in all three. The positional order of `run_fitter` also differs:
   pointing starts with these stopping arguments, whereas global fitters start
   with the bound arguments. Use common keyword arguments (`newx0`,
   `tilt_plusminus`, `zern_plusminus`) when switching classes.
3. **Initialization differs from the reported focus sign convention.** All three
   initialize the fitted offset to positive recorded `map{M//2}` Z, although
   putting that map at zero modeled defocus would require the negative value.
   They also assume a `map0` key for residual allocation and numerical map names
   for initialization. This is distinct from the iteration-based middle-map
   selection used for final reporting. Review the intended starting guess.
4. **Different coma and pointing flexibility.** Pointing fits shared coma and
   independent per-map pointing; the global fitters exclude explicit coma and
   instead fit M2-induced aberrations. Different solutions, parameter
   correlations, and rank deficiency can therefore be physical/model effects.
5. **Surface plots are partial OPD views.** They show the common Zernike surface
   terms, excluding pointing, defocus, and M2-induced OPD. Pointing's surface
   includes its fitted coma; the other two cannot include explicit coma terms.
   These plots do not depict each map's complete modeled wavefront.
