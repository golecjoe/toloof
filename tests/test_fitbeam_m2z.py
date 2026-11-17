"""Regression checks for the shared axial convention of the M2 and pointing fitters."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares


# Isolate the fitters from unrelated pixell/beamclass imports. The PSF below
# makes axial and angular parameters independently observable in these tests.
SOURCE = Path(__file__).resolve().parents[1] / 'toloof' / 'fitbeam.py'
NAMES = [f'fit_beam_with_glob_tilt_M2{kind}_leastsquares'
         for kind in ('offsets', 'tilts')]
NAMES.append('fit_beam_with_pointing_tilt_offsets_leastsquares')
namespace = dict(np=np, plt=plt, least_squares=least_squares, json=json, os=os)
tree = ast.parse(SOURCE.read_text())
classes = ast.Module(body=[node for node in tree.body
                          if isinstance(node, ast.ClassDef) and node.name in NAMES + ['_LeastSquaresUncertainties']],
                     type_ignores=[])
exec(compile(classes, str(SOURCE), 'exec'), namespace)
OFFSETS, TILTS, POINTING = [namespace[name] for name in NAMES]


class SyntheticBeam:
    def __init__(self, count):
        # Non-numerical iteration order tests the middle-map selection.
        names = ['map1', 'map0', 'map2', 'map3'][:count]
        self.m2z_vals = dict(zip(names, np.linspace(0.002, 0.005, count)))
        self.trunc_maps = {name: np.ones((3, 3)) for name in names}
        self.zernike_polynomials = np.zeros((9, 3, 3))
        self.calls = []

    def make_psf(self, **kwargs):
        self.calls.append(kwargs)
        c = kwargs['c']
        return np.array([
            1., 1e6 * kwargs['secondary_offset'] ** 2,
            kwargs['del_x'] + kwargs['del_alph_x'],
            kwargs['del_y'] + kwargs['del_alph_y'],
            c[1], c[2], c[3], c[5], c[6],
        ]).reshape(3, 3)


class M2ZConsistencyTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_axial_fit_recovery_and_residual_agreement(self):
        for count in (3, 4):
            for offset in (-0.0012, 0.0012):
                with self.subTest(count=count, offset=offset):
                    beam = SyntheticBeam(count)
                    offsets, tilts = OFFSETS(beam), TILTS(beam)
                    x = np.array([2., offset, .03, 2., -.02, -3.,
                                  .1, -.2, .3, -.4, .5])
                    # Both PSF parameterizations have equivalent responses here.
                    residual = tilts.chisquared(x)
                    np.testing.assert_allclose(offsets.chisquared(x), residual)
                    for name, res in zip(beam.trunc_maps, residual.reshape(count, 3, 3)):
                        beam.trunc_maps[name] -= res
                    np.testing.assert_allclose(tilts.chisquared(x), 0., atol=1e-12)
                    initial = x.copy()
                    initial[1] += 0.0002
                    for fitter in (offsets, tilts):
                        fitter.run_fitter(newx0=initial.copy())
                        np.testing.assert_allclose(fitter.results.x, x, atol=1e-7)
                    self.assertAlmostEqual(offsets.strehl_ratio, tilts.strehl_ratio)
                    middle = list(beam.trunc_maps)[count // 2]
                    z = beam.m2z_vals[middle]
                    call = beam.calls[-1]
                    self.assertAlmostEqual(call['secondary_offset'], z + offset)
                    self.assertAlmostEqual(call['del_alph_x'], x[2] + x[3] * z)
                    self.assertAlmostEqual(call['del_alph_y'], x[4] + x[5] * z)
                    self.assertEqual(call['c'][1], 0.)
                    self.assertEqual(call['c'][2], 0.)

    def test_saved_targets_moves_units_and_export(self):
        for count in (3, 4):
            for offset in (-0.0012, 0.0012):
                with self.subTest(count=count, offset=offset), tempfile.TemporaryDirectory() as directory:
                    beam = SyntheticBeam(count)
                    x = np.array([2., offset, .03, 2., -.02, -3.,
                                  .1, -.2, .3, -.4, .5])
                    fitters = [OFFSETS(beam), TILTS(beam), POINTING(beam)]
                    saved = []
                    for index, fitter in enumerate(fitters):
                        fitter_x = x if index < 2 else np.zeros(fitter.fit_vec_size)
                        fitter_x[:2] = x[:2]
                        fitter.results = type('Result', (), {'x': fitter_x})()
                        fitter.strehl_ratio = .8
                        filename = Path(directory) / f'{index}.json'
                        # Capture the actual generated table, including display units.
                        tables = []
                        def capture_table(*args, **kwargs):
                            tables.extend(plt.gcf().axes[0].tables)
                        with patch.object(plt, 'savefig', side_effect=capture_table):
                            fitter.save_results(str(filename))
                        saved.append(json.loads(filename.read_text()))
                        rows = list(tables[0].get_celld().values())
                        text = [cell.get_text().get_text() for cell in rows]
                        for key in ('M2.Z_offset', 'M2.Z_target', 'M2.Z_middle', 'M2.Z_move'):
                            pos = text.index(key)
                            self.assertEqual(text[pos + 1], f'{saved[-1][key] * 1e3:.4f}')
                            self.assertEqual(text[pos + 2], 'mm')
                        dat = Path(directory) / f'{index}.dat'
                        fitter.save_subref_dat(dat)
                        lines = dat.read_text().splitlines()
                        self.assertEqual(lines[0], f'0 1 M2Z {-offset * 1e6:.6f}')
                        if index >= 1:
                            self.assertEqual(lines[1:], ['1 0 M2X 0.000000', '2 0 M2Y 0.000000'])
                    for key in ('M2.Z_offset', 'M2.Z_target', 'M2.Z_middle', 'M2.Z_move'):
                        self.assertEqual(saved[0][key], saved[1][key])
                        self.assertEqual(saved[0][key], saved[2][key])
                    result = saved[1]
                    middle = list(beam.trunc_maps)[count // 2]
                    self.assertEqual(result['M2.Z_target'], -offset)
                    self.assertEqual(result['M2.Z_move'], -offset - beam.m2z_vals[middle])
                    self.assertEqual(result['M2.alphX_target'], x[2] - x[3] * offset)
                    self.assertEqual(result['M2.alphY_target'], x[4] - x[5] * offset)
                    for name in beam.trunc_maps:
                        self.assertEqual(result[f'M2.alphX_{name}'], x[2] + x[3] * beam.m2z_vals[name])


    def test_pointing_axial_fit_and_middle_map_strehl(self):
        for count in (3, 4):
            for offset in (-0.0012, 0.0012):
                with self.subTest(count=count, offset=offset):
                    beam = SyntheticBeam(count)
                    fitter = POINTING(beam)
                    x = np.zeros(fitter.fit_vec_size)
                    x[:2] = [2., offset]
                    # Use the reference fitter to generate axial-only data.
                    reference = OFFSETS(beam)
                    reference_x = np.zeros(reference.fit_vec_size)
                    reference_x[:2] = x[:2]
                    residual = reference.chisquared(reference_x)
                    for name, res in zip(beam.trunc_maps, residual.reshape(count, 3, 3)):
                        beam.trunc_maps[name] -= res
                    np.testing.assert_allclose(fitter.chisquared(x), 0., atol=1e-12)

                    def axial_residual(axial):
                        trial = x.copy()
                        trial[1] = axial[0]
                        return fitter.chisquared(trial)

                    recovered = least_squares(axial_residual, [offset + .0002])
                    self.assertAlmostEqual(recovered.x[0], offset, places=10)
                    # Exercise post-fit reporting independently of solver degeneracies
                    # in the simplified PSF (which omits two Zernike modes).
                    with patch.dict(namespace, least_squares=lambda *args, **kwargs:
                                    type('Result', (), {'x': x})()):
                        fitter.run_fitter()
                    middle = list(beam.trunc_maps)[count // 2]
                    expected_defocus = beam.m2z_vals[middle] + offset
                    self.assertAlmostEqual(beam.calls[-1]['secondary_offset'], expected_defocus)
                    self.assertAlmostEqual(fitter.strehl_ratio,
                                           max(1., 1e6 * expected_defocus**2))
                    self.assertEqual(beam.calls[-1]['c'][1], 0.)
                    self.assertEqual(beam.calls[-1]['c'][2], 0.)
                    # Export must also work directly after fitting, before save_results.
                    with tempfile.TemporaryDirectory() as directory:
                        filename = Path(directory) / 'subref.dat'
                        fitter.save_subref_dat(filename)
                        self.assertEqual(filename.read_text().splitlines()[0],
                                         f'0 1 M2Z {-offset * 1e6:.6f}')



class UncertaintyTests(unittest.TestCase):
    def make_fitter(self, zernike_count=9, fitter_class=TILTS, count=4):
        beam = SyntheticBeam(count)
        beam.zernike_polynomials = np.zeros((zernike_count, 3, 3))
        with contextlib.redirect_stdout(io.StringIO()):
            fitter = fitter_class(beam)
        size = fitter.fit_vec_size
        x = np.linspace(.01, .2, size)
        x[:6] = [2., -.0012, .03, 2., -.02, -3.]
        rng = np.random.default_rng(42)
        jac = rng.normal(size=(3 * size, size)) * np.linspace(1., 10., size)
        residuals = np.linspace(-.2, .3, 3 * size)
        fitter.results = type('Result', (), dict(x=x, jac=jac, fun=residuals))()
        fitter.strehl_ratio = .8
        return fitter

    def test_covariance_outputs_and_propagation(self):
        import csv
        for zernike_count in (9, 16):
            with self.subTest(zernike_count=zernike_count), tempfile.TemporaryDirectory() as directory:
                fitter = self.make_fitter(zernike_count)
                result = fitter.results
                expected = np.linalg.inv(result.jac.T @ result.jac)
                expected *= (result.fun @ result.fun) / (result.fun.size - result.x.size)
                filename = Path(directory) / 'uncertainties.json'
                # Works immediately after fitting, without calling save_results.
                fitter.save_uncertainties(filename)
                variances = json.loads(filename.read_text())
                np.testing.assert_allclose(fitter.covariance_matrix, expected, rtol=1e-10, atol=1e-15)
                with open(filename.with_suffix('.csv'), newline='') as f:
                    rows = list(csv.reader(f))
                self.assertEqual(rows[0][1:9], ['source_amp', 'M2.Z_offset',
                    'a_m2alph_x', 'b_m2alph_x', 'a_m2alph_y', 'b_m2alph_y',
                    'TILT_H', 'TILT_V'])
                self.assertEqual(rows[0][1:], [row[0] for row in rows[1:]])
                matrix = np.array([row[1:] for row in rows[1:]], dtype=float)
                np.testing.assert_allclose(matrix, expected, rtol=1e-10, atol=1e-15)
                for idx, name in enumerate(rows[0][1:]):
                    self.assertAlmostEqual(variances[name], expected[idx, idx])
                self.assertTrue(filename.with_suffix('.png').read_bytes().startswith(b'\x89PNG'))
                self.assertEqual(plt.get_fignums(), [])
                # Mirror all results keys, including generic higher-order modes.
                with contextlib.redirect_stdout(io.StringIO()), patch.object(plt, 'savefig'):
                    fitter.save_results(str(Path(directory) / 'fit.json'))
                self.assertEqual(list(variances), list(fitter.resultsdict))
                self.assertAlmostEqual(variances['M2.Z_target'], expected[1, 1])
                self.assertAlmostEqual(variances['M2.Z_move'], expected[1, 1])
                self.assertEqual(variances['M2.Z_middle'], 0.)
                x = result.x
                gradient = np.zeros(x.size)
                gradient[1:4] = [-x[3], 1., -x[1]]
                self.assertAlmostEqual(variances['M2.alphX_target'], gradient @ expected @ gradient)
                z = fitter.tmpbeamclass.m2z_vals['map1']
                self.assertAlmostEqual(variances['M2.alphX_map1'],
                    expected[2, 2] + 2 * z * expected[2, 3] + z**2 * expected[3, 3])
                middle = list(fitter.tmpbeamclass.trunc_maps)[2]
                z = fitter.tmpbeamclass.m2z_vals[middle]
                self.assertAlmostEqual(variances['strehl_ratio'],
                    (2e6 * (z + x[1]))**2 * expected[1, 1], places=6)

    def test_other_fitters_covariance_reporting_and_propagation(self):
        import csv
        for fitter_class in (OFFSETS, POINTING):
            for count, zernike_count in ((3, 9), (4, 16)):
                with self.subTest(fitter=fitter_class.__name__, count=count,
                                  zernike_count=zernike_count), tempfile.TemporaryDirectory() as directory:
                    fitter = self.make_fitter(zernike_count, fitter_class, count)
                    x, jac, residuals = fitter.results.x, fitter.results.jac, fitter.results.fun
                    expected = np.linalg.inv(jac.T @ jac) * (residuals @ residuals) / (residuals.size - x.size)
                    filename = Path(directory) / 'uncertainties'
                    fitter.save_uncertainties(filename)
                    np.testing.assert_allclose(fitter.covariance_matrix, expected, rtol=1e-10, atol=1e-15)
                    variances = json.loads(filename.with_suffix('.json').read_text())
                    with open(filename.with_suffix('.csv'), newline='') as f:
                        rows = list(csv.reader(f))
                    self.assertEqual(rows[0][1:], [row[0] for row in rows[1:]])
                    np.testing.assert_allclose(np.array([row[1:] for row in rows[1:]], float), expected)
                    for idx, name in enumerate(rows[0][1:]):
                        self.assertAlmostEqual(variances[name], expected[idx, idx])
                    self.assertTrue(filename.with_suffix('.png').read_bytes().startswith(b'\x89PNG'))
                    with contextlib.redirect_stdout(io.StringIO()), patch.object(plt, 'savefig'):
                        fitter.save_results(Path(directory) / 'fit.json')
                    self.assertEqual(list(variances), list(fitter.resultsdict))
                    self.assertEqual(variances['M2.Z_middle'], 0.)
                    self.assertAlmostEqual(variances['M2.Z_target'], expected[1, 1])
                    self.assertAlmostEqual(variances['M2.Z_move'], expected[1, 1])
                    middle = list(fitter.tmpbeamclass.trunc_maps)[count // 2]
                    z = fitter.tmpbeamclass.m2z_vals[middle]
                    np.testing.assert_allclose(variances['strehl_ratio'],
                        (2e6 * (z + x[1]))**2 * expected[1, 1], rtol=1e-9)
                    if fitter_class is OFFSETS:
                        gradient = np.zeros(x.size)
                        gradient[1:4] = [-x[3], 1., -x[1]]
                        self.assertAlmostEqual(variances['M2.X_target'], gradient @ expected @ gradient)
                        self.assertAlmostEqual(variances[f'M2.X_{middle}'],
                            expected[2, 2] + 2 * z * expected[2, 3] + z**2 * expected[3, 3])
                        np.testing.assert_allclose(fitter.tmpbeamclass.calls[-1]['del_x'], x[2] + z * x[3])
                        self.assertEqual(fitter.tmpbeamclass.calls[-1]['del_alph_x'], 0.)
                    else:
                        names = list(fitter.tmpbeamclass.trunc_maps)
                        self.assertEqual(rows[0][3:7], [f'Az_Off_{names[0]}', f'El_Off_{names[0]}',
                                                       f'Az_Off_{names[1]}', f'El_Off_{names[1]}'])
                        for idx, name in enumerate(names):
                            self.assertEqual(fitter.resultsdict[f'Az_Off_{name}'], x[2 + 2 * idx])
                            self.assertEqual(fitter.resultsdict[f'El_Off_{name}'], x[3 + 2 * idx])
                        self.assertIn('COMA_H', variances)
                        self.assertIn('COMA_V', variances)
                    self.assertEqual(plt.get_fignums(), [])

    def test_strehl_variance_includes_model_specific_aberrations(self):
        for fitter_class in (TILTS, OFFSETS, POINTING):
            with self.subTest(fitter=fitter_class.__name__), tempfile.TemporaryDirectory() as directory:
                fitter = self.make_fitter(16, fitter_class)
                beam = fitter.tmpbeamclass
                x = fitter.results.x
                z = beam.m2z_vals[list(beam.trunc_maps)[2]]
                weights = np.arange(1., 17.)

                def psf(**kwargs):
                    peak = (1e6 * kwargs['secondary_offset']**2
                            + 2 * kwargs['del_x'] + 3 * kwargs['del_y']
                            + 5 * kwargs['del_alph_x'] + 7 * kwargs['del_alph_y']
                            + weights @ kwargs['c'])
                    return np.full((3, 3), peak)

                beam.make_psf = psf
                expected_gradient = np.zeros(x.size)
                expected_gradient[1] = 2e6 * (z + x[1])
                if fitter_class is POINTING:
                    expected_gradient[fitter.tilt_offset_end_index:] = weights[[3] + list(range(5, 16))]
                else:
                    a, b = (5., 7.) if fitter_class is TILTS else (2., 3.)
                    expected_gradient[2:6] = [a, a * z, b, b * z]
                    expected_gradient[8:] = weights[[3, 5, 6] + list(range(9, 16))]
                fitter.save_uncertainties(Path(directory) / 'uncertainties')
                expected = expected_gradient @ fitter.covariance_matrix @ expected_gradient
                np.testing.assert_allclose(fitter.uncertaintiesdict['strehl_ratio'], expected, rtol=1e-8)

    def test_common_reporting_methods_and_zernike_reconstruction(self):
        import inspect
        reference_methods = {name for name in dir(TILTS) if callable(getattr(TILTS, name))
                             and not name.startswith('_')}
        reporting = ('save_results', 'save_uncertainties', 'surface_plot',
                     'plot_fit_results', 'combine_result_pngs',
                     'save_zernike_dat', 'save_subref_dat')
        for fitter_class in (TILTS, OFFSETS, POINTING):
            with self.subTest(fitter=fitter_class.__name__):
                public_methods = {name for name in dir(fitter_class)
                                  if callable(getattr(fitter_class, name)) and not name.startswith('_')}
                self.assertEqual(public_methods, reference_methods)
                for name in reporting:
                    self.assertEqual(inspect.signature(getattr(fitter_class, name)),
                                     inspect.signature(getattr(TILTS, name)))
                fitter = self.make_fitter(16, fitter_class)
                fitter.chisquared(fitter.results.x)
                names = list(fitter.tmpbeamclass.trunc_maps)
                middle = len(names) // 2
                expected = fitter.tmpbeamclass.calls[middle]['c']
                np.testing.assert_allclose(fitter._fitted_zernikes(), expected)
                without_pointing = expected.copy()
                without_pointing[1:3] = 0.
                np.testing.assert_allclose(fitter._fitted_zernikes(include_tilt=False), without_pointing)

    def test_invalid_uncertainty_estimates_rejected(self):
        for fitter_class in (TILTS, OFFSETS, POINTING):
            with tempfile.TemporaryDirectory() as directory:
                filename = Path(directory) / 'uncertainties'
                fitter = self.make_fitter(fitter_class=fitter_class)
                del fitter.results
                with self.assertRaisesRegex(ValueError, 'Run the fitter'):
                    fitter.save_uncertainties(filename)
                for kind in ('zero_column', 'dependent_columns', 'too_few_residuals', 'nonfinite'):
                    with self.subTest(fitter=fitter_class.__name__, kind=kind):
                        fitter = self.make_fitter(fitter_class=fitter_class)
                        if kind == 'zero_column':
                            fitter.results.jac[:, 0] = 0.
                        elif kind == 'dependent_columns':
                            fitter.results.jac[:, 0] = fitter.results.jac[:, 1]
                        elif kind == 'too_few_residuals':
                            size = fitter.results.x.size
                            fitter.results.jac = np.eye(size)
                            fitter.results.fun = np.ones(size)
                        else:
                            fitter.results.jac[0, 0] = np.nan
                        with self.assertRaises(ValueError):
                            fitter.save_uncertainties(filename)
                self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
