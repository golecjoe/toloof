import glob
import sys
import os
import numpy as np
from pixell import enmap, enplot, reproject, utils, curvedsky,wcsutils
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pathlib import Path

# tmp_toloof_path = os.path.expanduser("../toloof_work/toloof/toloof_v2")
# print('Adding '+tmp_toloof_path+ ' to PATH')
# sys.path.insert(0, tmp_toloof_path)

tmp_toloof_path = os.path.expanduser("../../toloof")
print('Adding '+tmp_toloof_path+ ' to PATH')
sys.path.insert(0, tmp_toloof_path)

from beamclass import Beam
from fitbeam import *
from scipy.optimize import minimize

import time
import json


# define a path to the data
# this assumes that the fits files are organized similarly to citlali outputs

data_path = Path('data/')

# define appropriate variables



tunenum = 157246

obsnum1 = 157247
obsnum2 = 157248
obsnum3 = 157249


tmpn = 4
tmpm = 4

# fitmodel can be: 
# 1. fit_beam_with_pointing_tilt_offsets_leastsquares
# 2. fit_beam_with_glob_tilt_M2offsets_leastsquares
# 3. fit_beam_with_glob_tilt_M2tilts_leastsquares

fitmodel = fit_beam_with_glob_tilt_M2offsets_leastsquares

baseresultsdir = f'test_multiband_M2off_results_{tunenum}/'
os.makedirs(baseresultsdir, exist_ok=True)


print(f'RUNNING FIT: {tunenum} with model {fitmodel.__name__}')

obsnums = [obsnum1, obsnum2, obsnum3] 

# first fit the a2000 data

fit_array = 'a2000'

results_dir = baseresultsdir + f'results_{fit_array}/'
os.makedirs(results_dir, exist_ok=True)

test_freqs = np.array([150.])
test_wavelengths = 300.E-3/test_freqs

map_file_paths = []


for i in obsnums:
	map_file_paths.append(data_path / f'{i}/raw/toltec_commissioning_{fit_array}_pointing_{i}_citlali.fits')

t0 = time.time()

# if fit_array =='a2000':
#     test_freqs = np.array([150.])
#     test_wavelengths = 300.E-3/test_freqs
# elif fit_array =='a1400':
#     test_freqs = np.array([220.])
#     test_wavelengths = 300.E-3/test_freqs
# elif fit_array =='a1100':
#     test_freqs = np.array([280.])
#     test_wavelengths = 300.E-3/test_freqs


beamclass = Beam(map_file_paths,test_wavelengths,bandpass=None,padpixels = 20,science_map_flag = False,mask_radius=0.75/60.)
beamclass.initialize_model(aperture_plane_resolution = 1.0,center_on_brightest_pix=False,
						  include_legs=True,plot_aperture=False,save_aperture=None,
						  aperture_fwhm = 45.,edge_taper_diameter=45.,plot_illumination=False,
						  n=tmpn,m=tmpm)

fitclass = fitmodel(beamclass)
fitclass.run_fitter(newx0=None)
fitclass.plot_fit_results(savefigname=results_dir+f'DataModelResidual_{tunenum}.png',vmin_val = -50.)
fitclass.save_results(results_dir+f'FitResults_{tunenum}.json')
fitclass.surface_plot(results_dir+f'SurfacePlot_{tunenum}.png')
fitclass.combine_result_pngs(outfile=results_dir+f"CombinedResults_{tunenum}.png",dpi=200)
fitclass.save_zernike_dat(results_dir+f'toloof_n{tmpn}m{tmpm}_{fit_array}_{tunenum}_zernike.dat')
fitclass.save_subref_dat(results_dir+f'toloof_n{tmpn}m{tmpm}_{fit_array}_{tunenum}_subref.dat')
fitclass.save_uncertainties(results_dir+f'uncertainties_{tunenum}')

a2000resx = fitclass.results.x


# fit the a1400 data

fit_array = 'a1400'

results_dir = baseresultsdir + f'results_{fit_array}/'
os.makedirs(results_dir, exist_ok=True)

test_freqs = np.array([220.])
test_wavelengths = 300.E-3/test_freqs

map_file_paths = []


for i in obsnums:
    map_file_paths.append(data_path / f'{i}/raw/toltec_commissioning_{fit_array}_pointing_{i}_citlali.fits')

beamclass = Beam(map_file_paths,test_wavelengths,bandpass=None,padpixels = 20,science_map_flag = False,mask_radius=0.75/60.)
beamclass.initialize_model(aperture_plane_resolution = 1.0,center_on_brightest_pix=False,
                          include_legs=True,plot_aperture=False,save_aperture=None,
                          aperture_fwhm = 45.,edge_taper_diameter=45.,plot_illumination=False,
                          n=tmpn,m=tmpm)

fitclass = fitmodel(beamclass)
fitclass.run_fitter(newx0=a2000resx)
fitclass.plot_fit_results(savefigname=results_dir+f'DataModelResidual_{tunenum}.png',vmin_val = -50.)
fitclass.save_results(results_dir+f'FitResults_{tunenum}.json')
fitclass.surface_plot(results_dir+f'SurfacePlot_{tunenum}.png')
fitclass.combine_result_pngs(outfile=results_dir+f"CombinedResults_{tunenum}.png",dpi=200)
fitclass.save_zernike_dat(results_dir+f'toloof_n{tmpn}m{tmpm}_{fit_array}_{tunenum}_zernike.dat')
fitclass.save_subref_dat(results_dir+f'toloof_n{tmpn}m{tmpm}_{fit_array}_{tunenum}_subref.dat')
fitclass.save_uncertainties(results_dir+f'uncertainties_{tunenum}')

a1400resx = fitclass.results.x

# fit the a1100 data

fit_array = 'a1100'

results_dir = baseresultsdir + f'results_{fit_array}/'
os.makedirs(results_dir, exist_ok=True)

test_freqs = np.array([280.])
test_wavelengths = 300.E-3/test_freqs

map_file_paths = []


for i in obsnums:
    map_file_paths.append(data_path / f'{i}/raw/toltec_commissioning_{fit_array}_pointing_{i}_citlali.fits')

beamclass = Beam(map_file_paths,test_wavelengths,bandpass=None,padpixels = 20,science_map_flag = False,mask_radius=0.75/60.)
beamclass.initialize_model(aperture_plane_resolution = 1.0,center_on_brightest_pix=False,
                          include_legs=True,plot_aperture=False,save_aperture=None,
                          aperture_fwhm = 45.,edge_taper_diameter=45.,plot_illumination=False,
                          n=tmpn,m=tmpm)

fitclass = fitmodel(beamclass)
fitclass.run_fitter(newx0=a1400resx)
fitclass.plot_fit_results(savefigname=results_dir+f'DataModelResidual_{tunenum}.png',vmin_val = -50.)
fitclass.save_results(results_dir+f'FitResults_{tunenum}.json')
fitclass.surface_plot(results_dir+f'SurfacePlot_{tunenum}.png')
fitclass.combine_result_pngs(outfile=results_dir+f"CombinedResults_{tunenum}.png",dpi=200)
fitclass.save_zernike_dat(results_dir+f'toloof_n{tmpn}m{tmpm}_{fit_array}_{tunenum}_zernike.dat')
fitclass.save_subref_dat(results_dir+f'toloof_n{tmpn}m{tmpm}_{fit_array}_{tunenum}_subref.dat')
fitclass.save_uncertainties(results_dir+f'uncertainties_{tunenum}')

a1100resx = fitclass.results.x


print('FINISHED FIT. Time = ',time.time()-t0)







