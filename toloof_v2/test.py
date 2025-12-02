from beamclass import Beam
import glob
import os
import numpy as np
from pixell import enmap, enplot, reproject, utils, curvedsky,wcsutils
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from fitbeam import fit_beam_with_pointing_offsets, plot_fit_results,save_results,fit_beam_with_M2_offsets
from scipy.optimize import minimize

import time

t1 = time.time()

test_freqs = np.array([150.]) #np.linspace(125,175,5)
test_wavelengths = 300.E-3/test_freqs

toltec_bandpass = np.load(os.path.expanduser('~/Documents/Work/TolTEC/toloof/model_passbands.npz'))

interpt_BP_f = interp1d(toltec_bandpass['f_GHz'],toltec_bandpass['band_150'])

tmp_BP = interpt_BP_f(test_freqs)


#test_freqs = np.linspace(120,170,10)

obsnums = [145084, 145085, 145086]#[145082, 145083, 145084, 145085, 145086, 145087, 145088]

map_file_paths = []
tel_file_paths = []

for i in obsnums:
	map_file_paths.append(f'~/Documents/Projects/OOF_Work/OOF_BasisFunctions_102925/subset_of_maps/{i}/redu00/{i}/raw/toltec_commissioning_a2000_pointing_{i}_citlali.fits')
	tmp_tel_path = glob.glob(os.path.expanduser(f'~/Documents/Projects/OOF_Work/OOF_BasisFunctions_102925/subset_of_maps/{i}/tel_toltec_*.nc'))
	tel_file_paths.append(tmp_tel_path[0])


tmpclass = Beam(map_file_paths,tel_file_paths,test_wavelengths,bandpass=None)

tmpclass.initialize_model(aperture_plane_resolution = 1.0,center_on_brightest_pix=False,
						  include_legs=True,plot_aperture=False,save_aperture=None,
						  aperture_fwhm = 45.,edge_taper_diameter=45.,plot_illumination=False,
						  n=4,m=4)


# tmpfitclass = fit_beam_with_pointing_offsets(tmpclass)

tmpfitclass = fit_beam_with_M2_offsets(tmpclass)
tmpfitclass.run_fitter()
tmpfitclass.plot_fit_results(savefigname='m2test_3image.png')
tmpfitclass.save_results('m2testresults_3image.json')

t2 = time.time()

print('Time it took = ',t2-t1, ' seconds')

# plot_fit_results(tmpclass,tmpfitclass,results,vmax_frac_of_source_flux = 0.2,
# 	             title='test title',savefigname='testfig_BP.png')

# save_results(results,tmpfitclass,'test_results_BP.json')

