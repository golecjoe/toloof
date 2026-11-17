import glob
import os
import numpy as np
from pixell import enmap, enplot, reproject, utils, curvedsky,wcsutils
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import sys

tmp_toloof_path = os.path.expanduser("../../toloof")
print('Adding '+tmp_toloof_path+ ' to PATH')
sys.path.insert(0, tmp_toloof_path)

from simbeam import SimBeam,make_tolteca_simu_input_fits

import time


''' 

This script contains example code to simulate an aberrated beam.
It is somewhat limited in that it only explicitly calls out 
zernike terms up to n=m=4. 
If you want to simulate higher-order terms you will need to accordingly modify the higher indices
in the c_tmp_microns vector.
The code follows the OSA/ANSI index convention so this should be a straightforward modification.

'''

results_dir = f'03_inputs/'
os.makedirs(results_dir, exist_ok=True)

pixel_size_deg = 2.0/3600. # in degrees
map_size_deg = 10./60. # in degrees

simbeam_a2000 = SimBeam(300.E-3/np.array([150.]),pixel_size_deg,map_size_deg,bandpass = None)
simbeam_a1400 = SimBeam(300.E-3/np.array([220.]),pixel_size_deg,map_size_deg,bandpass = None)
simbeam_a1100 = SimBeam(300.E-3/np.array([280.]),pixel_size_deg,map_size_deg,bandpass = None)

simbeam_a2000.initialize_model(include_legs=True,plot_aperture=False,save_aperture=None,
						  aperture_fwhm = 45.,edge_taper_diameter=45.,plot_illumination=False,
						  n=4,m=4)
simbeam_a1400.initialize_model(include_legs=True,plot_aperture=False,save_aperture=None,
						  aperture_fwhm = 45.,edge_taper_diameter=45.,plot_illumination=False,
						  n=4,m=4)
simbeam_a1100.initialize_model(include_legs=True,plot_aperture=False,save_aperture=None,
						  aperture_fwhm = 45.,edge_taper_diameter=45.,plot_illumination=False,
						  n=4,m=4)


c_microns = np.zeros(simbeam_a2000.zernike_polynomials.shape[0])

c_microns[0] = 0. #PISTON
c_microns[1] = 0. # Y-TILT
c_microns[2] = 0. # X-TILT
c_microns[3] = 0. # OBLIQUE ASTIGMATISM
c_microns[4] = 0. # DEFOCUS
c_microns[5] = 0. # VERTICAL ASTIGMATISM
c_microns[6] = 0. # VERTICAL TREFOIL
c_microns[7] = 0. # VERTICAL COMA
c_microns[8] = 0. # HORIZONTAL COMA
c_microns[9] = 0. # OBLIQUE TREFOIL
c_microns[10] = 0. # OBLIQUE QUADFOIL
c_microns[11] = 0. # OBLIQUE SECONDARY ASTIGMATISM
c_microns[12] = 0. # SPHERICAL ABERRATION
c_microns[13] = 0. # VERTICAL SECONDARY ASTIGMATISM
c_microns[14] = 0. # VERTICAL QUADFOIL


M2z_offset_array = np.array([-2,0,2])*1E-3

for count,m2z_offset in enumerate(M2z_offset_array):

	tmppsf_a2000 = simbeam_a2000.make_psf(c=c_microns,
		                        secondary_offset=m2z_offset,
			                    del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
					            f=17.5,F=525.,D=50.)
	tmppsf_a1400 = simbeam_a1400.make_psf(c=c_microns,
		                        secondary_offset=m2z_offset,
			                    del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
					            f=17.5,F=525.,D=50.)
	tmppsf_a1100 = simbeam_a1100.make_psf(c=c_microns,
		                        secondary_offset=m2z_offset,
			                    del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
					            f=17.5,F=525.,D=50.)

	make_tolteca_simu_input_fits(tmppsf_a2000,tmppsf_a1400,tmppsf_a1100,results_dir+f'test_im{count}.fits')







