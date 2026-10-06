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

from simbeam import SimBeam

import time


''' 

This script contains example code to simulate an aberrated beam.
It is somewhat limited in that it only explicitly calls out 
zernike terms up to n=m=4. 
If you want to simulate higher-order terms you will need to accordingly modify the higher indices
in the c_tmp_microns vector.
The code follows the OSA/ANSI index convention so this should be a straightforward modification.

'''

results_dir = f'simulation_results/'
os.makedirs(results_dir, exist_ok=True)

test_freqs = np.array([150.0]) # in GHz. An array. If ordered multiple values are passed will integrate over the uniform bandpass
test_wavelengths = 300.E-3/test_freqs

pixel_size_deg = 2.0/3600. # in degrees
map_size_deg = 10./60. # in degrees

simbeam1 = SimBeam(test_wavelengths,pixel_size_deg,map_size_deg,bandpass = None)

simbeam1.initialize_model(
						include_legs=True,plot_aperture=False,save_aperture=None,
						aperture_fwhm = 45.,edge_taper_diameter=45.,plot_illumination=False,
						n=4,m=4)


c_microns = np.zeros(simbeam1.zernike_polynomials.shape[0])
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

M2z_offset = 0.0 # meters



tmppsf1 = simbeam1.make_psf(c=c_microns,
	                        secondary_offset=M2z_offset,
		                    del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
				            f=17.5,F=525.,D=50.)

# tmppsf is a pixell enmap object that contains data and wcs information

corners_tmp = np.rad2deg(enmap.corners(tmppsf1.shape,tmppsf1.wcs))
imextent_tmp = [corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]]
fig=plt.figure(figsize=(8,8))
im = plt.imshow(tmppsf1,extent=imextent_tmp,origin='lower',vmin=0.,vmax=1.)
# im = plt.imshow(10*np.log10(tmppsf1),extent=imextent_tmp,origin='lower',vmin=-30,vmax=0)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('Az Offset (deg)')
plt.ylabel('El Offset (deg)')
boxsize = 1.5/60.
plt.xlim([boxsize/2.,-boxsize/2.])
plt.ylim([-boxsize/2.,boxsize/2.])
cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(label='Normalized Amplitude',rotation=270,labelpad=15,fontsize=14) 
plt.savefig(results_dir+'simulated_psf.png',bbox_inches='tight')
plt.show()

# save the map to fits using pixell

enmap.write_map(results_dir+'SimulatedBeam.fits',tmppsf1)






