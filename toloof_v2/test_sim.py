from simbeam import SimBeam
import glob
import os
import numpy as np
from pixell import enmap, enplot, reproject, utils, curvedsky,wcsutils
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import time


test_freqs = np.linspace(125,175,5)
test_wavelengths = 300.E-3/test_freqs

toltec_bandpass = np.load(os.path.expanduser('~/Documents/Work/TolTEC/toloof/model_passbands.npz'))

interpt_BP_f = interp1d(toltec_bandpass['f_GHz'],toltec_bandpass['band_150'])

tmp_BP = interpt_BP_f(test_freqs)

simbeam1 = SimBeam(test_wavelengths,2.0/3600.,10./60.,bandpass = tmp_BP)

simbeam1.initialize_model(
						include_legs=True,plot_aperture=False,save_aperture=None,
						aperture_fwhm = 48.,edge_taper_diameter=48.,plot_illumination=False,
						n=4,m=4)


c_tmp_microns = np.zeros(simbeam1.zernike_polynomials.shape[0])

c_tmp_microns[0] = 0. #PISTON
c_tmp_microns[1] = 0. # Y-TILT
c_tmp_microns[2] = 0. # X-TILT
c_tmp_microns[3] = 0. # OBLIQUE ASTIGMATISM
c_tmp_microns[4] = 0. # DEFOCUS
c_tmp_microns[5] = 200. # VERTICAL ASTIGMATISM
c_tmp_microns[6] = 0. # VERTICAL TREFOIL
c_tmp_microns[7] = 0. # VERTICAL COMA
c_tmp_microns[8] = 0. # HORIZONTAL COMA
c_tmp_microns[9] = 0. # OBLIQUE TREFOIL
c_tmp_microns[10] = 0. # OBLIQUE QUADFOIL
c_tmp_microns[11] = 0. # OBLIQUE SECONDARY ASTIGMATISM
c_tmp_microns[12] = 0. # SPHERICAL ABERRATION
c_tmp_microns[13] = 0. # VERTICAL SECONDARY ASTIGMATISM
c_tmp_microns[14] = 0. # VERTICAL QUADFOIL


c_tmp_phase = (c_tmp_microns*1E-6*2*np.pi)/(np.mean(test_wavelengths)*np.sqrt(2))


tmppsf1 = simbeam1.make_psf(c=c_tmp_phase,
	                        secondary_offset=1E-3,
		                    del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
				            f=17.5,F=525.,D=50.)



corners_tmp = np.rad2deg(enmap.corners(tmppsf1.shape,tmppsf1.wcs))
imextent_tmp = [corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]]
fig=plt.figure(figsize=(8,8))
# im = plt.imshow(tmppsf1,extent=imextent_tmp,origin='lower',vmin=0,vmax=0.1)
im = plt.imshow(10*np.log10(tmppsf1),extent=imextent_tmp,origin='lower',vmin=-30,vmax=0)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel('Right Ascension (deg)')
plt.ylabel('Declination (deg)')

boxsize = 1.5/60.

plt.xlim([boxsize/2.,-boxsize/2.])
plt.ylim([-boxsize/2.,boxsize/2.])
cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.7])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label(label='mJy/beam',rotation=270,labelpad=15,fontsize=14)    


plt.show()