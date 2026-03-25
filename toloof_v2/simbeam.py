import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap, enplot, reproject, utils, curvedsky,wcsutils
from scipy.ndimage import rotate
from astropy.coordinates import SkyCoord 
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.io import fits
from datetime import datetime, UTC
import warnings
from astropy.io.fits.verify import VerifyWarning

from map_io import CitlaliMaps, make_mask_enmap, make_coordinate_grids,build_tangent_wcs
from telfile_io import get_M2z_from_tel
from optics import gaussian, gen_zernike_polys,gen_defocus_cassegrain_telescope,gen_phase_error_secondary_lat_displacement,gen_phase_error_secondary_tilt
from diffraction import Fraunhofer, Convert_field_to_PSF


class SimBeam:
	def __init__(self,wavelengths,pixelsize,imagesize,bandpass=None):

		self.wavelengths = wavelengths
		self.pixelsize = pixelsize
		self.N = int(imagesize/pixelsize)
		self.delta_x = np.mean(wavelengths)/np.deg2rad(pixelsize*self.N)

		self.model_map_wcs = wcsutils.build((0,0),res=pixelsize, shape=[self.N,self.N], system='tan')

		if bandpass is None:
			self.bandpass = np.ones(wavelengths.size)
		else:
			self.bandpass = bandpass

	def set_LMT_aperture(self,include_legs=True,plot_aperture=False,save_aperture=None):
		"""
		Create a circular aperture mask representing the LMT primary mirror.

		Parameters
		----------
		include_legs : bool
			Whether to simulate support legs (quadrupod shadows).
		plot_aperture : bool
			If True, show a plot of the resulting aperture field.
		"""
		
		delta_x = self.delta_x
		#print('The Pixel Size in the Aperture Plane is ',delta_x, ' meters')
		L = self.N*delta_x

		###  make the coordinate grid
		x,y,r,phi = make_coordinate_grids(self.N,L)

		self.x = x
		self.y = y
		self.r = r
		self.phi = phi
		#figure out the legs better
		xbins = x[0,:]
		ybins = y[:,0]
		self.xbins = xbins
		self.ybins = ybins

		primary_diamter = 50.
		primaryaperture = np.ones(self.y.shape)
		primaryaperture[self.r>primary_diamter/2.] = 0.

		secondarydiameter = 2.*1.625 #2.5
		secondaryblockage = np.ones(self.y.shape)
		secondaryblockage[self.r<secondarydiameter/2.] = 0.

		# make the legs 

		yminval = np.amin(self.y[np.where(self.y>0)])
		smallestyindex = np.where(self.y==yminval)[0][0]
		xvals = self.x[smallestyindex,:]
		rvals = np.zeros(xvals.size)
		rvals[xvals>=0]=xvals[xvals>=0]

		flength = 17.5
		tanalpha = 1.27495
		AB = 3.3225
		W = 1.0
		spidershadow_y = (W/(2.*AB))*(rvals-(flength*tanalpha)+((rvals**2/(4.*flength))*tanalpha))
		spidershadow_y[spidershadow_y<(W/2.)]=W/2.
		spidershadow_y[xvals<0]=0
		spidershadow_y[xvals>25.]=0

		spider_template = np.ones(self.y.shape)

		for i in range(spidershadow_y.size):
			spider_template[np.where(np.abs(self.y[:,i])<spidershadow_y[i]),i] = 0

		# theta in degrees; positive = counterclockwise
		spider_leg1 = rotate(spider_template, angle=45., reshape=False,
						  order=0, mode='constant', cval=0)
		spider_leg1 = (spider_leg1 > 0.5).astype(spider_template.dtype)  # keep it binary

		spider_leg2 = rotate(spider_template, angle=135., reshape=False,
						  order=0, mode='constant', cval=0)
		spider_leg2 = (spider_leg2 > 0.5).astype(spider_template.dtype)  # keep it binary

		spider_leg3 = rotate(spider_template, angle=225., reshape=False,
						  order=0, mode='constant', cval=0)
		spider_leg3 = (spider_leg3 > 0.5).astype(spider_template.dtype)  # keep it binary

		spider_leg4 = rotate(spider_template, angle=315, reshape=False,
						  order=0, mode='constant', cval=0)
		spider_leg4 = (spider_leg4 > 0.5).astype(spider_template.dtype)  # keep it binary

		spider_legs_total = spider_leg1*spider_leg2*spider_leg3*spider_leg4

		A = primaryaperture*secondaryblockage
		
		if include_legs:
			A = A*spider_legs_total



		
		self.Aperture = A
		self.L = L
		
		## plot the aperature fields
		if plot_aperture:
			plt.figure()
			plt.imshow(A,extent=([-L/2, L/2, -L/2, L/2]))
			plt.title("Aperature field pattern")
			plt.xlabel("x [m]")
			plt.ylabel("y [m]")
			plt.colorbar()
			if save_aperture is not None:
				plt.savefig(save_aperture,bbox_inches='tight')
			plt.close()


	def set_illumination(self,aperture_fwhm = 48.,edge_taper_diameter=48.,plot_illumination=False):
		"""
		Define a radial Gaussian illumination function over the aperture.

		Parameters
		----------
		aperture_fwhm : float
			Full-width at half-maximum of the Gaussian illumination (in meters).
		edge_taper_diameter : float
			Diameter beyond which the illumination is zeroed.
		plot_illumination : bool
			If True, show a plot of the illumination pattern.
		"""

		sig0 = aperture_fwhm/(2*np.sqrt(2*np.log(2)))
		edge_taper = np.ones([self.N,self.N])
		edge_taper[np.where(self.r>edge_taper_diameter/2.)]=0
		illumination = gaussian(self.r,sig0)*edge_taper
		self.illumination = illumination
		if plot_illumination:
			plt.figure()
			plt.imshow(self.illumination,extent=([-self.L/2, self.L/2, -self.L/2, self.L/2]))
			plt.title("illumination")
			plt.xlabel("x [m]")
			plt.ylabel("y [m]")
			plt.colorbar()
			plt.show()

	def get_zernike_polynomials(self,n,m):
		"""
		Generate Zernike polynomials over the normalized aperture.

		Parameters
		----------
		n : int
			Maximum radial order.
		m : int
			Maximum azimuthal index.
		"""
		diam_primary = 50. 
		self.zernike_polynomials = gen_zernike_polys(n,m,self.r/(diam_primary/2.),self.phi)

	def make_phase(self,wavelength,c=None,secondary_offset=0.,del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
				  f=17.5,F=525.,D=50.):

		"""
		Apply a phase screen composed of Zernike coefficients and Cassegrain defocus.

		Parameters
		----------
		secondary_offset : float
			Longitudinal offset of secondary mirror in meters.
		c : ndarray or None
			Array of Zernike coefficients. If None, uses all zeros.
		plot_phase : bool
			If True, display the resulting phase map.
		"""
		
		
		if c is None:
			c = np.zeros(self.zernike_polynomials.shape[0])
		c[0] = 0
		
		Phi = np.zeros([self.zernike_polynomials.shape[1],self.zernike_polynomials.shape[2]])
		for i in range(c.size):
			Phi+=c[i]*self.zernike_polynomials[i,:,:]
		delta_phase = gen_defocus_cassegrain_telescope(self.r,secondary_offset,f=f,F=F,D=D)
		delta_phase2 = gen_phase_error_secondary_lat_displacement(self.x,self.y,del_x,del_y,f=f,F=F,D=D)
		delta_phase3 = gen_phase_error_secondary_tilt(self.x,self.y,del_alph_x,del_alph_y,f=f,F=F,c_minus_a=0.8548,D=D)

		#tmpphase = Phi+((2.*np.pi)*delta_phase/self.wavelength)+((2.*np.pi)*delta_phase2/self.wavelength)+((2.*np.pi)*delta_phase3/self.wavelength)

		# tmpphase = Phi+((2.*np.pi)*delta_phase/np.mean(self.wavelengths))+((2.*np.pi)*delta_phase2/np.mean(self.wavelengths))+((2.*np.pi)*delta_phase3/np.mean(self.wavelengths))
		tmpphase = Phi+((2.*np.pi)*delta_phase/wavelength)+((2.*np.pi)*delta_phase2/wavelength)+((2.*np.pi)*delta_phase3/wavelength)

		return tmpphase

	def make_psf_monochromatic(self,wavelength,c=None,secondary_offset=0.,
							   del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.):
		"""
		Construct the modeled PSF using the current aperture, illumination, and phase.
		Stores a normalized enmap PSF in `self.PSF`.
		"""
		if c is None:
			c = np.zeros(self.zernike_polynomials.shape[0])

		
		phase = self.make_phase(wavelength,c=c,secondary_offset=secondary_offset,
								del_x=del_x,del_y=del_y,del_alph_x=del_alph_x,del_alph_y=del_alph_y,
								f=f,F=F,D=D)

		A_complex = self.Aperture*self.illumination*np.exp(phase*1j)
		farfield_im_size, U = Fraunhofer(A_complex,wavelength,self.delta_x)
		farfield_pix_size = farfield_im_size/U.shape[0]

		PSF = Convert_field_to_PSF(U)
		tmpnewwcs = build_tangent_wcs(U.shape[0], U.shape[1], farfield_pix_size)

		PSF = enmap.enmap(PSF,tmpnewwcs)

		PSF_proj = enmap.project(PSF,(self.N,self.N),self.model_map_wcs)

		return PSF_proj

	def make_psf(self,c=None,secondary_offset=0.,
							   del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.):
		"""
		Construct the modeled PSF using the current aperture, illumination, and phase.
		Stores a normalized enmap PSF in `self.PSF`.
		"""

		self.m2pos = np.array([del_x,del_y,secondary_offset])

		if c is None:
			c = np.zeros(self.zernike_polynomials.shape[0])

		

		if self.wavelengths.size==1:
			tmppsf = self.make_psf_monochromatic(self.wavelengths[0],c=c,secondary_offset=secondary_offset,
							   del_x=del_x,del_y=del_y,del_alph_x=del_alph_x,del_alph_y=del_alph_y,
							   f=f,F=F,D=D)
			#self.M2pos = np.array([del_x,del_y,secondary_offset])
			self.PSF = tmppsf/self.norm_amplitude
			return tmppsf/self.norm_amplitude

		else:
			delta_wavelengths = np.diff(self.wavelengths)
			total_psf = enmap.zeros((self.N,self.N),self.model_map_wcs)
			BP_denom = 0

			for i in range(len(self.wavelengths)-1):
				wavelength = self.wavelengths[i]
				phase = self.make_phase(wavelength,c=c,secondary_offset=secondary_offset,
									del_x=del_x,del_y=del_y,del_alph_x=del_alph_x,del_alph_y=del_alph_y,
									f=f,F=F,D=D)
				psf_i = self.make_psf_monochromatic(wavelength,c=c,secondary_offset=secondary_offset,
							   del_x=del_x,del_y=del_y,del_alph_x=del_alph_x,del_alph_y=del_alph_y,
							   f=f,F=F,D=D)
				total_psf += delta_wavelengths[i]*self.bandpass[i]*psf_i
				BP_denom += delta_wavelengths[i]*self.bandpass[i]

			PSF = total_psf/BP_denom

			self.PSF = PSF/self.norm_amplitude
			
			return PSF/self.norm_amplitude

	def make_normalizing_amplitude(self):
		A_complex = self.Aperture*self.illumination

		phase = np.zeros(A_complex.shape)


		if self.wavelengths.size==1:
			self.norm_amplitude = np.amax(self.make_psf_monochromatic(self.wavelengths[0],c=None,secondary_offset=0.,
							   del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.))

		else:
			delta_wavelengths = np.diff(self.wavelengths)
			total_psf = enmap.zeros((self.N,self.N),self.model_map_wcs)
			BP_denom = 0

			for i in range(len(self.wavelengths)-1):
				wavelength = self.wavelengths[i]
				psf_i = self.make_psf_monochromatic(wavelength,c=None,secondary_offset=0.,
							   del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.)

				total_psf += delta_wavelengths[i]*self.bandpass[i]*psf_i
				BP_denom += delta_wavelengths[i]*self.bandpass[i]

			PSF_BP = total_psf/BP_denom
			self.norm_amplitude = np.amax(PSF_BP)

	def initialize_model(self,
						include_legs=True,plot_aperture=False,save_aperture=None,
						aperture_fwhm = 48.,edge_taper_diameter=48.,plot_illumination=False,
						n=4,m=4):

		self.set_LMT_aperture(include_legs=include_legs,plot_aperture=plot_aperture,save_aperture=save_aperture)
		self.set_illumination(aperture_fwhm = aperture_fwhm,edge_taper_diameter=edge_taper_diameter,plot_illumination=plot_illumination)
		self.get_zernike_polynomials(n,m)
		self.make_normalizing_amplitude()


	def make_citlali_fits(self,obsnum, array,fname):

		primary_hdu = fits.PrimaryHDU(header=make_primary_header(obsnum,array,self.m2pos*1E6))


		# get psf array in citlali fits format
		signal_hdu = make_image_hdu(self.PSF,array,'signal_I')
		weights_hdu = make_image_hdu(self.PSF*0.,array,'weight_I')
		coverage_hdu = make_image_hdu(self.PSF*0.,array,'coverage_I')
		covbool_hdu = make_image_hdu(self.PSF*0.,array,'coverage_bool_I')
		sig2noise_hdu = make_image_hdu(self.PSF*0.,array,'sig2noise_I')
		hdul = fits.HDUList([primary_hdu, signal_hdu, weights_hdu,coverage_hdu,covbool_hdu,sig2noise_hdu])
		with warnings.catch_warnings():
			warnings.filterwarnings(
				"ignore",
				category=VerifyWarning,
				message=r".*HIERARCH card will be created.*",
			)
			warnings.filterwarnings(
			"ignore",
			category=VerifyWarning,
			message=r".*Card is too long, comment will be truncated.*",
			)
			hdul.writeto(fname, overwrite=True)







def make_primary_header(obsnum, array, M2pos,
						mean_az = 0., mean_el = 0.,
						oof_rms = 0.):

	if array=='a2000':
		bmaj = 10.
		bmin = 10.
		oof_w = 0.002
		inst_id = 2000
		meantaustr = 'mean tau (a2000) '


	elif array=='a1400':
		bmaj = 6.
		bmin = 6.
		oof_w = 0.0014
		inst_id = 1400
		meantaustr = 'mean tau (a1400) '


	elif array=='a1100':
		bmaj = 5.
		bmin = 5.
		oof_w = 0.0011
		inst_id = 1100
		meantaustr = 'mean tau (a1100) '



	hdu = fits.PrimaryHDU()
	# Access the header
	hdr = hdu.header
	now = datetime.now(UTC)
	timestamp = now.strftime("%Y-%m-%dT%H:%M:%S")

	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore",
			category=VerifyWarning,
			message=r".*HIERARCH card will be created.*",
		)
		warnings.filterwarnings(
		"ignore",
		category=VerifyWarning,
		message=r".*Card is too long, comment will be truncated.*",
		)

		hdr['DATE'] = (timestamp,'file creation date (YYYY-MM-DDThh:mm:ss UT)') 
		hdr['TO_MJY/BEAM'] = (      1,'Conversion to mJy/beam')
		hdr['TO_MJY/SR'] = (0.294737357402983, 'Conversion to MJy/sr')
		hdr['TO_UK']   = (6.84424356664095E-06, 'Conversion to uK') 
		hdr['TO_JY/PIXEL'] = (2.77105349774227E-05, 'Conversion to Jy/pixel')
		hdr['OBSNUM0'] = (f"{int(obsnum):06d}",'Observation Number 0')
		hdr['DATEOBS0'] = (timestamp,'Date and time of observation 0') 
		hdr['SOURCE']  = ('FAUX','Source name')
		hdr['INSTRUME'] = ('TolTEC  ','Instrument')
		hdr['HWPR']    =   (False, 'HWPR installed')
		hdr['TELESCOP'] = ('LMT     ', 'Telescope')
		hdr['WAV'] = (array, 'Wavelength')
		hdr['PIPELINE'] = ('CITLALI ', 'Redu pipeline')
		hdr['VERSION'] = ('v4.0.0-62-ge0090e2d', 'CITLALI_GIT_VERSION')
		hdr['KIDS'] = ('4a3428e ', 'KIDSCPP_GIT_VERSION')
		hdr['TULA'] = ('28875ba ', 'TULA_GIT_VERSION')
		hdr['PROJID'] = ('Tol-SIM', 'Project ID')
		hdr['GOAL'] = ('pointing', 'Reduction type')
		hdr['OBSGOAL'] = ('Oof', 'Obs goal')
		hdr['TYPE'] = ('xs      ', 'TOD Type')
		hdr['GROUPING'] = ('array   ', 'Map grouping')
		hdr['METHOD'] = ('jinc    ', 'Map method')
		hdr['EXPTIME'] = (63.8894081115723, 'Exposure time (sec)')
		hdr['RADESYS'] = ('altaz   ', 'Coord Reference Frame')
		hdr['SRC_RA'] = (0., 'Source RA (radians)')
		hdr['SRC_DEC'] = (0., 'Source Dec (radians)')
		hdr['TAN_RA'] = (0., 'Map Tangent Point RA (radians)')
		hdr['TAN_DEC'] = (0., 'Map Tangent Point Dec (radians)')
		hdr['MEAN_EL'] = (mean_el, 'Mean Elevation (deg)')
		hdr['MEAN_AZ'] = (mean_az, 'Mean Azimuth (deg)')
		hdr['MEAN_PA'] = (0., 'Mean Parallactic angle (deg)')
		hdr['BMAJ'] = (bmaj, 'beammaj (arcsec)')
		hdr['BMIN'] = (bmin, 'beammin (arcsec)')
		hdr['BPA'] = (0., 'beampa (deg)')
		hdr['BUNIT'] = ('mJy/beam', 'bunit')
		hdr['JINC_R'] = (1.5, 'Jinc filter R_max')
		hdr['JINC_A'] = (1.1, 'Jinc filter param a')
		hdr['JINC_B'] = (3.17, 'Jinc filter param b')
		hdr['JINC_C'] = (2.0, 'Jinc filter param c')
		hdr['MEAN_TAU'] = (0.0, 'mean tau (a2000)')
		hdr['SAMPRATE'] = (122.0703125, 'sample rate (Hz)')
		hdr['APT'] = ('apt_152393_matched.ecsv', 'APT table used')
		hdr['OOF_RMS'] = (oof_rms, 'rms of map background (mJy/beam)')
		hdr['OOF_W'] = (oof_w, 'wavelength (m)')
		hdr['OOF_ID'] = (inst_id, 'instrument id')
		hdr['OOF_T'] = (3.0, 'taper (dB)')
		hdr['OOF_M2X'] = (M2pos[0], 'oof m2x (microns)')
		hdr['OOF_M2Y'] = (M2pos[1], 'oof m2y (microns)')
		hdr['OOF_M2Z'] = (M2pos[2], 'oof m2z (microns)')
		hdr['OOF_RO'] = (25.0, 'outer diameter of the antenna (m)')
		hdr['OOF_RI'] = (1.65, 'inner diameter of the antenna (m)')
		hdr['FRUITLOOPS_ITER'] = (9, 'Current fruit loops iteration')
		hdr['CONFIG.VERBOSE'] = (False, 'Reduced in verbose mode')
		hdr['CONFIG.POLARIZED'] = (False, 'Polarized Obs')
		hdr['CONFIG.DESPIKED'] = (False, 'Despiked')
		hdr['CONFIG.TODFILTERED'] = (True, 'TOD Filtered')
		hdr['CONFIG.DOWNSAMPLED'] = (True, 'Downsampled')
		hdr['CONFIG.CALIBRATED'] = (True, 'Calibrated')
		hdr['CONFIG.EXTINCTION'] = (False, 'Extinction corrected')
		hdr['CONFIG.EXTINCTION.EXTMODEL'] = ('N/A     ', 'Extinction model')
		hdr['CONFIG.WEIGHT.TYPE'] = ('approximate', 'Weighting scheme')
		hdr['CONFIG.INV_VAR.RTC.WTLOW'] = (0.0, 'RTC lower inv var cutoff')
		hdr['CONFIG.INV_VAR.RTC.WTHIGH'] = (10.0, 'RTC upper inv var cutoff')
		hdr['CONFIG.INV_VAR.PTC.WTLOW'] = (0.0, 'PTC lower inv var cutoff')
		hdr['CONFIG.INV_VAR.PTC.WTHIGH'] = (10.0, 'PTC upper inv var cutoff')
		hdr['CONFIG.WEIGHT.PTC.WTLOW'] = (0.0, 'PTC lower weight cutoff')
		hdr['CONFIG.WEIGHT.PTC.WTHIGH'] = (11.0, 'PTC upper weight cutoff')
		hdr['CONFIG.WEIGHT.MEDWTFACTOR'] = (10.0, 'Median weight factor')
		hdr['CONFIG.CLEANED'] = (True, 'Cleaned')
		hdr['CONFIG.CLEANED.NEIG'] = (5, 'Number of eigenvalues removed')
		hdr['CONFIG.FRUITLOOPS'] = (True, 'Fruit loops')
		hdr['CONFIG.FRUITLOOPS.PATH'] = ('null    ', 'Fruit loops path')
		hdr['CONFIG.FRUITLOOPS.TYPE'] = ('obsnum/raw', 'Fruit loops type')
		hdr['CONFIG.FRUITLOOPS.S2N'] = (100.0, 'Fruit loops S/N')
		hdr['CONFIG.FRUITLOOPS.FLUX'] = (10.0, 'Fruit loops flux (mJy/beam)')
		hdr['CONFIG.FRUITLOOPS.MAXITER'] = (10, 'Fruit loops iterations')
		hdr['HEADER.DCS.CALMODE'] = (2.0, 'Header.Dcs.CalMode')
		hdr['HEADER.DCS.INTEGRATIONTIME'] = (60.0800000877352, 'Header.Dcs.IntegrationTime')
		hdr['HEADER.DCS.OBSMODE'] = (0.0, 'Header.Dcs.ObsMode')
		hdr['HEADER.DCS.OBSNUM'] = (152393.0, 'Header.Dcs.ObsNum')
		hdr['HEADER.DCS.OBSTYPE'] = (0.0, 'Header.Dcs.ObsType')
		hdr['HEADER.DCS.REQUESTEDTIME'] = (60.0, 'Header.Dcs.RequestedTime')
		hdr['HEADER.DCS.SCANNUM'] = (2.0, 'Header.Dcs.ScanNum')
		hdr['HEADER.DCS.SUBOBSNUM'] = (0.0, 'Header.Dcs.SubObsNum')
		hdr['HEADER.GPS.IGNORELOCK'] = (1.0, 'Header.Gps.IgnoreLock')
		hdr['HEADER.LISSAJOUS.EXECMODE'] = (0.0, 'Header.Lissajous.ExecMode')
		hdr['HEADER.LISSAJOUS.SCANRATE'] = (0.000242406840554768, 'Header.Lissajous.ScanRate')
		hdr['HEADER.LISSAJOUS.TSCAN'] = (60.0, 'Header.Lissajous.TScan')
		hdr['HEADER.LISSAJOUS.XDELTA'] = (0.785398163397448, 'Header.Lissajous.XDelta')
		hdr['HEADER.LISSAJOUS.XDELTAMINOR'] = (0.0, 'Header.Lissajous.XDeltaMinor')
		hdr['HEADER.LISSAJOUS.XLENGTH'] = (0.000581776417331443, 'Header.Lissajous.XLength')
		hdr['HEADER.LISSAJOUS.XLENGTHMINOR'] = (0.0, 'Header.Lissajous.XLengthMinor')
		hdr['HEADER.LISSAJOUS.XOMEGA'] = (5.0, 'Header.Lissajous.XOmega')
		hdr['HEADER.LISSAJOUS.XOMEGAMINOR'] = (0.25, 'Header.Lissajous.XOmegaMinor')
		hdr['HEADER.LISSAJOUS.XOMEGAMINORNORM'] = (0.0325362003934596, 'Header.Lissajous.XOmegaMinorNorm')
		hdr['HEADER.LISSAJOUS.XOMEGANORM'] = (0.650724007869192, 'Header.Lissajous.XOmegaNorm')
		hdr['HEADER.LISSAJOUS.YLENGTH'] = (0.000581776417331443, 'Header.Lissajous.YLength')
		hdr['HEADER.LISSAJOUS.YLENGTHMINOR'] = (0.0, 'Header.Lissajous.YLengthMinor')
		hdr['HEADER.LISSAJOUS.YOMEGA'] = (4.0, 'Header.Lissajous.YOmega')
		hdr['HEADER.LISSAJOUS.YOMEGAMINOR'] = (0.2, 'Header.Lissajous.YOmegaMinor')
		hdr['HEADER.LISSAJOUS.YOMEGAMINORNORM'] = (0.0260289603147677, 'Header.Lissajous.YOmegaMinorNorm')
		hdr['HEADER.LISSAJOUS.YOMEGANORM'] = (0.520579206295354, 'Header.Lissajous.YOmegaNorm')
		hdr['HEADER.M1.ACTPOS'] = (-137.0, 'Header.M1.ActPos')
		hdr['HEADER.M1.CMDPOS'] = (-145.0, 'Header.M1.CmdPos')
		hdr['HEADER.M1.MODELENABLED'] = (1.0, 'Header.M1.ModelEnabled')
		hdr['HEADER.M1.MODELMODE'] = (1.0, 'Header.M1.ModelMode')
		hdr['HEADER.M1.ZERNIKEC'] = (46.0, 'Header.M1.ZernikeC')
		hdr['HEADER.M1.ZERNIKEENABLED'] = (1.0, 'Header.M1.ZernikeEnabled')
		hdr['HEADER.M2.ACUHEARTBEAT'] = (879385.0, 'Header.M2.AcuHeartbeat')
		hdr['HEADER.M2.ALIVE'] = (1.0, 'Header.M2.Alive')
		hdr['HEADER.M2.AZPCOR'] = (0.000123139960045208, 'Header.M2.AzPcor')
		hdr['HEADER.M2.CORENABLED'] = (1.0, 'Header.M2.CorEnabled')
		hdr['HEADER.M2.ELCMD'] = (1.16657605208666, 'Header.M2.ElCmd')
		hdr['HEADER.M2.ELPCOR'] = (-9.16959614270676E-05, 'Header.M2.ElPcor')
		hdr['HEADER.M2.FOLLOW'] = (3.0, 'Header.M2.Follow')
		hdr['HEADER.M2.HOLD'] = (0.0, 'Header.M2.Hold')
		hdr['HEADER.M2.M2HEARTBEAT'] = (12.0, 'Header.M2.M2Heartbeat')
		hdr['HEADER.M2.MODELMODE'] = (0.0, 'Header.M2.ModelMode')
		hdr['HEADER.M2.TILTACT'] = (-7.80175014369888E-07, 'Header.M2.TiltAct')
		hdr['HEADER.M2.TILTCMD'] = (0.0, 'Header.M2.TiltCmd')
		hdr['HEADER.M2.TILTDES'] = (0.0, 'Header.M2.TiltDes')
		hdr['HEADER.M2.TILTPCOR'] = (0.0, 'Header.M2.TiltPcor')
		hdr['HEADER.M2.TILTREQ'] = (0.0, 'Header.M2.TiltReq')
		hdr['HEADER.M2.TIPACT'] = (-1.23662459827756E-06, 'Header.M2.TipAct')
		hdr['HEADER.M2.TIPCMD'] = (0.0, 'Header.M2.TipCmd')
		hdr['HEADER.M2.TIPDES'] = (0.0, 'Header.M2.TipDes')
		hdr['HEADER.M2.TIPPCOR'] = (0.0, 'Header.M2.TipPcor')
		hdr['HEADER.M2.TIPREQ'] = (0.0, 'Header.M2.TipReq')
		hdr['HEADER.M2.XACT'] = (-2.88632965087891, 'Header.M2.XAct')
		hdr['HEADER.M2.XCMD'] = (-2.8863, 'Header.M2.XCmd')
		hdr['HEADER.M2.XDES'] = (-2.8863, 'Header.M2.XDes')
		hdr['HEADER.M2.XPCOR'] = (-4.2863, 'Header.M2.XPcor')
		hdr['HEADER.M2.XREQ'] = (0.0, 'Header.M2.XReq')
		hdr['HEADER.M2.YACT'] = (2.14922118186951, 'Header.M2.YAct')
		hdr['HEADER.M2.YCMD'] = (2.14927837697674, 'Header.M2.YCmd')
		hdr['HEADER.M2.YDES'] = (2.14927837697674, 'Header.M2.YDes')
		hdr['HEADER.M2.YPCOR'] = (0.499278376976741, 'Header.M2.YPcor')
		hdr['HEADER.M2.YREQ'] = (0.0, 'Header.M2.YReq')
		hdr['HEADER.M2.ZACT'] = (-19.8716411590576, 'Header.M2.ZAct')
		hdr['HEADER.M2.ZCMD'] = (-19.8718474284995, 'Header.M2.ZCmd')
		hdr['HEADER.M2.ZDES'] = (-19.8718474284995, 'Header.M2.ZDes')
		hdr['HEADER.M2.ZPCOR'] = (-18.9188674977295, 'Header.M2.ZPcor')
		hdr['HEADER.M2.ZREQ'] = (-0.95297993077, 'Header.M2.ZReq')
		hdr['HEADER.M3.ACUHEARTBEAT'] = (222234349.0, 'Header.M3.AcuHeartbeat')
		hdr['HEADER.M3.ALIVE'] = (1.0, 'Header.M3.Alive')
		hdr['HEADER.M3.ELDESENABLED'] = (1.0, 'Header.M3.ElDesEnabled')
		hdr['HEADER.M3.FAULT'] = (0.0, 'Header.M3.Fault')
		hdr['HEADER.M3.M3HEARTBEAT'] = (3557.0, 'Header.M3.M3Heartbeat')
		hdr['HEADER.M3.M3OFFPOS'] = (0.0, 'Header.M3.M3OffPos')
		hdr['HEADER.POINTMODEL.AZM2COR'] = (0.000123139960045208, 'Header.PointModel.AzM2Cor')
		hdr['HEADER.POINTMODEL.AZPADDLEOFF'] = (0.0, 'Header.PointModel.AzPaddleOff')
		hdr['HEADER.POINTMODEL.AZPOINTMODELCOR'] = (0.0014347372815083, 'Header.PointModel.AzPointModelCor')
		hdr['HEADER.POINTMODEL.AZRECEIVERCOR'] = (0.0, 'Header.PointModel.AzReceiverCor')
		hdr['HEADER.POINTMODEL.AZRECEIVEROFF'] = (0.0, 'Header.PointModel.AzReceiverOff')
		hdr['HEADER.POINTMODEL.AZTILTCOR'] = (0.0, 'Header.PointModel.AzTiltCor')
		hdr['HEADER.POINTMODEL.AZTOTALCOR'] = (0.00151865245756675, 'Header.PointModel.AzTotalCor')
		hdr['HEADER.POINTMODEL.AZUSEROFF'] = (-3.92247839867614E-05, 'Header.PointModel.AzUserOff')
		hdr['HEADER.POINTMODEL.ELM2COR'] = (-9.16959614270676E-05, 'Header.PointModel.ElM2Cor')
		hdr['HEADER.POINTMODEL.ELPADDLEOFF'] = (0.0, 'Header.PointModel.ElPaddleOff')
		hdr['HEADER.POINTMODEL.ELPOINTMODELCOR'] = (0.000970807427510183, 'Header.PointModel.ElPointModelCor')
		hdr['HEADER.POINTMODEL.ELRECEIVERCOR'] = (0.0, 'Header.PointModel.ElReceiverCor')
		hdr['HEADER.POINTMODEL.ELRECEIVEROFF'] = (0.0, 'Header.PointModel.ElReceiverOff')
		hdr['HEADER.POINTMODEL.ELREFRACCOR'] = (7.33874041145806E-05, 'Header.PointModel.ElRefracCor')
		hdr['HEADER.POINTMODEL.ELTILTCOR'] = (0.0, 'Header.PointModel.ElTiltCor')
		hdr['HEADER.POINTMODEL.ELTOTALCOR'] = (0.00103435554851646, 'Header.PointModel.ElTotalCor')
		hdr['HEADER.POINTMODEL.ELUSEROFF'] = (8.18566783187621E-05, 'Header.PointModel.ElUserOff')
		hdr['HEADER.POINTMODEL.M2CORENABLED'] = (1.0, 'Header.PointModel.M2CorEnabled')
		hdr['HEADER.POINTMODEL.MODREV'] = (36.0, 'Header.PointModel.ModRev')
		hdr['HEADER.POINTMODEL.POINTMODELCORENABLED'] = (1.0, 'Header.PointModel.PointModelCorEnabled')
		hdr['HEADER.POINTMODEL.RECEIVEROFFENABLED'] = (1.0, 'Header.PointModel.ReceiverOffEnabled')
		hdr['HEADER.POINTMODEL.REFRACCORENABLED'] = (1.0, 'Header.PointModel.RefracCorEnabled')
		hdr['HEADER.POINTMODEL.TILTCORENABLED'] = (0.0, 'Header.PointModel.TiltCorEnabled')
		hdr['HEADER.RADIOMETER.TAU'] = (0.018, 'Header.Radiometer.Tau')
		hdr['HEADER.RADIOMETER.TAU2'] = (0.0, 'Header.Radiometer.Tau2')
		hdr['HEADER.SCANFILE.VALID'] = (1.0, 'Header.ScanFile.Valid')
		hdr['HEADER.SKY.BARYVEL'] = (-5.43941966183112, 'Header.Sky.BaryVel')
		hdr['HEADER.SKY.OBSVEL'] = (-9.48009204905341, 'Header.Sky.ObsVel')
		hdr['HEADER.SKY.PARANG'] = (-2.64763592594106, 'Header.Sky.ParAng')
		hdr['HEADER.SKY.RAOFFSETSYS'] = (1.0, 'Header.Sky.RaOffsetSys')
		hdr['HEADER.SOURCE.B'] = (1.24739573016117, 'Header.Source.B')
		hdr['HEADER.SOURCE.COORDSYS'] = (0.0, 'Header.Source.CoordSys')
		hdr['HEADER.SOURCE.DEC'] = (0.697716234865568, 'Header.Source.Dec')
		hdr['HEADER.SOURCE.DECPROPERMOTIONCOR'] = (0.0, 'Header.Source.DecProperMotionCor')
		hdr['HEADER.SOURCE.ELOBSMAX'] = (1.39626340159546, 'Header.Source.ElObsMax')
		hdr['HEADER.SOURCE.ELOBSMIN'] = (0.349065850398866, 'Header.Source.ElObsMin')
		hdr['HEADER.SOURCE.EPOCH'] = (2000.0, 'Header.Source.Epoch')
		hdr['HEADER.SOURCE.L'] = (2.87893480087201, 'Header.Source.L')
		hdr['HEADER.SOURCE.PLANET'] = (0.0, 'Header.Source.Planet')
		hdr['HEADER.SOURCE.RA'] = (3.08474567269498, 'Header.Source.Ra')
		hdr['HEADER.SOURCE.RAPROPERMOTIONCOR'] = (0.0, 'Header.Source.RaProperMotionCor')
		hdr['HEADER.SOURCE.VELSYS'] = (0.0, 'Header.Source.VelSys')
		hdr['HEADER.SOURCE.VELOCITY'] = (0.0, 'Header.Source.Velocity')
		hdr['HEADER.TELESCOPE.AZACTPOS'] = (6.69524128030548, 'Header.Telescope.AzActPos')
		hdr['HEADER.TELESCOPE.AZDESPOS'] = (6.69524606425013, 'Header.Telescope.AzDesPos')
		hdr['HEADER.TELESCOPE.CRANEINBEAM'] = (0.0, 'Header.Telescope.CraneInBeam')
		hdr['HEADER.TELESCOPE.ELACTPOS'] = (1.1676073367912, 'Header.Telescope.ElActPos')
		hdr['HEADER.TELESCOPE.ELDESPOS'] = (1.1676067986496, 'Header.Telescope.ElDesPos')
		hdr['HEADER.TELESCOPE.POINTINGTOLERANCE'] = (9.69627362219072E-06, 'Header.Telescope.PointingTolerance')
		hdr['HEADER.TELESCOPEBACKEND.CALOBSNUM'] = (0.0, 'Header.TelescopeBackend.CalObsNum')
		hdr['HEADER.TELESCOPEBACKEND.MASTER'] = (0.0, 'Header.TelescopeBackend.Master')
		hdr['HEADER.TELESCOPEBACKEND.NUMPIXELS'] = (1.0, 'Header.TelescopeBackend.NumPixels')
		hdr['HEADER.TELESCOPEBACKEND.OBSNUM'] = (152393.0, 'Header.TelescopeBackend.ObsNum')
		hdr['HEADER.TELESCOPEBACKEND.SCANNUM'] = (2.0, 'Header.TelescopeBackend.ScanNum')
		hdr['HEADER.TELESCOPEBACKEND.SUBOBSNUM'] = (0.0, 'Header.TelescopeBackend.SubObsNum')
		hdr['HEADER.TILTMETER_0_.TEMP'] = (1.3671875, 'Header.Tiltmeter_0_.Temp')
		hdr['HEADER.TILTMETER_0_.TILTX'] = (-4.730224609375E-06, 'Header.Tiltmeter_0_.TiltX')
		hdr['HEADER.TILTMETER_0_.TILTY'] = (9.1705322265625E-05, 'Header.Tiltmeter_0_.TiltY')
		hdr['HEADER.TILTMETER_1_.TEMP'] = (1.5289306640625, 'Header.Tiltmeter_1_.Temp')
		hdr['HEADER.TILTMETER_1_.TILTX'] = (-0.000797119140625, 'Header.Tiltmeter_1_.TiltX')
		hdr['HEADER.TILTMETER_1_.TILTY'] = (-0.00036895751953125, 'Header.Tiltmeter_1_.TiltY')
		hdr['HEADER.TIMEPLACE.LST'] = (2.88617322737119, 'Header.TimePlace.LST')
		hdr['HEADER.TIMEPLACE.OBSELEVATION'] = (4.64, 'Header.TimePlace.ObsElevation')
		hdr['HEADER.TIMEPLACE.OBSLATITUDE'] = (0.331370114380515, 'Header.TimePlace.ObsLatitude')
		hdr['HEADER.TIMEPLACE.OBSLONGITUDE'] = (-1.69846005926431, 'Header.TimePlace.ObsLongitude')
		hdr['HEADER.TIMEPLACE.UT1'] = (1.97938206486085, 'Header.TimePlace.UT1')
		hdr['HEADER.TIMEPLACE.UTDATE'] = (2026.13510966696, 'Header.TimePlace.UTDate')
		hdr['HEADER.TOLTEC.AZPOINTCOR'] = (0.0, 'Header.Toltec.AzPointCor')
		hdr['HEADER.TOLTEC.AZPOINTOFF'] = (0.0, 'Header.Toltec.AzPointOff')
		hdr['HEADER.TOLTEC.BEAMSELECTED'] = (0.0, 'Header.Toltec.BeamSelected')
		hdr['HEADER.TOLTEC.ELPOINTCOR'] = (0.0, 'Header.Toltec.ElPointCor')
		hdr['HEADER.TOLTEC.ELPOINTOFF'] = (0.0, 'Header.Toltec.ElPointOff')
		hdr['HEADER.TOLTEC.M3DIR'] = (0.0, 'Header.Toltec.M3Dir')
		hdr['HEADER.TOLTEC.NUMBANDS'] = (0.0, 'Header.Toltec.NumBands')
		hdr['HEADER.TOLTEC.NUMBEAMS'] = (0.0, 'Header.Toltec.NumBeams')
		hdr['HEADER.TOLTEC.NUMPIXELS'] = (0.0, 'Header.Toltec.NumPixels')
		hdr['HEADER.TOLTEC.REMOTE'] = (0.0, 'Header.Toltec.Remote')
		hdr['HEADER.WEATHER.HUMIDITY'] = (15.6650000000031, 'Header.Weather.Humidity')
		hdr['HEADER.WEATHER.PRECIPITATION'] = (0.0, 'Header.Weather.Precipitation')
		hdr['HEADER.WEATHER.PRESSURE'] = (0.589103333333335, 'Header.Weather.Pressure')
		hdr['HEADER.WEATHER.RADIATION'] = (-0.065037999999986, 'Header.Weather.Radiation')
		hdr['HEADER.WEATHER.TEMPERATURE'] = (2.29999999999921, 'Header.Weather.Temperature')
		hdr['HEADER.WEATHER.TIMEOFDAY'] = (0.0, 'Header.Weather.TimeOfDay')
		hdr['HEADER.WEATHER.WINDDIR1'] = (333.433333333333, 'Header.Weather.WindDir1')
		hdr['HEADER.WEATHER.WINDDIR2'] = (0.0, 'Header.Weather.WindDir2')
		hdr['HEADER.WEATHER.WINDSPEED1'] = (5.27499999999983, 'Header.Weather.WindSpeed1')
		hdr['HEADER.WEATHER.WINDSPEED2'] = (0.0, 'Header.Weather.WindSpeed2')


	
	return hdr


def make_image_hdu(image,array,extname):
	if array=='a2000':
		crval3 = 1.50E+11

	elif array=='a1400':
		crval3 = 2.20E+11

	elif array=='a1100':
		crval3 = 2.80E+11

	tmpim = np.ones([1,1,image.shape[0],image.shape[1]])
	tmpim[0,0,::-1,:] = image
	image_hdu = fits.ImageHDU(tmpim)
	with warnings.catch_warnings():
		warnings.filterwarnings(
			"ignore",
			category=VerifyWarning,
			message=r".*HIERARCH card will be created.*",
		)
		warnings.filterwarnings(
		"ignore",
		category=VerifyWarning,
		message=r".*Card is too long, comment will be truncated.*",
		)
		image_hdu.header['EXTNAME'] = (extname,'')
		image_hdu.header['HDUVERS'] = (1, '')
		image_hdu.header['EQUINOX'] = (2000.0, 'WCS: Equinox')
		image_hdu.header['CTYPE1'] = ('AZOFFSET', 'WCS: Projection Type 1')
		image_hdu.header['CUNIT1'] = ('arcsec  ', 'WCS: Axis Unit 1')
		image_hdu.header['CRVAL1'] = (image.wcs.wcs.crval[0]*3600., 'WCS: Ref Pixel Value 1')
		image_hdu.header['CDELT1'] = (image.wcs.wcs.cdelt[0]*3600., 'WCS: Pixel Scale 1')
		image_hdu.header['CRPIX1'] = (image.wcs.wcs.crpix[0], 'WCS: Ref Pixel 1')
		image_hdu.header['CTYPE2'] = ('ELOFFSET', 'WCS: Projection Type 2')
		image_hdu.header['CUNIT2'] = ('arcsec  ', 'WCS: Axis Unit 2')
		image_hdu.header['CRVAL2'] = (image.wcs.wcs.crval[1]*3600., 'WCS: Ref Pixel Value 2')
		image_hdu.header['CDELT2'] = (image.wcs.wcs.cdelt[1]*3600., 'WCS: Pixel Scale 2')
		image_hdu.header['CRPIX2'] = (image.wcs.wcs.crpix[1], 'WCS: Ref Pixel 2')
		image_hdu.header['CTYPE3'] = ('FREQ    ', 'WCS: Projection Type 3')
		image_hdu.header['CUNIT3'] = ('Hz      ', 'WCS: Axis Unit 3')
		image_hdu.header['CRVAL3'] = (crval3, 'WCS: Ref Pixel Value 3')
		image_hdu.header['CDELT3'] = (1.0, 'WCS: Pixel Scale 3')
		image_hdu.header['CRPIX3'] = (1.0, 'WCS: Ref Pixel 3')
		image_hdu.header['CTYPE4'] = ('STOKES  ', 'WCS: Projection Type 4')
		image_hdu.header['CUNIT4'] = ('        ', 'WCS: Axis Unit 4')
		image_hdu.header['CRVAL4'] = (0.0, 'WCS: Ref Pixel Value 4')
		image_hdu.header['CDELT4'] = (1.0, 'WCS: Pixel Scale 4')
		image_hdu.header['CRPIX4'] = (1.0, 'WCS: Ref Pixel 4')
		image_hdu.header['UNIT'] = ('mJy/beam', 'Unit of map')

	return image_hdu

















