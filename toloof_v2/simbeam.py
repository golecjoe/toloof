import numpy as np
from pixell import enmap, enplot, reproject, utils, curvedsky,wcsutils
from scipy.ndimage import rotate
from astropy.coordinates import SkyCoord 
from astropy import units as u
from astropy.nddata import Cutout2D

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
		if c is None:
			c = np.zeros(self.zernike_polynomials.shape[0])

		

		if self.wavelengths.size==1:
			tmppsf = self.make_psf_monochromatic(self.wavelengths[0],c=c,secondary_offset=secondary_offset,
		                       del_x=del_x,del_y=del_y,del_alph_x=del_alph_x,del_alph_y=del_alph_y,
				               f=f,F=F,D=D)
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


