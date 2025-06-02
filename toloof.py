import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from pixell import enmap, enplot, reproject, utils, curvedsky,wcsutils 
from matplotlib import cm
from scipy.optimize import curve_fit,minimize
from scipy.linalg import sqrtm
from scipy.stats import mode
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
import matplotlib as mpl
from math import factorial

class citlali_maps:
	def __init__(self,fitsfilepath):
		fitsfile = fits.open(fitsfilepath)

		self.primary_header = fitsfile[0].header
		
		self.maps = {}
		
		for i, hdu in enumerate(fitsfile):
			# Check if there is data in the HDU (it might be None for non-image extensions)
			if hdu.data is not None:
				# Try to get the extension name from the header.
				extname = hdu.header.get('EXTNAME')
				# If there's no EXTNAME, use the index as a fallback.
				key = extname if extname is not None else f"EXT_{i}"
				
				tmpimage = hdu.data[0,0,:,:]
				tmpheader = hdu.header.copy()
				del tmpheader['NAXIS3']
				del tmpheader['NAXIS4']
				del tmpheader['CTYPE3']
				del tmpheader['CUNIT3']
				del tmpheader['CRVAL3']
				del tmpheader['CDELT3']
				del tmpheader['CRPIX3']
				del tmpheader['CTYPE4']
				del tmpheader['CUNIT4']
				del tmpheader['CRVAL4']
				del tmpheader['CDELT4']
				del tmpheader['CRPIX4']
				tmpheader['NAXIS'] = 2
				tmpwcs = WCS(header=tmpheader)
				tmpenmap = enmap.enmap(tmpimage,wcs=tmpwcs)
				
				self.maps[key] = tmpenmap
		fitsfile.close()

	def convert_signalmap_to_MJypersr(self,arrayname):
		if arrayname=='a2000':
			beamsize = 10.*u.arcsec
		elif arrayname=='a1400':
			beamsize = 6.*u.arcsec
		elif arrayname=='a1100':
			beamsize = 5.*u.arcsec
		beamsolidangle = 2*np.pi*(beamsize/2.355)**2
		convertfactor = ((1.*u.mJy/(beamsolidangle)).to(u.MJy/u.sr)).value
		self.maps['signal_I'] = self.maps['signal_I']*convertfactor

	def make_submaps(self,boxcenter,boxsize):
		self.submaps = {}
		box = np.deg2rad(np.array([[boxcenter[1]-boxsize/2.,boxcenter[0]+boxsize/2.],[boxcenter[1]+boxsize/2.,boxcenter[0]-boxsize/2.]]))
		for i in self.maps:
			self.submaps[i] = enmap.submap(self.maps[i],box)


	def fit_2DGaussian(self,
		gauss_center_guess=np.array([0.,0.]),fwhms_guess=np.array([5./3600,5./3600.]),
		theta_guess=0., amp_guess=None,bounds=None,
		print_fit_results = False):
		if not hasattr(self, 'submaps'):
			print("submaps have not been generated. Do that first.")
			return 0
		else:
			if amp_guess is None:
				tmpampguess = np.amax(self.submaps['signal_I'])
			else:
				tmpampguess = amp_guess
			beam_params = np.array([gauss_center_guess[0],gauss_center_guess[1],fwhms_guess[0],fwhms_guess[1],theta_guess,tmpampguess])

			if bounds is None:
				bounds_center0 = (gauss_center_guess[0]-(5./3600.),gauss_center_guess[0]+(5./3600.))
				bounds_center1 = (gauss_center_guess[1]-(5./3600.),gauss_center_guess[1]+(5./3600.))
				bounds_fwhm0 = (5./3600.,12./3600.)
				bounds_fwhm1 = (5./3600.,12./3600.)
				bounds_theta0 = (-5,5)
				bounds_amp = (0.75*np.amax(self.submaps['signal_I']),1.25*np.amax(self.submaps['signal_I']))
				bounds_tot = (bounds_center0,bounds_center1,bounds_fwhm0,bounds_fwhm1,bounds_theta0,bounds_amp)


			results = minimize(beam_fit_chisquare,beam_params,args=self.submaps['signal_I'],bounds=bounds_tot)
			self.beamfit_results = results.x
			if print_fit_results:
				print_results(results)
			tmpresults = gaussian_2D(self.submaps['signal_I'],[results.x[0],results.x[1]],[results.x[2],results.x[3]],results.x[4],results.x[5])
			self.fitted_beam = tmpresults
			return results



	def plot_summary(self,signal_map_vmin = None,signal_map_vmax=None):


		fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), subplot_kw={'projection': self.maps['signal_I'].wcs})


		# Flatten axes array for easy iteration
		#axes = axes.flatten()
		
		for i,name in enumerate(self.maps):
			# Set the correct WCS projection for each subplot

			ax = plt.subplot(2, 3, i + 1)


		
			# Display the image with WCS projection

			tmpvmin = self.maps[name].min()
			tmpvmax = self.maps[name].max()
			if signal_map_vmin is not None:
				tmpsignal_vmin = signal_map_vmin
			if signal_map_vmax is not None:
				tmpsignal_vmax = signal_map_vmax
			if name=='signal_I' and signal_map_vmin==None:
				tmpsignal_vmin = np.median(self.maps[name][self.maps[name]!=0.0])-(0.001*np.median(self.maps[name][self.maps[name]!=0.0]))
			if name=='signal_I' and signal_map_vmax==None:
				tmpsignal_vmax = np.median(self.maps[name][self.maps[name]!=0.0])+(500*np.median(self.maps[name][self.maps[name]!=0.0]))
			if name=='signal_I':
				tmpvmin = tmpsignal_vmin
				tmpvmax = tmpsignal_vmax
			if name=='kernel':
				tmpvmin=0
				tmpvmax=0.9
			
			img = ax.imshow(self.maps[name], origin='lower', cmap='gray', vmin=tmpvmin, vmax=tmpvmax)
			cbar = fig.colorbar(img, ax=ax, orientation="vertical", shrink=0.5)
			ax.set_title(name)
		
			# Set WCS coordinate labels
			ax.set_xlabel("RA")
			ax.set_ylabel("Dec")
		
		plt.tight_layout()
		plt.subplots_adjust(hspace=-0.3) 
		plt.show()


def make_mask_enmap(testmap,radius_deg,centervals = None,apod_width = None):

	opos0 = testmap.posmap()
	dec0, ra0 = opos0[0,:,:],  opos0[1,:,:]

	# get centers
	if centervals is not None:
		ra_cent0 = np.deg2rad(centervals[0])
		dec_cent0 = np.deg2rad(centervals[1])
		
	else:
		ra_cent0 = np.mean(ra0)
		dec_cent0 = np.mean(dec0)

	ra0_centered = np.rad2deg(ra0 - ra_cent0)
	dec0_centered = np.rad2deg(dec0 - dec_cent0)

	radsquared = ra0_centered**2 + dec0_centered**2


	rad1 = radius_deg


	innercircle = np.where(radsquared<=rad1**2)

	maskarray = np.zeros(radsquared.shape)

	maskarray[innercircle[0],innercircle[1]] = 1.0
	
	mask_map = enmap.enmap(maskarray, wcs=testmap.wcs)
	
	if apod_width is None:
		return mask_map
	else:
		apodmask = enmap.apod_mask(mask_map, width=np.deg2rad(apod_width))

		return apodmask

def make_coordinate_grids(N,L):
	## make coordinate arrays on the aperature palne
	x,y = np.meshgrid(np.linspace(-L/2,L/2,N),  ## cartesian coordinates
					  np.linspace(-L/2,L/2,N))
	r = np.sqrt(x**2 + y**2)                    ## radial coordainte
	phi = np.arctan2(y,x)
	return(x,y,r,phi)

def gaussian(rarray,sig):
	return np.exp(-0.5*rarray**2/sig**2)#(1./(sig*np.sqrt(2.*np.pi)))*np.exp(-0.5*rarray**2/sig**2)
def edge_taper(rarray,diameter):
	tmparray = np.zeros(rarray.shape)
	tmparray[np.where(rarray<diameter/2.)]=1.
	return tmparray#(1./(sig*np.sqrt(2.*np.pi)))*np.exp(-0.5*rarray**2/sig**2)

def Fraunhofer(A,wavelength,delta_x):
	U = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(A)))
	angular_width = wavelength / delta_x  ## this is in radians
	angular_width *= 180./np.pi       ## this converts to degrees
	angular_width *= 60      ## this converts to arcmin
	#angular_width *= 60      ## this converts to arcsec
	return(angular_width,U)

def Convert_field_to_PSF(U):
	PSF = np.abs(U)**2
	PSF_temp= PSF #/ np.max(PSF)  ## peak normalized-- not the right thing for convolution... but is nice for seeing it
	## make a logrythmic version of the radiated power so we can better see faint features
	PSF_temp +=1e-25 ## avoid infinities by regularizing zeroes
	PSF_dB = 10#*np.log10(PSF_temp)
	return(PSF, PSF_dB)

def radial_poly(n,m,rho):
    if (n-abs(m))%2==1:
        return np.zeros(rho.shape)
    else:
        topval = int((n-abs(m))/2.)
        radial_poly = 0

        for k in range(0,topval+1):
            numerator = ((-1)**k)*factorial(int(n-k))
            denominator = factorial(k)*factorial(int((n+abs(m))/2)-k)*factorial(int((n-abs(m))/2)-k)
            radial_poly+= (numerator/denominator)*rho**(n-(2*k))
            #k+=1
        return radial_poly
            
def kron_delta(m,n):
    if m==n:
        return 1
    else:
        return 0
def zern_normalization(n,m):
    return np.sqrt((2.*(n+1))/(1+kron_delta(m,0)))
    

def zernike_poly(n,m,rho,phi):
    radial_part = radial_poly(n,abs(m),rho)
    if m==0:
        angular_part = 1.
    elif m>0:
        angular_part = np.cos(m*phi)
    elif m<0:
        angular_part = -np.sin(m*phi)
    tmpzern = radial_part*angular_part
    tmpzern[np.where(rho>0.9999)] = 0
    return tmpzern*zern_normalization(n,m)

def gen_zernike_polys(n,m,rho,phi):
    if (n-m)%2!=0:
        print('n-m must be even!!!!!')
        return False
    numberofpolys = ((n*(n+2))+m)/2
    zernike_array = np.empty([int(numberofpolys)+1,rho.shape[0],rho.shape[1]])
    for i in range(n+1):
        m_max_tmp = i
        for j in range(-m_max_tmp,m_max_tmp+1):
            if (i-abs(j))%2==1:
                continue
            else:
                jlabel = ((i*(i+2))+j)/2.
                zernike_array[int(jlabel),:,:] = zernike_poly(i,j,rho,phi)
    return zernike_array

def gen_defocus_cassegrain_telescope(r,dz,f=17.5,F=525.,D=50.):
    a = r/(2.*f)
    b = r/(2.*F)
    tmpdefocus = dz*(((1-a**2)/(1+a**2))+((1-b**2)/(1+b**2)))
    tmpdefocus[np.where(r>(D/2.))]=0
    return tmpdefocus



class Fraunhofer_Beamfit:
	def __init__(self,paths2files,wavelength,mask_radius=2./60.,map_center = [0.,0.],padpixels = None,inputfitsfileformat='citlali'):
		
		# this class fits one wavelength at a time
			
		self.wavelength = wavelength
		
		self.surface_error = {}
		
		#load in the maps 
		#only make this work with citlali maps
		tmpmapdict = {}
		signalmaps = {}
		mapnums = np.array(len(paths2files))
		for i,path in enumerate(paths2files):
			tmpmap = citlali_maps(path)
			tmpmapdict['map'+str(i)] = tmpmap
			signalmaps['map'+str(i)] = tmpmap.maps['signal_I']

		for i in signalmaps:
			tmpmap = signalmaps[i]
			if tmpmap.wcs.wcs.ctype[0]=='AZOFFSET':
				tmpmap.wcs.wcs.ctype[0] = 'RA---TAN'
				tmpmap.wcs.wcs.cdelt[0] = tmpmap.wcs.wcs.cdelt[0]/3600.
				tmpmap.wcs.wcs.cunit[0] = 'deg'
			if tmpmap.wcs.wcs.ctype[1]=='ELOFFSET':
				tmpmap.wcs.wcs.ctype[1] = 'DEC--TAN'
				tmpmap.wcs.wcs.cdelt[1] = tmpmap.wcs.wcs.cdelt[1]/3600.
				tmpmap.wcs.wcs.cunit[1] = 'deg'
			signalmaps[i] = tmpmap

		for i in signalmaps:
			tmpmap = signalmaps[i]
			tmpmapproj = enmap.project(tmpmap,signalmaps['map0'].shape,signalmaps['map0'].wcs)
			signalmaps[i] = tmpmapproj

		for i in signalmaps:
			tmpmap = signalmaps[i]
			if padpixels is not None:
				tmpmap = enmap.pad(tmpmap,padpixels)
			signalmaps[i] = tmpmap

		
		self.original_maps = signalmaps

		tmpmapload = signalmaps['map0']
		self.map_center = SkyCoord(ra=map_center[0]*u.deg,dec=map_center[1]*u.deg)
		map_mask_tmp= make_mask_enmap(tmpmapload,mask_radius,centervals = map_center,apod_width = None)
		self.map_mask = map_mask_tmp
		
		
		maskinner = make_mask_enmap(tmpmapload,1./60.,centervals = map_center,apod_width = None)
		noisering = tmpmapload*(map_mask_tmp-maskinner)
		noise_values_tmp = np.sqrt(np.mean(noisering**2))
		self.noise_values = noise_values_tmp

	def truncate_maps(self,desired_deltax_size,center_on_brightest_pix=True):
		self.newmapwcs = {}
		self.trunc_maps = {}
		self.peak_pixel = {}
		self.N = {}
		#mapsize = int(wavelength/(np.deg2rad(self.trunc_map.wcs.wcs.cdelt[1])*desired_deltax_size))
		
		for i in self.original_maps:

			masked_map = self.original_maps[i]*self.map_mask
			brightestpix = np.where(masked_map==np.amax(masked_map))
			self.peak_pixel[i] = np.amax(masked_map)
			N_tmp = int(self.wavelength/(np.deg2rad(abs(self.original_maps[i].wcs.wcs.cdelt[1]))*desired_deltax_size))
			self.N[i] = N_tmp
			if center_on_brightest_pix:
				tmpcoords = enmap.pix2sky(masked_map.shape, masked_map.wcs, brightestpix)
				center_skycoord = SkyCoord(ra=tmpcoords[1]*u.rad,dec=tmpcoords[0]*u.rad)
			else:
				center_skycoord = self.map_center
			tmp = Cutout2D(masked_map, center_skycoord, [N_tmp,N_tmp], wcs=masked_map.wcs)
			self.newmapwcs[i] = tmp.wcs
			self.trunc_maps[i] = enmap.enmap(tmp.data,wcs=tmp.wcs)


	def set_LMT_aperture(self,include_legs=False,plot_aperture=False):
		
		wavelength = self.wavelength
		delta_x = abs(wavelength/np.deg2rad(self.trunc_maps['map0'].wcs.wcs.cdelt[1]*self.trunc_maps['map0'].shape[1]))
		self.delta_x = delta_x
		#print('The Pixel Size in the Aperture Plane is ',delta_x, ' meters')
		L = self.N['map0']*delta_x
		diam_primary = 50. ## the diameter of the primary in meters
		diam_secondary = 2.5 # diameter of the secondary in meters
		legwidths = 0.5#0.125 # quadrupod leg width in meters
		quadrupod_diam = 31. # diameter of circle defined by secondary suport
		###  make the coordinate grid
		x,y,r,phi = make_coordinate_grids(self.N['map0'],L)

		self.x = x
		self.y = y
		self.r = r
		self.phi = phi
		#figure out the legs better
		xbins = x[0,:]
		ybins = y[:,0]
		self.xbins = xbins
		self.ybins = ybins
		origx1 = x[np.where(np.logical_and(r<quadrupod_diam/2.,np.abs(x)<legwidths))]
		origy1 = y[np.where(np.logical_and(r<quadrupod_diam/2.,np.abs(x)<legwidths))]
		origx2 = x[np.where(np.logical_and(r<quadrupod_diam/2.,np.abs(y)<legwidths))]
		origy2 = y[np.where(np.logical_and(r<quadrupod_diam/2.,np.abs(y)<legwidths))]
		theta_rot = np.deg2rad(45.)
		
		rotx1 = (np.cos(theta_rot)*origx1)-(np.sin(theta_rot)*origy1)
		roty1 = (np.sin(theta_rot)*origx1)+(np.cos(theta_rot)*origy1)
		
		rotx1_digitize = np.digitize(rotx1, xbins)
		roty1_digitize = np.digitize(roty1, ybins)
		
		rotx2 = (np.cos(theta_rot)*origx2)-(np.sin(theta_rot)*origy2)
		roty2 = (np.sin(theta_rot)*origx2)+(np.cos(theta_rot)*origy2)
		
		rotx2_digitize = np.digitize(rotx2, xbins)
		roty2_digitize = np.digitize(roty2, ybins)
		
		## make the aperature fields THIS IS COMMON FOR ALL WAVELENGTHS
		
		A = np.ones([self.N['map0'],self.N['map0']])
		A[np.where(r>diam_primary/2)] = 0
		A[np.where(r<diam_secondary/2.)] = 0
		
		if include_legs:
			A[roty1_digitize,rotx1_digitize] = 0
			A[roty2_digitize,rotx2_digitize] = 0
		
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
			plt.show()

	def get_zernike_polynomials(self,n,m):
		diam_primary = 50. 
		self.zernike_polynomials = gen_zernike_polys(n,m,self.r/(diam_primary/2.),self.phi)
	
	def set_phase(self,secondary_offset=0.,c=None,plot_phase=False):
		
		
		if c is None:
			c = np.zeros(self.zernike_polynomials.shape[0])
		c[0] = 0
		
		Phi = np.zeros([self.zernike_polynomials.shape[1],self.zernike_polynomials.shape[2]])
		for i in range(c.size):
			Phi+=c[i]*self.zernike_polynomials[i,:,:]
		delta_phase = gen_defocus_cassegrain_telescope(self.r,secondary_offset,f=17.5,F=525.,D=50.)

		self.phase = Phi+delta_phase
		#phase *= A
		
		if plot_phase:
			plt.figure()
			plt.imshow(self.phase,extent=([-self.L/2, self.L/2, -self.L/2, self.L/2]))
			plt.title("Phase")
			plt.xlabel("x [m]")
			plt.ylabel("y [m]")
			plt.colorbar()
			plt.show()

	def set_illumination(self,aperture_fwhm = 48.,edge_taper_diameter=48.,plot_illumination=False):

		sig0 = aperture_fwhm/(2*np.sqrt(2*np.log(2)))
		edge_taper = np.ones([self.N['map0'],self.N['map0']])
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

	def make_normalizing_amplitude(self):
		A_complex = self.Aperture*self.illumination#*np.exp(self.phase*1j)
		angular_width,U = Fraunhofer(A_complex,self.wavelength,self.delta_x)
		PSF_nom = np.abs(U)**2
		self.normalizing_amplitude = np.amax(PSF_nom)



	def make_psf(self):
		A_complex = self.Aperture*self.illumination*np.exp(self.phase*1j)
		angular_width,U = Fraunhofer(A_complex,self.wavelength,self.delta_x)
		PSF,PSF_dB = Convert_field_to_PSF(U)
		self.PSF = enmap.enmap(PSF/self.normalizing_amplitude,wcs=self.newmapwcs['map0'])
	
	def function2minimize(self,x):
		#list of fit params
		# x[0] = source amplitude
		# x[1] = Map 0 X tilt
		# x[2] = Map 0 Y tilt
		# x[3] = Map 1 X tilt
		# x[4] = Map 1 Y tilt
		# x[5] = Map 2 X tilt
		# x[6] = Map 2 Y tilt
		# the following are shared phase info
		# x[7] = Defocus Center
		# x[8] = Defocus Slope
		# x[9] = oblique astig
		# x[10] = vertical astig
		# ...

		
		# make map 0 zern params


		c_map0 = np.zeros(self.zernike_polynomials.shape[0])
		c_map0[0] = 0 # piston is always 0
		c_map0[1] = x[2] # piston is always 0
		c_map0[2] = x[3] # piston is always 0
		c_map0[3:] = x[9:]
		c_map0[4] = 0
		m2offset0 = x[8] - 1E-3

		# make map 1 zern params

		c_map1 = np.zeros(self.zernike_polynomials.shape[0])
		c_map1[0] = 0 # piston is always 0
		c_map1[1] = x[4] # piston is always 0
		c_map1[2] = x[5] # piston is always 0
		c_map1[3:] = x[9:]
		c_map1[4] = 0
		m2offset1 = x[8] 

		# make map 2 zern params

		c_map2 = np.zeros(self.zernike_polynomials.shape[0])
		c_map2[0] = 0 # piston is always 0
		c_map2[1] = x[6] # piston is always 0
		c_map2[2] = x[7] # piston is always 0
		c_map2[3:] = x[9:]
		c_map2[4] = 0
		m2offset2 = x[8] + 1E-3



		# make map 0 psf
	
		Phi = np.zeros([self.zernike_polynomials.shape[1],self.zernike_polynomials.shape[2]])
		for i in range(c_map0.size):
			Phi+=c_map0[i]*self.zernike_polynomials[i,:,:]
		delta_phase = (2*np.pi/self.wavelength)*gen_defocus_cassegrain_telescope(self.r,m2offset0,f=x[1],F=525.,D=50.)

		phase0 = Phi+delta_phase
	
		A_complex_map0 = self.Aperture*self.illumination*np.exp(phase0*1j)
		angular_width_map0,U_map0 = Fraunhofer(A_complex_map0,self.wavelength,self.delta_x)
		PSF_map0,PSF_dB = Convert_field_to_PSF(U_map0)
		model_map0 = enmap.enmap(x[0]*(PSF_map0/self.normalizing_amplitude),wcs=self.newmapwcs['map0'])
		
		resids_map0 = self.trunc_maps['map0'] - model_map0
		rms_map0 = np.sqrt(np.mean(resids_map0**2))

		# make map 1 psf
	
		Phi = np.zeros([self.zernike_polynomials.shape[1],self.zernike_polynomials.shape[2]])
		for i in range(c_map1.size):
			Phi+=c_map1[i]*self.zernike_polynomials[i,:,:]
		delta_phase = (2*np.pi/self.wavelength)*gen_defocus_cassegrain_telescope(self.r,m2offset1,f=x[1],F=525.,D=50.)

		phase1 = Phi+delta_phase
		A_complex_map1 = self.Aperture*self.illumination*np.exp(phase1*1j)
		angular_width_map1,U_map1 = Fraunhofer(A_complex_map1,self.wavelength,self.delta_x)
		PSF_map1,PSF_dB = Convert_field_to_PSF(U_map1)
		model_map1 = enmap.enmap(x[0]*(PSF_map1/self.normalizing_amplitude),wcs=self.newmapwcs['map0'])
		
		resids_map1 = self.trunc_maps['map1'] - model_map1
		rms_map1 = np.sqrt(np.mean(resids_map1**2))

		# make map 2 psf
	
		Phi = np.zeros([self.zernike_polynomials.shape[1],self.zernike_polynomials.shape[2]])
		for i in range(c_map2.size):
			Phi+=c_map2[i]*self.zernike_polynomials[i,:,:]
		delta_phase = (2*np.pi/self.wavelength)*gen_defocus_cassegrain_telescope(self.r,m2offset2,f=x[1],F=525.,D=50.)

		phase2 = Phi+delta_phase
		A_complex_map2 = self.Aperture*self.illumination*np.exp(phase2*1j)
		angular_width_map2,U_map2 = Fraunhofer(A_complex_map2,self.wavelength,self.delta_x)
		PSF_map2,PSF_dB = Convert_field_to_PSF(U_map2)
		model_map2 = enmap.enmap(x[0]*(PSF_map2/self.normalizing_amplitude),wcs=self.newmapwcs['map0'])
		
		resids_map2 = self.trunc_maps['map2'] - model_map2
		rms_map2 = np.sqrt(np.mean(resids_map2**2))
	
		
		chisquare = np.sqrt(rms_map0**2+rms_map1**2+rms_map2**2)
		self.cost = np.sum(chisquare)
		self.fit_step_counter+=1
		if self.fit_step_counter%500==0:
			print('On fit step ',self.fit_step_counter,' with cost ',np.sum(chisquare))
		return chisquare
		
	def fit_beam(self,c_guess=None,boundvals = None):
		#list of fit params
		# x[0] = source amplitude
		# x[1] = Primary Mirror focal length
		# x[2] = Map 0 X tilt
		# x[3] = Map 0 Y tilt
		# x[4] = Map 1 X tilt
		# x[5] = Map 1 Y tilt
		# x[6] = Map 2 X tilt
		# x[7] = Map 2 Y tilt
		# the following are shared phase info
		# x[8] = M2.Z offset
		# x[9] = oblique astig
		# x[10] = defocus
		# ...


		x0 = np.zeros(self.zernike_polynomials.shape[0]+6) # need to make an array that has the zernike coeffs plus 1 for theta0
		x0[0] = np.amax(self.trunc_maps['map1'])
		x0[1] = 17.5 # guess for primary focal length
		
		if c_guess is not None:
			x0[9:] = c_guess
		
		self.fit_step_counter = 0
		results = minimize(self.function2minimize,x0,bounds=boundvals)
		self.results = results
		if results.success:
			print('The fit was successful. Cost =', self.cost)
		else:
			print('The fit was not successful, you may need to run again with different guess or bounds. Cost =', self.cost)


		c_bestfit_map0 = np.empty(self.zernike_polynomials.shape[0])
		c_bestfit_map0[0] = 0
		c_bestfit_map0[1] = results.x[2]
		c_bestfit_map0[2] = results.x[3]
		c_bestfit_map0[3:] = results.x[9:]
		c_bestfit_map0[4] = 0

		c_bestfit_map1 = np.empty(self.zernike_polynomials.shape[0])
		c_bestfit_map1[0] = 0
		c_bestfit_map1[1] = results.x[4]
		c_bestfit_map1[2] = results.x[5]
		c_bestfit_map1[3:] = results.x[9:]
		c_bestfit_map1[4] = 0

		c_bestfit_map2 = np.empty(self.zernike_polynomials.shape[0])
		c_bestfit_map2[0] = 0
		c_bestfit_map2[1] = results.x[6]
		c_bestfit_map2[2] = results.x[7]
		c_bestfit_map2[3:] = results.x[9:]
		c_bestfit_map2[4] = 0

		c_bestfit_ideal = np.zeros(self.zernike_polynomials.shape[0])
		c_bestfit_ideal[0] = 0
		c_bestfit_ideal[3:] = results.x[9:]
		c_bestfit_ideal[4] = 0

		# make bestfitmaps for resids
		self.bestfit_maps = {}

		Phi = np.zeros([self.zernike_polynomials.shape[1],self.zernike_polynomials.shape[2]])
		for i in range(c_bestfit_map0.size):
			Phi+=c_bestfit_map0[i]*self.zernike_polynomials[i,:,:]
		delta_phase = (2*np.pi/self.wavelength)*gen_defocus_cassegrain_telescope(self.r,self.results.x[8]-1E-3,f=results.x[1],F=525.,D=50.)

		phase0_fit = Phi+delta_phase

		A_complex = self.Aperture*self.illumination*np.exp(phase0_fit*1j)
		angular_width,U = Fraunhofer(A_complex,self.wavelength,self.delta_x)
		PSF,PSF_dB = Convert_field_to_PSF(U)
		tmppsf_raw = enmap.enmap(results.x[0]*(PSF/self.normalizing_amplitude),wcs=self.newmapwcs['map0'])
		self.bestfit_maps['map0'] = tmppsf_raw

		Phi = np.zeros([self.zernike_polynomials.shape[1],self.zernike_polynomials.shape[2]])
		for i in range(c_bestfit_map1.size):
			Phi+=c_bestfit_map1[i]*self.zernike_polynomials[i,:,:]
		delta_phase = (2*np.pi/self.wavelength)*gen_defocus_cassegrain_telescope(self.r,self.results.x[8],f=results.x[1],F=525.,D=50.)

		phase1_fit = Phi+delta_phase
		A_complex = self.Aperture*self.illumination*np.exp(phase1_fit*1j)
		angular_width,U = Fraunhofer(A_complex,self.wavelength,self.delta_x)
		PSF,PSF_dB = Convert_field_to_PSF(U)
		tmppsf_raw = enmap.enmap(results.x[0]*(PSF/self.normalizing_amplitude),wcs=self.newmapwcs['map0'])
		self.bestfit_maps['map1'] = tmppsf_raw

		Phi = np.zeros([self.zernike_polynomials.shape[1],self.zernike_polynomials.shape[2]])
		for i in range(c_bestfit_map2.size):
			Phi+=c_bestfit_map2[i]*self.zernike_polynomials[i,:,:]
		delta_phase = (2*np.pi/self.wavelength)*gen_defocus_cassegrain_telescope(self.r,self.results.x[8]+1.E-3,f=results.x[1],F=525.,D=50.)

		phase2_fit = Phi+delta_phase
		A_complex = self.Aperture*self.illumination*np.exp(phase2_fit*1j)
		angular_width,U = Fraunhofer(A_complex,self.wavelength,self.delta_x)
		PSF,PSF_dB = Convert_field_to_PSF(U)
		tmppsf_raw = enmap.enmap(results.x[0]*(PSF/self.normalizing_amplitude),wcs=self.newmapwcs['map0'])
		self.bestfit_maps['map2'] = tmppsf_raw


		self.zern_coefficients = c_bestfit_ideal
		
		Phi = np.zeros([self.zernike_polynomials.shape[1],self.zernike_polynomials.shape[2]])
		for i in range(c_bestfit_ideal.size):
			Phi+=c_bestfit_ideal[i]*self.zernike_polynomials[i,:,:]

		phase_ideal_fit = Phi
		self.phase = phase_ideal_fit
		
		A_complex = self.Aperture*self.illumination*np.exp(phase_ideal_fit*1j)
		angular_width,U = Fraunhofer(A_complex,self.wavelength,self.delta_x)
		PSF,PSF_dB = Convert_field_to_PSF(U)
		tmppsf_raw = enmap.enmap(results.x[0]*(PSF/self.normalizing_amplitude),wcs=self.newmapwcs['map0'])
		gain_loss = np.amax(tmppsf_raw/results.x[0])
		self.gain_loss = gain_loss

		self.bestfitbeam = tmppsf_raw

	def plot_phase(self,plot_vmin=None,plot_vmax=None,save_fig_name=None,noshow=False):
		plt.figure()
		plt.imshow(self.phase,vmin=plot_vmin,vmax=plot_vmax,extent=([-self.L/2, self.L/2, -self.L/2, self.L/2]))
		plt.colorbar()
		
		plt.title("Phase")
		plt.xlabel("x [m]")
		plt.ylabel("y [m]")
		
		plt.xlim(-30,30)
		plt.ylim(-30,30)
		if save_fig_name is not None:
			plt.savefig(save_fig_name,facecolor='white',transparent=False,bbox_inches='tight')
		if noshow:
			plt.close()
		else:
			plt.show()

	def plot_surface_error(self,plot_vmin=None,plot_vmax=None,save_fig_name=None,noshow=False):

		tmpwavelength = self.wavelength
		
		self.surface_error = 1E6*tmpwavelength*self.phase/(2*np.pi)
		print('RMS = ',np.sqrt(np.mean(self.surface_error[np.nonzero(self.surface_error)]**2)),'microns')
		plt.figure()
		plt.imshow(self.surface_error,vmin=plot_vmin,vmax=plot_vmax,extent=([-self.L/2, self.L/2, -self.L/2, self.L/2]))
		cbar = plt.colorbar()
		cbar.set_label(label='microns',rotation=270,labelpad=15,fontsize=14)
		
		plt.title("Surface Error")
		plt.xlabel("x [m]")
		plt.ylabel("y [m]")
		
		plt.xlim(-30,30)
		plt.ylim(-30,30)
		if save_fig_name is not None:
			plt.savefig(save_fig_name,facecolor='white',transparent=False,bbox_inches='tight')
		
		if noshow:
			plt.close()
		else:
			plt.show()

	def plot_results(self,plot_vmin=None,plot_vmax=None,resids_vmin=None,resids_vmax=None,save_fig_name=None,noshow=False,plot_title=None,lowerleft=[-0.75/60,-0.75/60.],upperright=[0.75/60,0.75/60.]):
		
		fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), subplot_kw={'projection': self.trunc_maps['map0'].wcs})


		# Flatten axes array for easy iteration
		#axes = axes.flatten()
		
		for i,name in enumerate(self.trunc_maps):
			# Set the correct WCS projection for each subplot

			ax = plt.subplot(3, 3, i + 1)

			
			img = ax.imshow(self.trunc_maps[name], origin='lower', cmap='gray', vmin=plot_vmin, vmax=plot_vmax)
			cbar = fig.colorbar(img, ax=ax, orientation="vertical", shrink=0.7)
			ax.set_title(name)
		
			# Set WCS coordinate labels
			ra = ax.coords['ra']
			ra.set_format_unit(u.deg, decimal=True)

			# 1) clear the axis label
			ra.set_axislabel('')

			# 2) hide the tick marks
			ra.set_ticks_visible(False)

			# 3) hide the tick *labels*
			ra.set_ticklabel_visible(False)
			if i%3==0:
				ax.set_ylabel("El Offset")
				dec = ax.coords['dec']
				dec.set_format_unit(u.arcmin, decimal=True)
				dec.set_major_formatter('d.dd')
			else:
				# suppose ax is one of your WCSAxes subplots
				dec = ax.coords['dec']

				# 1) clear the axis label
				dec.set_axislabel('')

				# 2) hide the tick marks
				dec.set_ticks_visible(False)

				# 3) hide the tick *labels*
				dec.set_ticklabel_visible(False)



			if lowerleft is not None:
				lower_left  = SkyCoord(lowerleft[0]*u.deg,  lowerleft[1]*u.deg, frame='icrs')
				x0, y0 = self.trunc_maps['map0'].wcs.world_to_pixel(lower_left)
			if upperright is not None:
				upper_right = SkyCoord(upperright[0]*u.deg,  upperright[1]*u.deg, frame='icrs')
				x1, y1 = self.trunc_maps['map0'].wcs.world_to_pixel(upper_right)

			

			ax.set_xlim(x0, x1)
			ax.set_ylim(y0, y1)
		for i,name in enumerate(self.bestfit_maps):
			# Set the correct WCS projection for each subplot

			ax = plt.subplot(3, 3, i + 4)

			
			img = ax.imshow(self.bestfit_maps[name], origin='lower', cmap='gray', vmin=plot_vmin, vmax=plot_vmax)
			cbar = fig.colorbar(img, ax=ax, orientation="vertical", shrink=0.7)
			#ax.set_title(name)
		
			# Set WCS coordinate labels
			ra = ax.coords['ra']
			ra.set_format_unit(u.deg, decimal=True)

			# 1) clear the axis label
			ra.set_axislabel('')

			# 2) hide the tick marks
			ra.set_ticks_visible(False)

			# 3) hide the tick *labels*
			ra.set_ticklabel_visible(False)
			if i%3==0:
				ax.set_ylabel("El Offset")
				dec = ax.coords['dec']
				dec.set_format_unit(u.arcmin, decimal=True)
				dec.set_major_formatter('d.dd')
			else:
				# suppose ax is one of your WCSAxes subplots
				dec = ax.coords['dec']

				# 1) clear the axis label
				dec.set_axislabel('')

				# 2) hide the tick marks
				dec.set_ticks_visible(False)

				# 3) hide the tick *labels*
				dec.set_ticklabel_visible(False)
			if lowerleft is not None:
				lower_left  = SkyCoord(lowerleft[0]*u.deg,  lowerleft[1]*u.deg, frame='icrs')
				x0, y0 = self.trunc_maps['map0'].wcs.world_to_pixel(lower_left)
			if upperright is not None:
				upper_right = SkyCoord(upperright[0]*u.deg,  upperright[1]*u.deg, frame='icrs')
				x1, y1 = self.trunc_maps['map0'].wcs.world_to_pixel(upper_right)

			

			ax.set_xlim(x0, x1)
			ax.set_ylim(y0, y1)

		for i,name in enumerate(self.bestfit_maps):
			# Set the correct WCS projection for each subplot

			ax = plt.subplot(3, 3, i + 7)

			
			img = ax.imshow(self.trunc_maps[name]-self.bestfit_maps[name], origin='lower', cmap='gray', vmin=resids_vmin, vmax=resids_vmax)
			cbar = fig.colorbar(img, ax=ax, orientation="vertical", shrink=0.7)
			#ax.set_title(name)
		
			# Set WCS coordinate labels
			ax.set_xlabel("AZ Offset")
			ra = ax.coords['ra']
			ra.set_format_unit(u.arcmin, decimal=True)
			ra.set_major_formatter('d.dd')
			if i%3==0:
				ax.set_ylabel("El Offset")
				dec = ax.coords['dec']
				dec.set_format_unit(u.arcmin, decimal=True)
				dec.set_major_formatter('d.dd')
			else:
				# suppose ax is one of your WCSAxes subplots
				dec = ax.coords['dec']

				# 1) clear the axis label
				dec.set_axislabel('')

				# 2) hide the tick marks
				dec.set_ticks_visible(False)

				# 3) hide the tick *labels*
				dec.set_ticklabel_visible(False)
			if lowerleft is not None:
				lower_left  = SkyCoord(lowerleft[0]*u.deg,  lowerleft[1]*u.deg, frame='icrs')
				x0, y0 = self.trunc_maps['map0'].wcs.world_to_pixel(lower_left)
			if upperright is not None:
				upper_right = SkyCoord(upperright[0]*u.deg,  upperright[1]*u.deg, frame='icrs')
				x1, y1 = self.trunc_maps['map0'].wcs.world_to_pixel(upper_right)

			

			ax.set_xlim(x0, x1)
			ax.set_ylim(y0, y1)
		plt.tight_layout()
		plt.subplots_adjust(hspace=0.1,wspace=-0.4) 
		if plot_title is not None:
			plt.suptitle(plot_title)
		
		if save_fig_name is not None:
			plt.savefig(save_fig_name,facecolor='white',transparent=False,bbox_inches='tight')
		
		if noshow:
			plt.close()
		else:
			plt.show()

	def plot_psf(self,plot_vmin=None,plot_vmax=None,save_fig_name=None,xlims=None,ylims=None):
		corners_tmp = np.rad2deg(enmap.corners(self.PSF.shape,self.PSF.wcs))
		imextent_tmp = [corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]]
		fig=plt.figure()
		im = plt.imshow(self.PSF*np.amax(self.trunc_map),extent=imextent_tmp,origin='lower',vmin=plot_vmin,vmax=plot_vmax)
		plt.xticks(fontsize=10)
		plt.yticks(fontsize=10)
		if xlims is not None:
			plt.xlim(xlims)
		if ylims is not None:
			plt.ylim(ylims)
		plt.title('PSF')
		plt.colorbar()
		if save_fig_name is not None:
			plt.savefig(save_fig_name,facecolor='white',transparent=False,bbox_inches='tight')
		plt.show()

	def plot_inputmaps(self,plot_vmin=None,plot_vmax=None,save_fig_name=None,noshow=False,lowerleft=[-0.75/60,-0.75/60.],upperright=[0.75/60,0.75/60.]):
		
		fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), subplot_kw={'projection': self.trunc_maps['map0'].wcs})


		# Flatten axes array for easy iteration
		#axes = axes.flatten()
		
		for i,name in enumerate(self.trunc_maps):
			# Set the correct WCS projection for each subplot

			ax = plt.subplot(1, 3, i + 1)

			
			img = ax.imshow(self.trunc_maps[name], origin='lower', cmap='gray', vmin=plot_vmin, vmax=plot_vmax)
			cbar = fig.colorbar(img, ax=ax, orientation="vertical", shrink=0.6)
			ax.set_title(name)
		
			# Set WCS coordinate labels
			ax.set_xlabel("AZ Offset")
			ra = ax.coords['ra']
			ra.set_format_unit(u.arcmin, decimal=True)
			ra.set_major_formatter('d.dd')
			if i%3==0:
				ax.set_ylabel("El Offset")
				dec = ax.coords['dec']
				dec.set_format_unit(u.arcmin, decimal=True)
				dec.set_major_formatter('d.dd')
			else:
				# suppose ax is one of your WCSAxes subplots
				dec = ax.coords['dec']

				# 1) clear the axis label
				dec.set_axislabel('')

				# 2) hide the tick marks
				dec.set_ticks_visible(False)

				# 3) hide the tick *labels*
				dec.set_ticklabel_visible(False)



			
			lower_left  = SkyCoord(lowerleft[0]*u.deg,  lowerleft[1]*u.deg, frame='icrs')
			x0, y0 = self.trunc_maps['map0'].wcs.world_to_pixel(lower_left)
			
			upper_right = SkyCoord(upperright[0]*u.deg,  upperright[1]*u.deg, frame='icrs')
			x1, y1 = self.trunc_maps['map0'].wcs.world_to_pixel(upper_right)

			

			ax.set_xlim(x0, x1)
			ax.set_ylim(y0, y1)
		plt.tight_layout()
		plt.subplots_adjust(wspace=0.1) 
		
		if save_fig_name is not None:
			plt.savefig(save_fig_name,facecolor='white',transparent=False,bbox_inches='tight')
		
		if noshow:
			plt.close()
		else:
			plt.show()

class Fraunhofer_Image:
	def __init__(self,wavelength,pixelsize,imagesize):
		self.wavelength = wavelength
		if abs(wavelength-2.0E-3)<1E-6:
			self.arrayname = 'a2000'
		elif abs(wavelength-1.4E-3)<1E-6:
			self.arrayname = 'a1400'
		elif abs(wavelength-1.1E-3)<1E-6:
			self.arrayname = 'a1100'
		self.pixelsize = pixelsize
		self.N = int(imagesize/pixelsize)
		
		self.delta_x = wavelength/np.deg2rad(pixelsize*self.N)
		
		self.newmapwcs = wcsutils.build((0,0), res=pixelsize, shape=[self.N,self.N], system='tan')




	def set_LMT_aperture(self,include_legs=False,plot_aperture=False):
		
		wavelength = self.wavelength
		delta_x = self.delta_x
		
		#print('The Pixel Size in the Aperture Plane is ',delta_x, ' meters')
		L = self.N*delta_x
		diam_primary = 50. ## the diameter of the primary in meters
		diam_secondary = 2.5 # diameter of the secondary in meters
		legwidths = 0.5#0.125 # quadrupod leg width in meters
		quadrupod_diam = 31. # diameter of circle defined by secondary suport
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
		origx1 = x[np.where(np.logical_and(r<quadrupod_diam/2.,np.abs(x)<legwidths))]
		origy1 = y[np.where(np.logical_and(r<quadrupod_diam/2.,np.abs(x)<legwidths))]
		origx2 = x[np.where(np.logical_and(r<quadrupod_diam/2.,np.abs(y)<legwidths))]
		origy2 = y[np.where(np.logical_and(r<quadrupod_diam/2.,np.abs(y)<legwidths))]
		theta_rot = np.deg2rad(45.)
		
		rotx1 = (np.cos(theta_rot)*origx1)-(np.sin(theta_rot)*origy1)
		roty1 = (np.sin(theta_rot)*origx1)+(np.cos(theta_rot)*origy1)
		
		rotx1_digitize = np.digitize(rotx1, xbins)
		roty1_digitize = np.digitize(roty1, ybins)
		
		rotx2 = (np.cos(theta_rot)*origx2)-(np.sin(theta_rot)*origy2)
		roty2 = (np.sin(theta_rot)*origx2)+(np.cos(theta_rot)*origy2)
		
		rotx2_digitize = np.digitize(rotx2, xbins)
		roty2_digitize = np.digitize(roty2, ybins)
		
		## make the aperature fields THIS IS COMMON FOR ALL WAVELENGTHS
		
		A = np.ones([self.N,self.N])
		A[np.where(r>diam_primary/2)] = 0
		A[np.where(r<diam_secondary/2.)] = 0
		
		if include_legs:
			A[roty1_digitize,rotx1_digitize] = 0
			A[roty2_digitize,rotx2_digitize] = 0
		
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
			plt.show()

	def get_zernike_polynomials(self,n,m):
		diam_primary = 50. 
		self.zernike_polynomials = gen_zernike_polys(n,m,self.r/(diam_primary/2.),self.phi)
	
	def set_phase(self,c=None,secondary_offset=0.,f=17.5,F=525.,D=50.,plot_phase=False):
		
		
		if c is None:
			c = np.zeros(self.zernike_polynomials.shape[0])
		c[0] = 0
		
		Phi = np.zeros([self.zernike_polynomials.shape[1],self.zernike_polynomials.shape[2]])
		for i in range(c.size):
			Phi+=c[i]*self.zernike_polynomials[i,:,:]
		delta_phase = gen_defocus_cassegrain_telescope(self.r,secondary_offset,f=f,F=F,D=D)

		self.phase = Phi+(delta_phase*2.*np.pi/self.wavelength)
		#phase *= A
		
		if plot_phase:
			plt.figure()
			plt.imshow(self.phase,extent=([-self.L/2, self.L/2, -self.L/2, self.L/2]))
			plt.title("Phase")
			plt.xlabel("x [m]")
			plt.ylabel("y [m]")
			plt.colorbar()
			plt.show()

	def set_illumination(self,aperture_fwhm = 48.,edge_taper_diameter=48.,plot_illumination=False):

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

	def make_normalizing_amplitude(self):
		A_complex = self.Aperture*self.illumination#*np.exp(self.phase*1j)
		angular_width,U = Fraunhofer(A_complex,self.wavelength,self.delta_x)
		PSF_nom = np.abs(U)**2
		self.normalizing_amplitude = np.amax(PSF_nom)



	def make_psf(self):
		A_complex = self.Aperture*self.illumination*np.exp(self.phase*1j)
		angular_width,U = Fraunhofer(A_complex,self.wavelength,self.delta_x)
		PSF,PSF_dB = Convert_field_to_PSF(U)
		self.PSF = enmap.enmap(PSF/self.normalizing_amplitude,wcs=self.newmapwcs)
	
	
		
	

	def plot_phase(self,plot_vmin=None,plot_vmax=None,save_fig_name=None,noshow=False):
		plt.figure()
		plt.imshow(self.phase,vmin=plot_vmin,vmax=plot_vmax,extent=([-self.L/2, self.L/2, -self.L/2, self.L/2]))
		plt.colorbar()
		
		plt.title("Phase")
		plt.xlabel("x [m]")
		plt.ylabel("y [m]")
		
		plt.xlim(-30,30)
		plt.ylim(-30,30)
		if save_fig_name is not None:
			plt.savefig(save_fig_name,facecolor='white',transparent=False,bbox_inches='tight')
		if noshow:
			plt.close()
		else:
			plt.show()

	def plot_surface_error(self,plot_vmin=None,plot_vmax=None,save_fig_name=None,noshow=False):

		tmpwavelength = self.wavelength
		
		self.surface_error = 1E6*tmpwavelength*self.phase/(2*np.pi)
		print('RMS = ',np.sqrt(np.mean(self.surface_error[np.nonzero(self.surface_error)]**2)),'microns')
		plt.figure()
		plt.imshow(self.surface_error,vmin=plot_vmin,vmax=plot_vmax,extent=([-self.L/2, self.L/2, -self.L/2, self.L/2]))
		cbar = plt.colorbar()
		cbar.set_label(label='microns',rotation=270,labelpad=15,fontsize=14)
		
		plt.title("Surface Error")
		plt.xlabel("x [m]")
		plt.ylabel("y [m]")
		
		plt.xlim(-30,30)
		plt.ylim(-30,30)
		if save_fig_name is not None:
			plt.savefig(save_fig_name,facecolor='white',transparent=False,bbox_inches='tight')
		
		if noshow:
			plt.close()
		else:
			plt.show()

	
	
	def plot_psf(self,plot_vmin=None,plot_vmax=None,save_fig_name=None,xlims=None,ylims=None):
		corners_tmp = np.rad2deg(enmap.corners(self.PSF.shape,self.PSF.wcs))
		imextent_tmp = [corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]]
		fig=plt.figure()
		im = plt.imshow(self.PSF,extent=imextent_tmp,origin='lower',vmin=plot_vmin,vmax=plot_vmax)
		plt.xticks(fontsize=10)
		plt.yticks(fontsize=10)
		if xlims is not None:
			plt.xlim(xlims)
		if ylims is not None:
			plt.ylim(ylims)
		plt.title('PSF')
		plt.colorbar()
		if save_fig_name is not None:
			plt.savefig(save_fig_name,facecolor='white',transparent=False,bbox_inches='tight')
		plt.show()
