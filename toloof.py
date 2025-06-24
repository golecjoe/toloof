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
from pathlib import Path
import netCDF4 as nc
import glob

class CitlaliMaps:
	def __init__(self,fitsfilepath):
		"""
		Load and parse a multi-extension FITS file into enmap objects.

		Parameters:
		----------
		fits_file_path : str
			Path to the FITS file containing map data.
		"""
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

	def convert_signalmap_to_MJypersr(self, array_name, map_key='signal_I'):
		"""
		Convert a map from mJy/beam to MJy/sr using an assumed beam size.

		Parameters
		----------
		array_name : str
			Name of the instrument array (e.g., 'a2000', 'a1400', 'a1100').
		map_key : str
			Name of the map in self.maps to convert (default is 'signal_I').

		Raises
		------
		ValueError
			If the array_name is not recognized.
		"""

		beam_sizes = {
			'a2000': 10. * u.arcsec,
			'a1400': 6. * u.arcsec,
			'a1100': 5. * u.arcsec,
		}

		if array_name not in beam_sizes:
			raise ValueError(f"Unknown array_name: {array_name}")

		beamsize = beam_sizes[array_name]
		beam_solid_angle = 2 * np.pi * (beamsize / 2.355)**2
		conversion_factor = (1. * u.mJy / beam_solid_angle).to(u.MJy / u.sr).value

		self.maps[map_key] *= conversion_factor

	def make_submaps(self, box_center, box_size_deg, map_keys=None):
		"""
		Create submaps centered on a given sky position.

		Parameters
		----------
		box_center : tuple of float
			(RA, Dec) center of the submap box, in degrees.
		box_size_deg : float
			Size of the square submap box, in degrees.
		map_keys : list of str, optional
			Subset of self.maps keys to extract. If None, use all maps.
		"""
		self.submaps = {}

		ra, dec = box_center
		half_size = box_size_deg / 2.
		box = np.deg2rad(np.array([
			[dec - half_size, ra + half_size],
			[dec + half_size, ra - half_size]
		]))

		if map_keys is None:
			map_keys = list(self.maps.keys())

		for key in map_keys:
			self.submaps[key] = enmap.submap(self.maps[key], box)



	def plot_summary(self,signal_map_vmin = None,signal_map_vmax=None):
		"""
		Plot a summary of the maps using WCS projections.

		Parameters
		----------
		signal_map_vmin : float, optional
			Minimum value for 'signal_I' map display. If None, computed automatically.
		signal_map_vmax : float, optional
			Maximum value for 'signal_I' map display. If None, computed automatically.
		"""


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
	"""
	Generate a circular mask centered on a given sky coordinate.

	Parameters
	----------
	testmap : enmap
		Map defining the WCS and geometry.
	radius_deg : float
		Radius of the mask (in degrees).
	centervals : tuple of float, optional
		(RA, Dec) center in degrees. If None, uses the center of the map.
	apod_width : float, optional
		Width (in degrees) of apodization taper. If None, no apodization.

	Returns
	-------
	mask_map : enmap
		A binary or apodized mask in enmap format.
	"""

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
	"""
	Generate 2D Cartesian and polar coordinate grids over a square aperture.

	Parameters
	----------
	N : int
		Number of pixels along each axis (output arrays will be N x N).
	L : float
		Physical size of the aperture (in same units as desired output), spanning [-L/2, L/2] in both x and y.

	Returns
	-------
	x : ndarray
		2D array of x-coordinates, shape (N, N).
	y : ndarray
		2D array of y-coordinates, shape (N, N).
	r : ndarray
		2D array of radial distances from the center, same shape as x.
	phi : ndarray
		2D array of polar angles (in radians), measured counter-clockwise from +x axis.
	"""
	x,y = np.meshgrid(np.linspace(-L/2,L/2,N),  ## cartesian coordinates
					  np.linspace(-L/2,L/2,N))
	r = np.sqrt(x**2 + y**2)                    ## radial coordainte
	phi = np.arctan2(y,x)
	return(x,y,r,phi)

def gaussian(rarray,sig):
	"""
	Evaluate a normalized 2D circular Gaussian (without amplitude prefactor) over a radial array.

	Parameters
	----------
	rarray : ndarray or float
		Radial distance(s) from the center, typically in the same units as sigma.
	sig : float
		Standard deviation (σ) of the Gaussian.

	Returns
	-------
	ndarray or float
		Value(s) of the Gaussian evaluated at each radius in rarray.

	Notes
	-----
	This returns only the exponential part of a 2D circular Gaussian:
		exp(-0.5 * r^2 / σ^2)
	The normalization factor (1 / (σ * sqrt(2π))) is omitted.
	"""
	return np.exp(-0.5*rarray**2/sig**2)#(1./(sig*np.sqrt(2.*np.pi)))*np.exp(-0.5*rarray**2/sig**2)
def edge_taper(rarray,diameter):
	tmparray = np.zeros(rarray.shape)
	tmparray[np.where(rarray<diameter/2.)]=1.
	return tmparray


def build_tangent_wcs(n, m, pixscale_deg):
	"""
	Construct a TAN WCS centered on an (n x m) array.

	Parameters
	----------
	n : int
		Number of pixels along the y-axis (rows).
	m : int
		Number of pixels along the x-axis (columns).
	pixscale_deg : float
		Pixel scale in degrees/pixel.

	Returns
	-------
	w : astropy.wcs.WCS
		WCS object with TAN projection.
	"""
	w = WCS(naxis=2)
	w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
	w.wcs.crval = [0.0, 0.0]  # reference coordinates in degrees

	# Reference pixel is the center of the array (1-based for FITS/WCS)
	w.wcs.crpix = [m / 2 + 0.5, n / 2 + 0.5]

	# Pixel scale matrix (CDELT in degrees per pixel)
	w.wcs.cdelt = [-pixscale_deg, pixscale_deg]  # flip RA axis to follow sky convention

	return w


def Fraunhofer(A,wavelength,delta_x):
	"""
	Compute the Fraunhofer diffraction pattern of a 2D aperture field.

	Parameters
	----------
	A : 2D ndarray
		Complex or real-valued aperture field (e.g., electric field across an aperture).
	wavelength : float
		Wavelength of the incident wave, in the same units as delta_x.
	delta_x : float
		Physical size of one pixel in the aperture plane (e.g., in meters).

	Returns
	-------
	angular_width : float
		Angular size of one pixel in the diffraction (far-field) pattern, in arcminutes.
	U : 2D ndarray
		Fraunhofer diffraction pattern (Fourier transform of the aperture field), with zero frequency centered.

	Notes
	-----
	This uses the 2D Fourier transform to compute the far-field (Fraunhofer) diffraction pattern:
		U = FFT2[ A(x, y) ]

	The angular scale is computed as:
		θ = λ / Δx,
	which gives radians per pixel. This is converted to arcminutes.
	"""
	U = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(A)))
	angular_width = wavelength / delta_x  ## this is in radians
	angular_width *= 180./np.pi       ## this converts to degrees

	return(angular_width,U)

def Convert_field_to_PSF(U):
	"""
	Convert a complex field (e.g., Fraunhofer diffraction pattern) to a point spread function (PSF).

	Parameters
	----------
	U : 2D ndarray (complex)
		Complex-valued field in the far-field (e.g., output of a Fraunhofer transform).

	Returns
	-------
	PSF : 2D ndarray (float)
		Power pattern (intensity) computed as the squared magnitude of U.
	PSF_dB : 2D ndarray (float)
		Logarithmic version of the PSF in decibels (dB), computed as:
			10 * log10(PSF + ε),
		where ε = 1e-25 is added to prevent log(0) errors.

	Notes
	-----
	The PSF represents the power distribution in the image (or far-field) plane.
	The dB version enhances dynamic range visibility, particularly for faint sidelobes.
	"""
	PSF = np.abs(U)**2
	PSF_temp= PSF 
	## make a logrythmic version of the radiated power so we can better see faint features
	PSF_temp +=1e-25 ## avoid infinities by regularizing zeroes
	PSF_dB = 10*np.log10(PSF_temp)
	return(PSF, PSF_dB)

def radial_poly(n,m,rho):
	"""
	Compute the radial component of the Zernike polynomial.

	Parameters
	----------
	n : int
		Radial order of the Zernike polynomial (non-negative integer).
	m : int
		Azimuthal frequency. Must satisfy |m| ≤ n and (n - |m|) even.
	rho : ndarray or float
		Radial coordinate(s), normalized to the unit disk (0 ≤ rho ≤ 1).

	Returns
	-------
	radial_poly : ndarray or float
		Values of the radial Zernike polynomial Rₙ^|m|(rho). Returns 0 if (n - |m|) is odd.

	Notes
	-----
	The radial Zernike polynomial is defined as:

		Rₙ^m(ρ) = ∑ₖ=0^[(n−|m|)/2] [(-1)^k * (n−k)!] /
				  [k! ((n+|m|)/2 − k)! ((n−|m|)/2 − k)!] * ρ^(n − 2k)

	This function returns zeros if the Zernike polynomial is undefined due to (n - |m|) being odd.
	"""
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
	"""
	Compute the Kronecker delta δₘₙ.

	Parameters
	----------
	m : int
		First index.
	n : int
		Second index.

	Returns
	-------
	int
		1 if m == n, else 0.

	Notes
	-----
	The Kronecker delta δₘₙ is defined as:
		δₘₙ = 1 if m = n,
			  0 otherwise.
	It is commonly used in summations and orthogonality conditions.
	"""
	if m==n:
		return 1
	else:
		return 0
def zern_normalization(n,m):
	"""
	Compute the normalization factor for Zernike polynomials.

	Parameters
	----------
	n : int
		Radial order of the Zernike polynomial.
	m : int
		Azimuthal frequency of the Zernike polynomial.

	Returns
	-------
	float
		Normalization constant such that the Zernike polynomials are orthonormal
		over the unit disk.

	Notes
	-----
	The normalization factor is given by:

		Nₙₘ = √[ (2(n + 1)) / (1 + δₘ₀) ]

	where δₘ₀ is the Kronecker delta. This ensures:

		∬ Zₙₘ(ρ, φ)² ρ dρ dφ = 1  over the unit disk.

	For m ≠ 0, the factor is √(2(n + 1)); for m = 0, it is √(n + 1).
	"""
	return np.sqrt((2.*(n+1))/(1+kron_delta(m,0)))
	

def zernike_poly(n,m,rho,phi):
	"""
	Evaluate the Zernike polynomial Zₙᵐ(ρ, φ) over a unit disk.

	Parameters
	----------
	n : int
		Radial order of the Zernike polynomial (n ≥ 0).
	m : int
		Azimuthal frequency (can be positive, negative, or zero).
		Must satisfy |m| ≤ n and (n - |m|) even.
	rho : ndarray
		Radial coordinate(s), normalized to [0, 1]. Should match shape of `phi`.
	phi : ndarray
		Angular coordinate(s), in radians. Should match shape of `rho`.

	Returns
	-------
	zern : ndarray
		Normalized Zernike polynomial evaluated at (ρ, φ).

	Notes
	-----
	The Zernike polynomial is defined as:

		Zₙᵐ(ρ, φ) = Rₙ^{|m|}(ρ) × cos(mφ)   for m > 0
					Rₙ^{|m|}(ρ) × sin(|m|φ) for m < 0
					Rₙ^0(ρ)                for m = 0

	where Rₙ^{|m|}(ρ) is the radial Zernike polynomial.

	The result is normalized such that:

		∬ Zₙᵐ(ρ, φ)² ρ dρ dφ = 1  over the unit disk.

	To prevent artifacts near the disk edge, the output is set to 0 where ρ > 0.9999.
	"""
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
	"""
	Generate all normalized Zernike polynomials Zₙᵐ(ρ, φ) up to radial order `n` 
	and azimuthal frequency `m`.

	Parameters
	----------
	n : int
		Maximum radial order to compute. Must be ≥ 0.
	m : int
		Maximum azimuthal frequency. Must satisfy (n - m) even.
	rho : ndarray
		2D array of radial coordinates, normalized to [0, 1].
	phi : ndarray
		2D array of angular coordinates, in radians. Must have the same shape as `rho`.

	Returns
	-------
	zernike_array : ndarray
		3D array of shape (N, Ny, Nx), where each slice [k, :, :] is a Zernike polynomial.
		N = ((n(n+2)) + m)/2 + 1. The ordering follows OSA/Fringe conventions.

	Notes
	-----
	- This function constructs all valid Zernike polynomials (even (n - |m|)) 
	  with radial order 0 ≤ n' ≤ n and azimuthal index -n' ≤ m' ≤ n'.
	- Polynomials are indexed using the fringe (Noll-like) index:
		  j = (n(n+2) + m) / 2
	  which uniquely maps each (n, m) pair to an integer index.
	- The result is normalized such that:
		  ∬ Zₙᵐ(ρ, φ)² ρ dρ dφ = 1  over the unit disk.
	- If (n - m) is odd, the function returns False and prints a warning.
	"""
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
	"""
	Generate the wavefront defocus profile for a classical Cassegrain telescope.

	Parameters
	----------
	r : ndarray
		Radial coordinate array (same units as `D`, typically meters), representing distance from the optical axis in the pupil plane.
	dz : float
		Physical displacement of the secondary mirror along the optical axis (in same units as `r` and `D`).
	f : float, optional
		Focal length of the primary mirror (default is 17.5 meters).
	F : float, optional
		Effective focal length of the Cassegrain system (default is 525 meters).
	D : float, optional
		Diameter of the primary mirror (default is 50 meters).

	Returns
	-------
	tmpdefocus : ndarray
		2D array representing the optical path difference (in same units as `dz`)
		introduced by defocus, clipped to zero outside the aperture (r > D/2).

	Notes
	-----
	The defocus wavefront error is computed using the geometric optics approximation
	for a classical Cassegrain telescope, based on the primary and effective focal lengths.

	The profile is clipped to zero beyond the edge of the circular aperture defined by `D/2`.

	Formula used:
		W(r) = dz × [ (1 - a²)/(1 + a²) + (1 - b²)/(1 + b²) ],
		where a = r / (2f), b = r / (2F)
	"""
	a = r/(2.*f)
	b = r/(2.*F)
	tmpdefocus = dz*(((1-a**2)/(1+a**2))+((1-b**2)/(1+b**2)))
	tmpdefocus[np.where(r>(D/2.))]=0
	return tmpdefocus

def gen_phase_error_secondary_lat_displacement(x,y,del_x,del_y,f=17.5,F=525.,D=50.):
	r = np.sqrt(x**2+y**2)
	phi = np.arctan2(y,x)

	sin_theta_p = (r/f)/((1+(r/(2.*f))**2))
	sin_theta_f = (r/F)/((1+(r/(2.*F))**2))

	tmpphaseerror = -((del_x*np.cos(phi))+(del_y*np.sin(phi)))*(sin_theta_p-sin_theta_f)
	tmpphaseerror[np.where(r>(D/2.))]=0
	return tmpphaseerror

def gen_phase_error_secondary_tilt(x,y,del_alph_x,del_alph_y,f=17.5,F=525.,c_minus_a=0.8548,D=50.):
	r = np.sqrt(x**2+y**2)
	phi = np.arctan2(y,x)

	sin_theta_p = (r/f)/((1+(r/(2.*f))**2))
	sin_theta_f = (r/F)/((1+(r/(2.*F))**2))

	M = F/f

	del_alph_x_rad = np.deg2rad(del_alph_x)
	del_alph_y_rad = np.deg2rad(del_alph_y)

	tmpphaseerror = ((del_alph_x_rad*np.sin(phi))-(del_alph_y_rad*np.cos(phi)))*c_minus_a*(sin_theta_p+(M*sin_theta_f))
	tmpphaseerror[np.where(r>(D/2.))]=0
	return tmpphaseerror

def get_M1zernike_from_tel(path2tel):
	tmpnc = nc.Dataset(path2tel)
	zernikes_array = tmpnc['Header.M1.ZernikeC'][:][:]

	zernikes_dict = {}

	zernike_labels = np.array(['AST_V','AST_O','COMA_H','COMA_V','TRE_O','TRE_V','SPH',
							   'QUAD_V','QUAD_O','AST2_O','AST2_V'])
	zernike_values = np.array([zernikes_array[0],zernikes_array[1],zernikes_array[2],zernikes_array[3],zernikes_array[4],zernikes_array[5],zernikes_array[6],
							   zernikes_array[9],zernikes_array[10],zernikes_array[8],zernikes_array[7]])
	zernikes_dict['labels']=zernike_labels
	zernikes_dict['values']=zernike_values
	# zernikes_dict['AST_V'] = zernikes_array[0]
	# zernikes_dict['AST_O'] = zernikes_array[1]
	# zernikes_dict['COMA_H'] = zernikes_array[2]
	# zernikes_dict['COMA_V'] = zernikes_array[3]
	# zernikes_dict['TRE_O'] = zernikes_array[4]
	# zernikes_dict['TRE_V'] = zernikes_array[5]
	# zernikes_dict['SPH'] = zernikes_array[6]
	# zernikes_dict['AST2_V'] = zernikes_array[7]
	# zernikes_dict['AST2_O'] = zernikes_array[8]
	# zernikes_dict['QUAD_V'] = zernikes_array[9]
	# zernikes_dict['QUAD_O'] = zernikes_array[10]
	# zernikes_dict['COMA2_H'] = zernikes_array[11]
	# zernikes_dict['COMA2_V'] = zernikes_array[12]
	# zernikes_dict['TRE2_O'] = zernikes_array[13]
	# zernikes_dict['TRE2_V'] = zernikes_array[14]
	# zernikes_dict['PEN_O'] = zernikes_array[15]
	# zernikes_dict['PEN_V'] = zernikes_array[16]
	# zernikes_dict['SPH2'] = zernikes_array[17]

	tmpnc.close()

	return zernikes_dict

def get_M2z_from_tel(path2tel):
	tmpnc = nc.Dataset(path2tel)
	m2z = float(tmpnc['Header.M2.ZReq'][:].data)
	tmpnc.close()

	return m2z


# class Fraunhofer_Beamfit:
# 	"""
# 	A class to model and fit far-field beam patterns using Fraunhofer diffraction,
# 	Zernike polynomials, and real input maps (e.g., from TolTEC/Citlali).

# 	This class supports the simulation of PSFs through aperture illumination,
# 	phase aberrations, and Cassegrain defocus, and fits observed beam maps
# 	to recover surface errors and misalignments.

# 	Parameters
# 	----------
# 	paths2files : list of str
# 		List of FITS file paths to beam maps.
# 	wavelength : float
# 		Observing wavelength in meters.
# 	mask_radius : float, optional
# 		Radius (in degrees) of the mask used to isolate the beam (default: 2 arcmin).
# 	map_center : list of float, optional
# 		[RA, Dec] of the map center in degrees (default: [0., 0.]).
# 	padpixels : int, optional
# 		Number of pixels to pad maps before beam truncation.
# 	inputfitsfileformat : str, optional
# 		Format of input FITS files. Currently supports 'citlali'.
# 	"""
# 	def __init__(self,paths2files,wavelength,mask_radius=2./60.,map_center = [0.,0.],padpixels = None,inputfitsfileformat='citlali'):
		
# 		# this class fits one wavelength at a time
			
# 		self.wavelength = wavelength
		
# 		self.surface_error = {}
		
# 		#load in the maps 
# 		#only make this work with citlali maps
# 		tmpmapdict = {}
# 		signalmaps = {}
# 		mapnums = np.array(len(paths2files))
# 		for i,path in enumerate(paths2files):
# 			tmpmap = CitlaliMaps(path)
# 			tmpmapdict['map'+str(i)] = tmpmap
# 			signalmaps['map'+str(i)] = tmpmap.maps['signal_I']

# 		for i in signalmaps:
# 			tmpmap = signalmaps[i]
# 			if tmpmap.wcs.wcs.ctype[0]=='AZOFFSET':
# 				tmpmap.wcs.wcs.ctype[0] = 'RA---TAN'
# 				tmpmap.wcs.wcs.cdelt[0] = tmpmap.wcs.wcs.cdelt[0]/3600.
# 				tmpmap.wcs.wcs.cunit[0] = 'deg'
# 			if tmpmap.wcs.wcs.ctype[1]=='ELOFFSET':
# 				tmpmap.wcs.wcs.ctype[1] = 'DEC--TAN'
# 				tmpmap.wcs.wcs.cdelt[1] = tmpmap.wcs.wcs.cdelt[1]/3600.
# 				tmpmap.wcs.wcs.cunit[1] = 'deg'
# 			signalmaps[i] = tmpmap

# 		for i in signalmaps:
# 			tmpmap = signalmaps[i]
# 			tmpmapproj = enmap.project(tmpmap,signalmaps['map0'].shape,signalmaps['map0'].wcs)
# 			signalmaps[i] = tmpmapproj

# 		for i in signalmaps:
# 			tmpmap = signalmaps[i]
# 			if padpixels is not None:
# 				tmpmap = enmap.pad(tmpmap,padpixels)
# 			signalmaps[i] = tmpmap

		
# 		self.original_maps = signalmaps

# 		tmpmapload = signalmaps['map0']
# 		self.map_center = SkyCoord(ra=map_center[0]*u.deg,dec=map_center[1]*u.deg)
# 		map_mask_tmp= make_mask_enmap(tmpmapload,mask_radius,centervals = map_center,apod_width = None)
# 		self.map_mask = map_mask_tmp
		
		
# 		maskinner = make_mask_enmap(tmpmapload,1./60.,centervals = map_center,apod_width = None)
# 		noisering = tmpmapload*(map_mask_tmp-maskinner)
# 		noise_values_tmp = np.sqrt(np.mean(noisering**2))
# 		self.noise_values = noise_values_tmp

# 	def truncate_maps(self,desired_deltax_size,center_on_brightest_pix=True):
# 		"""
# 		Truncate the input maps to a square region around the beam center.

# 		Parameters
# 		----------
# 		desired_deltax_size : float
# 			Physical size (in wavelengths) to use for truncation box.
# 		center_on_brightest_pix : bool, optional
# 			Whether to center cutout on peak pixel or map center.
# 		"""
# 		self.newmapwcs = {}
# 		self.trunc_maps = {}
# 		self.peak_pixel = {}
# 		self.N = {}
# 		#mapsize = int(wavelength/(np.deg2rad(self.trunc_map.wcs.wcs.cdelt[1])*desired_deltax_size))
		
# 		for i in self.original_maps:

# 			masked_map = self.original_maps[i]*self.map_mask
# 			brightestpix = np.where(masked_map==np.amax(masked_map))
# 			self.peak_pixel[i] = np.amax(masked_map)
# 			N_tmp = int(self.wavelength/(np.deg2rad(abs(self.original_maps[i].wcs.wcs.cdelt[1]))*desired_deltax_size))
# 			self.N[i] = N_tmp
# 			if center_on_brightest_pix:
# 				tmpcoords = enmap.pix2sky(masked_map.shape, masked_map.wcs, brightestpix)
# 				center_skycoord = SkyCoord(ra=tmpcoords[1]*u.rad,dec=tmpcoords[0]*u.rad)
# 			else:
# 				center_skycoord = self.map_center
# 			tmp = Cutout2D(masked_map, center_skycoord, [N_tmp,N_tmp], wcs=masked_map.wcs)
# 			self.newmapwcs[i] = tmp.wcs
# 			self.trunc_maps[i] = enmap.enmap(tmp.data,wcs=tmp.wcs)


# 	def set_LMT_aperture(self,include_legs=False,plot_aperture=False):
# 		"""
# 		Create a circular aperture mask representing the LMT primary mirror.

# 		Parameters
# 		----------
# 		include_legs : bool
# 			Whether to simulate support legs (quadrupod shadows).
# 		plot_aperture : bool
# 			If True, show a plot of the resulting aperture field.
# 		"""
		
# 		wavelength = self.wavelength
# 		delta_x = abs(wavelength/np.deg2rad(self.trunc_maps['map0'].wcs.wcs.cdelt[1]*self.trunc_maps['map0'].shape[1]))
# 		self.delta_x = delta_x
# 		#print('The Pixel Size in the Aperture Plane is ',delta_x, ' meters')
# 		L = self.N['map0']*delta_x
# 		diam_primary = 50. ## the diameter of the primary in meters
# 		diam_secondary = 2.5 # diameter of the secondary in meters
# 		legwidths = 0.5#0.125 # quadrupod leg width in meters
# 		quadrupod_diam = 31. # diameter of circle defined by secondary suport
# 		###  make the coordinate grid
# 		x,y,r,phi = make_coordinate_grids(self.N['map0'],L)

# 		self.x = x
# 		self.y = y
# 		self.r = r
# 		self.phi = phi
# 		#figure out the legs better
# 		xbins = x[0,:]
# 		ybins = y[:,0]
# 		self.xbins = xbins
# 		self.ybins = ybins
# 		origx1 = x[np.where(np.logical_and(r<quadrupod_diam/2.,np.abs(x)<legwidths))]
# 		origy1 = y[np.where(np.logical_and(r<quadrupod_diam/2.,np.abs(x)<legwidths))]
# 		origx2 = x[np.where(np.logical_and(r<quadrupod_diam/2.,np.abs(y)<legwidths))]
# 		origy2 = y[np.where(np.logical_and(r<quadrupod_diam/2.,np.abs(y)<legwidths))]
# 		theta_rot = np.deg2rad(45.)
		
# 		rotx1 = (np.cos(theta_rot)*origx1)-(np.sin(theta_rot)*origy1)
# 		roty1 = (np.sin(theta_rot)*origx1)+(np.cos(theta_rot)*origy1)
		
# 		rotx1_digitize = np.digitize(rotx1, xbins)
# 		roty1_digitize = np.digitize(roty1, ybins)
		
# 		rotx2 = (np.cos(theta_rot)*origx2)-(np.sin(theta_rot)*origy2)
# 		roty2 = (np.sin(theta_rot)*origx2)+(np.cos(theta_rot)*origy2)
		
# 		rotx2_digitize = np.digitize(rotx2, xbins)
# 		roty2_digitize = np.digitize(roty2, ybins)
		
# 		## make the aperature fields THIS IS COMMON FOR ALL WAVELENGTHS
		
# 		A = np.ones([self.N['map0'],self.N['map0']])
# 		A[np.where(r>diam_primary/2)] = 0
# 		A[np.where(r<diam_secondary/2.)] = 0
		
# 		if include_legs:
# 			A[roty1_digitize,rotx1_digitize] = 0
# 			A[roty2_digitize,rotx2_digitize] = 0
		
# 		self.Aperture = A
# 		self.L = L
		
# 		## plot the aperature fields
# 		if plot_aperture:
# 			plt.figure()
# 			plt.imshow(A,extent=([-L/2, L/2, -L/2, L/2]))
# 			plt.title("Aperature field pattern")
# 			plt.xlabel("x [m]")
# 			plt.ylabel("y [m]")
# 			plt.colorbar()
# 			plt.show()

# 	def get_zernike_polynomials(self,n,m):
# 		"""
# 		Generate Zernike polynomials over the normalized aperture.

# 		Parameters
# 		----------
# 		n : int
# 			Maximum radial order.
# 		m : int
# 			Maximum azimuthal index.
# 		"""
# 		diam_primary = 50. 
# 		self.zernike_polynomials = gen_zernike_polys(n,m,self.r/(diam_primary/2.),self.phi)
	
# 	def set_phase(self,c=None,secondary_offset=0.,del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
# 				  f=17.5,F=525.,D=50.,plot_phase=False):
# 		"""
# 		Apply a phase screen composed of Zernike coefficients and Cassegrain defocus.

# 		Parameters
# 		----------
# 		secondary_offset : float
# 			Longitudinal offset of secondary mirror in meters.
# 		c : ndarray or None
# 			Array of Zernike coefficients. If None, uses all zeros.
# 		plot_phase : bool
# 			If True, display the resulting phase map.
# 		"""
		
		
# 		if c is None:
# 			c = np.zeros(self.zernike_polynomials.shape[0])
# 		c[0] = 0
		
# 		Phi = np.zeros([self.zernike_polynomials.shape[1],self.zernike_polynomials.shape[2]])
# 		for i in range(c.size):
# 			Phi+=c[i]*self.zernike_polynomials[i,:,:]
# 		delta_phase = gen_defocus_cassegrain_telescope(self.r,secondary_offset,f=f,F=F,D=D)
# 		delta_phase2 = gen_phase_error_secondary_lat_displacement(self.x,self.y,del_x,del_y,f=f,F=F,D=D)
# 		delta_phase3 = gen_phase_error_secondary_tilt(self.x,self.y,del_alph_x,del_alph_y,f=f,F=F,c_minus_a=0.8548,D=D)


# 		self.phase = Phi+((2.*np.pi)*delta_phase/self.wavelength)+((2.*np.pi)*delta_phase2/self.wavelength)+((2.*np.pi)*delta_phase3/self.wavelength)
# 		#phase *= A
		
# 		if plot_phase:
# 			plt.figure()
# 			plt.imshow(self.phase,extent=([-self.L/2, self.L/2, -self.L/2, self.L/2]))
# 			plt.title("Phase")
# 			plt.xlabel("x [m]")
# 			plt.ylabel("y [m]")
# 			plt.colorbar()
# 			plt.show()

# 	def make_phase(self,c=None,secondary_offset=0.,del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
# 				  f=17.5,F=525.,D=50.):

# 		"""
# 		Apply a phase screen composed of Zernike coefficients and Cassegrain defocus.

# 		Parameters
# 		----------
# 		secondary_offset : float
# 			Longitudinal offset of secondary mirror in meters.
# 		c : ndarray or None
# 			Array of Zernike coefficients. If None, uses all zeros.
# 		plot_phase : bool
# 			If True, display the resulting phase map.
# 		"""
		
		
# 		if c is None:
# 			c = np.zeros(self.zernike_polynomials.shape[0])
# 		c[0] = 0
		
# 		Phi = np.zeros([self.zernike_polynomials.shape[1],self.zernike_polynomials.shape[2]])
# 		for i in range(c.size):
# 			Phi+=c[i]*self.zernike_polynomials[i,:,:]
# 		delta_phase = gen_defocus_cassegrain_telescope(self.r,secondary_offset,f=f,F=F,D=D)
# 		delta_phase2 = gen_phase_error_secondary_lat_displacement(self.x,self.y,del_x,del_y,f=f,F=F,D=D)
# 		delta_phase3 = gen_phase_error_secondary_tilt(self.x,self.y,del_alph_x,del_alph_y,f=f,F=F,c_minus_a=0.8548,D=D)


# 		tmpphase = Phi+((2.*np.pi)*delta_phase/self.wavelength)+((2.*np.pi)*delta_phase2/self.wavelength)+((2.*np.pi)*delta_phase3/self.wavelength)
# 		return tmpphase


# 	def set_illumination(self,aperture_fwhm = 48.,edge_taper_diameter=48.,plot_illumination=False):
# 		"""
# 		Define a radial Gaussian illumination function over the aperture.

# 		Parameters
# 		----------
# 		aperture_fwhm : float
# 			Full-width at half-maximum of the Gaussian illumination (in meters).
# 		edge_taper_diameter : float
# 			Diameter beyond which the illumination is zeroed.
# 		plot_illumination : bool
# 			If True, show a plot of the illumination pattern.
# 		"""

# 		sig0 = aperture_fwhm/(2*np.sqrt(2*np.log(2)))
# 		edge_taper = np.ones([self.N['map0'],self.N['map0']])
# 		edge_taper[np.where(self.r>edge_taper_diameter/2.)]=0
# 		illumination = gaussian(self.r,sig0)*edge_taper
# 		self.illumination = illumination
# 		if plot_illumination:
# 			plt.figure()
# 			plt.imshow(self.illumination,extent=([-self.L/2, self.L/2, -self.L/2, self.L/2]))
# 			plt.title("illumination")
# 			plt.xlabel("x [m]")
# 			plt.ylabel("y [m]")
# 			plt.colorbar()
# 			plt.show()

# 	def make_normalizing_amplitude(self):
# 		"""
# 		Calculate the peak PSF amplitude assuming a flat wavefront.
# 		Used to normalize later PSF models.
# 		"""
# 		A_complex = self.Aperture*self.illumination#*np.exp(self.phase*1j)
# 		angular_width,U = Fraunhofer(A_complex,self.wavelength,self.delta_x)
# 		PSF_nom = np.abs(U)**2
# 		self.normalizing_amplitude = np.amax(PSF_nom)



# 	def make_psf(self):
# 		"""
# 		Construct the modeled PSF using the current aperture, illumination, and phase.
# 		Stores a normalized enmap PSF in `self.PSF`.
# 		"""
# 		A_complex = self.Aperture*self.illumination*np.exp(self.phase*1j)
# 		angular_width,U = Fraunhofer(A_complex,self.wavelength,self.delta_x)
# 		PSF,PSF_dB = Convert_field_to_PSF(U)
# 		self.PSF = enmap.enmap(PSF/self.normalizing_amplitude,wcs=self.newmapwcs['map0'])
	
# 	def function2minimize(self,x):

# 		"""
# 		Objective function for beam model fitting.

# 		Parameters
# 		----------
# 		x : array-like
# 			Fit parameter vector including source amplitude, tilts, defocus,
# 			and Zernike coefficients.

# 		Returns
# 		-------
# 		float
# 			RMS residuals across maps 0, 1, and 2 combined.
# 		"""

# 		#list of fit params
# 		# x[0] = source amplitude
# 		# x[1] = Primary Focal Length
# 		# x[2] = M2.X offset
# 		# x[3] = M2.Y offset
# 		# x[4] = M2.Z offset

# 		# x[5] = TILT_Y
# 		# x[6] = TILT_X

# 		# x[7] = AST_O
# 		# x[8] = AST_V
# 		# x[9] = TRE_V
# 		# ...

		
# 		# make map 0 zern params


# 		c_map0 = np.zeros(self.zernike_polynomials.shape[0])
# 		c_map0[0] = 0. # piston is always 0
# 		c_map0[1] = x[5] # TILT_Y is always 0
# 		c_map0[2] = x[6] # TILT_X is always 0
# 		c_map0[3] = x[7]
# 		c_map0[4] = 0 # FOCUS is always 0
# 		c_map0[5:] = x[8:]

# 		# make map 1 zern params

# 		c_map1 = np.zeros(self.zernike_polynomials.shape[0])
# 		c_map1[0] = 0. # piston is always 0
# 		c_map1[1] = x[5] # TILT_Y is always 0
# 		c_map1[2] = x[6] # TILT_X is always 0
# 		c_map1[3] = x[7]
# 		c_map1[4] = 0 # FOCUS is always 0
# 		c_map1[5:] = x[8:]

# 		# make map 2 zern params

# 		c_map2 = np.zeros(self.zernike_polynomials.shape[0])
# 		c_map2[0] = 0. # piston is always 0
# 		c_map2[1] = x[5] # TILT_Y is always 0
# 		c_map2[2] = x[6] # TILT_X is always 0
# 		c_map2[3] = x[7]
# 		c_map2[4] = 0 # FOCUS is always 0
# 		c_map2[5:] = x[8:]



# 		# make map 0 psf
	

# 		phase0 = self.make_phase(c=c_map0,secondary_offset=x[4]-1.E-3,del_x=x[3],del_y=x[4],del_alph_x=0.,del_alph_y=0.,
# 								  f=x[1],F=525.,D=50.)

# 		A_complex_map0 = self.Aperture*self.illumination*np.exp(phase0*1j)
# 		angular_width_map0,U_map0 = Fraunhofer(A_complex_map0,self.wavelength,self.delta_x)
# 		PSF_map0,PSF_dB = Convert_field_to_PSF(U_map0)
# 		model_map0 = enmap.enmap(x[0]*(PSF_map0/self.normalizing_amplitude),wcs=self.newmapwcs['map0'])
		
# 		resids_map0 = self.trunc_maps['map0'] - model_map0
# 		rms_map0 = np.sqrt(np.mean(resids_map0**2))

# 		# make map 1 psf
	
# 		phase1 = self.make_phase(c=c_map1,secondary_offset=x[4],del_x=x[3],del_y=x[4],del_alph_x=0.,del_alph_y=0.,
# 								  f=x[1],F=525.,D=50.)

# 		A_complex_map1 = self.Aperture*self.illumination*np.exp(phase1*1j)
# 		angular_width_map1,U_map1 = Fraunhofer(A_complex_map1,self.wavelength,self.delta_x)
# 		PSF_map1,PSF_dB = Convert_field_to_PSF(U_map1)
# 		model_map1 = enmap.enmap(x[0]*(PSF_map1/self.normalizing_amplitude),wcs=self.newmapwcs['map0'])
		
# 		resids_map1 = self.trunc_maps['map1'] - model_map1
# 		rms_map1 = np.sqrt(np.mean(resids_map1**2))

# 		# make map 2 psf
	
# 		phase2 = self.make_phase(c=c_map2,secondary_offset=x[4]+1E-3,del_x=x[3],del_y=x[4],del_alph_x=0.,del_alph_y=0.,
# 								  f=x[1],F=525.,D=50.)
# 		A_complex_map2 = self.Aperture*self.illumination*np.exp(phase2*1j)
# 		angular_width_map2,U_map2 = Fraunhofer(A_complex_map2,self.wavelength,self.delta_x)
# 		PSF_map2,PSF_dB = Convert_field_to_PSF(U_map2)
# 		model_map2 = enmap.enmap(x[0]*(PSF_map2/self.normalizing_amplitude),wcs=self.newmapwcs['map0'])
		
# 		resids_map2 = self.trunc_maps['map2'] - model_map2
# 		rms_map2 = np.sqrt(np.mean(resids_map2**2))
	
		
# 		chisquare = np.sqrt(rms_map0**2+rms_map1**2+rms_map2**2)
# 		self.cost = np.sum(chisquare)
# 		self.fit_step_counter+=1
# 		if self.fit_step_counter%500==0:
# 			print('On fit step ',self.fit_step_counter,' with cost ',np.sum(chisquare))
# 		return chisquare
		
# 	def fit_beam(self,c_guess=None,boundvals = None):

# 		"""
# 		Fit the beam model to the input maps by optimizing Zernike phase terms
# 		and geometric defocus.

# 		Parameters
# 		----------
# 		c_guess : array-like, optional
# 			Initial guess for Zernike coefficients.
# 		boundvals : list of tuples
# 			Bounds for optimization variables, passed to scipy.optimize.minimize.
# 		"""

# 		#list of fit params
# 		# x[0] = source amplitude
# 		# x[1] = Primary Focal Length
# 		# x[2] = M2.X offset
# 		# x[3] = M2.Y offset
# 		# x[4] = M2.Z offset

# 		# x[5] = TILT_Y
# 		# x[6] = TILT_X

# 		# x[7] = AST_O
# 		# x[8] = AST_V
# 		# x[9] = TRE_V
# 		# ...


# 		x0 = np.zeros(self.zernike_polynomials.shape[0]+3) # need to make an array that has the zernike coeffs plus 1 for theta0
# 		x0[0] = np.amax(self.trunc_maps['map1'])
# 		x0[1] = 17.5 # guess for primary focal length
		
# 		if c_guess is not None:
# 			x0[8:] = c_guess
		
# 		self.fit_step_counter = 0
# 		results = minimize(self.function2minimize,x0,bounds=boundvals)
# 		self.results = results
# 		if results.success:
# 			print('The fit was successful. Cost =', self.cost)
# 		else:
# 			print('The fit was not successful, you may need to run again with different guess or bounds. Cost =', self.cost)


# 		c_bestfit_map0 = np.zeros(self.zernike_polynomials.shape[0])
# 		c_bestfit_map0[1] = results.x[5]
# 		c_bestfit_map0[2] = results.x[6]
# 		c_bestfit_map0[3] = results.x[7]
# 		c_bestfit_map0[5:] = results.x[8:]

# 		c_bestfit_map1 = np.zeros(self.zernike_polynomials.shape[0])
# 		c_bestfit_map1[1] = results.x[5]
# 		c_bestfit_map1[2] = results.x[6]
# 		c_bestfit_map1[3] = results.x[7]
# 		c_bestfit_map1[5:] = results.x[8:]

# 		c_bestfit_map2 = np.zeros(self.zernike_polynomials.shape[0])
# 		c_bestfit_map2[1] = results.x[5]
# 		c_bestfit_map2[2] = results.x[6]
# 		c_bestfit_map2[3] = results.x[7]
# 		c_bestfit_map2[5:] = results.x[8:]

# 		c_bestfit_ideal = np.zeros(self.zernike_polynomials.shape[0])
# 		c_bestfit_ideal[3] = results.x[7]
# 		c_bestfit_ideal[5:] = results.x[8:]

# 		# make bestfitmaps for resids
# 		self.bestfit_maps = {}

# 		phase0_fit = self.make_phase(c=c_bestfit_map0,secondary_offset=results.x[4]-1.E-3,del_x=results.x[3],del_y=results.x[4],del_alph_x=0.,del_alph_y=0.,
# 								  f=results.x[1],F=525.,D=50.)

# 		A_complex = self.Aperture*self.illumination*np.exp(phase0_fit*1j)
# 		angular_width,U = Fraunhofer(A_complex,self.wavelength,self.delta_x)
# 		PSF,PSF_dB = Convert_field_to_PSF(U)
# 		tmppsf_raw = enmap.enmap(results.x[0]*(PSF/self.normalizing_amplitude),wcs=self.newmapwcs['map0'])
# 		self.bestfit_maps['map0'] = tmppsf_raw

# 		phase1_fit = self.make_phase(c=c_bestfit_map1,secondary_offset=results.x[4],del_x=results.x[3],del_y=results.x[4],del_alph_x=0.,del_alph_y=0.,
# 								  f=results.x[1],F=525.,D=50.)
# 		A_complex = self.Aperture*self.illumination*np.exp(phase1_fit*1j)
# 		angular_width,U = Fraunhofer(A_complex,self.wavelength,self.delta_x)
# 		PSF,PSF_dB = Convert_field_to_PSF(U)
# 		tmppsf_raw = enmap.enmap(results.x[0]*(PSF/self.normalizing_amplitude),wcs=self.newmapwcs['map0'])
# 		self.bestfit_maps['map1'] = tmppsf_raw

# 		phase2_fit = self.make_phase(c=c_bestfit_map2,secondary_offset=results.x[4]+1.E-3,del_x=results.x[3],del_y=results.x[4],del_alph_x=0.,del_alph_y=0.,
# 								  f=results.x[1],F=525.,D=50.)
# 		A_complex = self.Aperture*self.illumination*np.exp(phase2_fit*1j)
# 		angular_width,U = Fraunhofer(A_complex,self.wavelength,self.delta_x)
# 		PSF,PSF_dB = Convert_field_to_PSF(U)
# 		tmppsf_raw = enmap.enmap(results.x[0]*(PSF/self.normalizing_amplitude),wcs=self.newmapwcs['map0'])
# 		self.bestfit_maps['map2'] = tmppsf_raw


# 		self.zern_coefficients = c_bestfit_ideal
		
# 		Phi = np.zeros([self.zernike_polynomials.shape[1],self.zernike_polynomials.shape[2]])
# 		for i in range(c_bestfit_ideal.size):
# 			Phi+=c_bestfit_ideal[i]*self.zernike_polynomials[i,:,:]

# 		phase_ideal_fit = Phi
# 		self.phase = phase_ideal_fit
		
# 		A_complex = self.Aperture*self.illumination*np.exp(phase_ideal_fit*1j)
# 		angular_width,U = Fraunhofer(A_complex,self.wavelength,self.delta_x)
# 		PSF,PSF_dB = Convert_field_to_PSF(U)
# 		tmppsf_raw = enmap.enmap(results.x[0]*(PSF/self.normalizing_amplitude),wcs=self.newmapwcs['map0'])
# 		gain_loss = np.amax(tmppsf_raw/results.x[0])
# 		self.gain_loss = gain_loss

# 		self.bestfitbeam = tmppsf_raw

# 	def plot_phase(self,plot_vmin=None,plot_vmax=None,save_fig_name=None,noshow=False):
# 		"""
# 		Plot the current phase screen.

# 		Parameters
# 		----------
# 		plot_vmin, plot_vmax : float, optional
# 			Color limits.
# 		save_fig_name : str, optional
# 			Path to save the figure.
# 		noshow : bool
# 			If True, suppress display (for script automation).
# 		"""
# 		plt.figure()
# 		plt.imshow(self.phase,vmin=plot_vmin,vmax=plot_vmax,extent=([-self.L/2, self.L/2, -self.L/2, self.L/2]))
# 		plt.colorbar()
		
# 		plt.title("Phase")
# 		plt.xlabel("x [m]")
# 		plt.ylabel("y [m]")
		
# 		plt.xlim(-30,30)
# 		plt.ylim(-30,30)
# 		if save_fig_name is not None:
# 			plt.savefig(save_fig_name,facecolor='white',transparent=False,bbox_inches='tight')
# 		if noshow:
# 			plt.close()
# 		else:
# 			plt.show()

# 	def plot_surface_error(self,plot_vmin=None,plot_vmax=None,save_fig_name=None,noshow=False):
# 		"""
# 		Plot the inferred surface error (in microns) from the fitted phase screen.

# 		Parameters
# 		----------
# 		plot_vmin, plot_vmax : float, optional
# 			Color limits.
# 		save_fig_name : str, optional
# 			Path to save the figure.
# 		noshow : bool
# 			If True, suppress display.
# 		"""
# 		tmpwavelength = self.wavelength

# 		phase_primary_only = self.make_phase(c=self.zern_coefficients,secondary_offset=0.,del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
# 								  f=17.5,F=525.,D=50.)
		
# 		self.surface_error = 1E6*tmpwavelength*phase_primary_only/(2*np.pi)
# 		print('RMS = ',np.sqrt(np.mean(self.surface_error[np.nonzero(self.surface_error)]**2)),'microns')
# 		plt.figure()
# 		plt.imshow(self.surface_error,vmin=plot_vmin,vmax=plot_vmax,extent=([-self.L/2, self.L/2, -self.L/2, self.L/2]))
# 		cbar = plt.colorbar()
# 		cbar.set_label(label='microns',rotation=270,labelpad=15,fontsize=14)
		
# 		plt.title("Surface Error")
# 		plt.xlabel("x [m]")
# 		plt.ylabel("y [m]")
		
# 		plt.xlim(-30,30)
# 		plt.ylim(-30,30)
# 		if save_fig_name is not None:
# 			plt.savefig(save_fig_name,facecolor='white',transparent=False,bbox_inches='tight')
		
# 		if noshow:
# 			plt.close()
# 		else:
# 			plt.show()

# 	def plot_results(self,plot_vmin=None,plot_vmax=None,resids_vmin=None,resids_vmax=None,save_fig_name=None,noshow=False,plot_title=None,lowerleft=[-0.75/60,-0.75/60.],upperright=[0.75/60,0.75/60.]):
# 		"""
# 		Show a 3×3 panel plot of input maps, best-fit models, and residuals.

# 		Parameters
# 		----------
# 		plot_vmin, plot_vmax : float, optional
# 			Color scale limits for data/model maps.
# 		resids_vmin, resids_vmax : float, optional
# 			Color scale limits for residual maps.
# 		save_fig_name : str, optional
# 			File name to save the figure.
# 		noshow : bool
# 			If True, don't display figure.
# 		plot_title : str, optional
# 			Title to place above all subplots.
# 		lowerleft, upperright : list of float
# 			Sky coordinate box (in degrees) to show in each subplot.
# 		"""
# 		fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10), subplot_kw={'projection': self.trunc_maps['map0'].wcs})


# 		# Flatten axes array for easy iteration
# 		#axes = axes.flatten()
		
# 		for i,name in enumerate(self.trunc_maps):
# 			# Set the correct WCS projection for each subplot

# 			ax = plt.subplot(3, 3, i + 1)

			
# 			img = ax.imshow(self.trunc_maps[name], origin='lower', cmap='gray', vmin=plot_vmin, vmax=plot_vmax)
# 			cbar = fig.colorbar(img, ax=ax, orientation="vertical", shrink=0.7)
# 			ax.set_title(name)
		
# 			# Set WCS coordinate labels
# 			ra = ax.coords['ra']
# 			ra.set_format_unit(u.deg, decimal=True)

# 			# 1) clear the axis label
# 			ra.set_axislabel('')

# 			# 2) hide the tick marks
# 			ra.set_ticks_visible(False)

# 			# 3) hide the tick *labels*
# 			ra.set_ticklabel_visible(False)
# 			if i%3==0:
# 				ax.set_ylabel("El Offset")
# 				dec = ax.coords['dec']
# 				dec.set_format_unit(u.arcmin, decimal=True)
# 				dec.set_major_formatter('d.dd')
# 			else:
# 				# suppose ax is one of your WCSAxes subplots
# 				dec = ax.coords['dec']

# 				# 1) clear the axis label
# 				dec.set_axislabel('')

# 				# 2) hide the tick marks
# 				dec.set_ticks_visible(False)

# 				# 3) hide the tick *labels*
# 				dec.set_ticklabel_visible(False)



# 			if lowerleft is not None:
# 				lower_left  = SkyCoord(lowerleft[0]*u.deg,  lowerleft[1]*u.deg, frame='icrs')
# 				x0, y0 = self.trunc_maps['map0'].wcs.world_to_pixel(lower_left)
# 			if upperright is not None:
# 				upper_right = SkyCoord(upperright[0]*u.deg,  upperright[1]*u.deg, frame='icrs')
# 				x1, y1 = self.trunc_maps['map0'].wcs.world_to_pixel(upper_right)

			

# 			ax.set_xlim(x0, x1)
# 			ax.set_ylim(y0, y1)
# 		for i,name in enumerate(self.bestfit_maps):
# 			# Set the correct WCS projection for each subplot

# 			ax = plt.subplot(3, 3, i + 4)

			
# 			img = ax.imshow(self.bestfit_maps[name], origin='lower', cmap='gray', vmin=plot_vmin, vmax=plot_vmax)
# 			cbar = fig.colorbar(img, ax=ax, orientation="vertical", shrink=0.7)
# 			#ax.set_title(name)
		
# 			# Set WCS coordinate labels
# 			ra = ax.coords['ra']
# 			ra.set_format_unit(u.deg, decimal=True)

# 			# 1) clear the axis label
# 			ra.set_axislabel('')

# 			# 2) hide the tick marks
# 			ra.set_ticks_visible(False)

# 			# 3) hide the tick *labels*
# 			ra.set_ticklabel_visible(False)
# 			if i%3==0:
# 				ax.set_ylabel("El Offset")
# 				dec = ax.coords['dec']
# 				dec.set_format_unit(u.arcmin, decimal=True)
# 				dec.set_major_formatter('d.dd')
# 			else:
# 				# suppose ax is one of your WCSAxes subplots
# 				dec = ax.coords['dec']

# 				# 1) clear the axis label
# 				dec.set_axislabel('')

# 				# 2) hide the tick marks
# 				dec.set_ticks_visible(False)

# 				# 3) hide the tick *labels*
# 				dec.set_ticklabel_visible(False)
# 			if lowerleft is not None:
# 				lower_left  = SkyCoord(lowerleft[0]*u.deg,  lowerleft[1]*u.deg, frame='icrs')
# 				x0, y0 = self.trunc_maps['map0'].wcs.world_to_pixel(lower_left)
# 			if upperright is not None:
# 				upper_right = SkyCoord(upperright[0]*u.deg,  upperright[1]*u.deg, frame='icrs')
# 				x1, y1 = self.trunc_maps['map0'].wcs.world_to_pixel(upper_right)

			

# 			ax.set_xlim(x0, x1)
# 			ax.set_ylim(y0, y1)

# 		for i,name in enumerate(self.bestfit_maps):
# 			# Set the correct WCS projection for each subplot

# 			ax = plt.subplot(3, 3, i + 7)

			
# 			img = ax.imshow(self.trunc_maps[name]-self.bestfit_maps[name], origin='lower', cmap='gray', vmin=resids_vmin, vmax=resids_vmax)
# 			cbar = fig.colorbar(img, ax=ax, orientation="vertical", shrink=0.7)
# 			#ax.set_title(name)
		
# 			# Set WCS coordinate labels
# 			ax.set_xlabel("AZ Offset")
# 			ra = ax.coords['ra']
# 			ra.set_format_unit(u.arcmin, decimal=True)
# 			ra.set_major_formatter('d.dd')
# 			if i%3==0:
# 				ax.set_ylabel("El Offset")
# 				dec = ax.coords['dec']
# 				dec.set_format_unit(u.arcmin, decimal=True)
# 				dec.set_major_formatter('d.dd')
# 			else:
# 				# suppose ax is one of your WCSAxes subplots
# 				dec = ax.coords['dec']

# 				# 1) clear the axis label
# 				dec.set_axislabel('')

# 				# 2) hide the tick marks
# 				dec.set_ticks_visible(False)

# 				# 3) hide the tick *labels*
# 				dec.set_ticklabel_visible(False)
# 			if lowerleft is not None:
# 				lower_left  = SkyCoord(lowerleft[0]*u.deg,  lowerleft[1]*u.deg, frame='icrs')
# 				x0, y0 = self.trunc_maps['map0'].wcs.world_to_pixel(lower_left)
# 			if upperright is not None:
# 				upper_right = SkyCoord(upperright[0]*u.deg,  upperright[1]*u.deg, frame='icrs')
# 				x1, y1 = self.trunc_maps['map0'].wcs.world_to_pixel(upper_right)

			

# 			ax.set_xlim(x0, x1)
# 			ax.set_ylim(y0, y1)
# 		plt.tight_layout()
# 		plt.subplots_adjust(hspace=0.1,wspace=-0.4) 
# 		if plot_title is not None:
# 			plt.suptitle(plot_title)
		
# 		if save_fig_name is not None:
# 			plt.savefig(save_fig_name,facecolor='white',transparent=False,bbox_inches='tight')
		
# 		if noshow:
# 			plt.close()
# 		else:
# 			plt.show()

# 	def plot_psf(self,plot_vmin=None,plot_vmax=None,save_fig_name=None,xlims=None,ylims=None):
# 		"""
# 		Display the final modeled PSF.

# 		Parameters
# 		----------
# 		plot_vmin, plot_vmax : float, optional
# 			Color limits.
# 		save_fig_name : str, optional
# 			File name to save the figure.
# 		xlims, ylims : tuple, optional
# 			Axes limits in degrees.
# 		"""
# 		corners_tmp = np.rad2deg(enmap.corners(self.PSF.shape,self.PSF.wcs))
# 		imextent_tmp = [corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]]
# 		fig=plt.figure()
# 		im = plt.imshow(self.PSF*np.amax(self.trunc_map),extent=imextent_tmp,origin='lower',vmin=plot_vmin,vmax=plot_vmax)
# 		plt.xticks(fontsize=10)
# 		plt.yticks(fontsize=10)
# 		if xlims is not None:
# 			plt.xlim(xlims)
# 		if ylims is not None:
# 			plt.ylim(ylims)
# 		plt.title('PSF')
# 		plt.colorbar()
# 		if save_fig_name is not None:
# 			plt.savefig(save_fig_name,facecolor='white',transparent=False,bbox_inches='tight')
# 		plt.show()

# 	def plot_inputmaps(self,plot_vmin=None,plot_vmax=None,save_fig_name=None,noshow=False,lowerleft=[-0.75/60,-0.75/60.],upperright=[0.75/60,0.75/60.]):
# 		"""
# 		Plot the truncated input maps used in the beam fit.

# 		Parameters
# 		----------
# 		plot_vmin, plot_vmax : float, optional
# 			Color scale limits.
# 		save_fig_name : str, optional
# 			File name to save the figure.
# 		noshow : bool
# 			If True, suppress figure display.
# 		lowerleft, upperright : list of float
# 			Coordinate box in degrees to crop view.
# 		"""
# 		fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), subplot_kw={'projection': self.trunc_maps['map0'].wcs})


# 		# Flatten axes array for easy iteration
# 		#axes = axes.flatten()
		
# 		for i,name in enumerate(self.trunc_maps):
# 			# Set the correct WCS projection for each subplot

# 			ax = plt.subplot(1, 3, i + 1)

			
# 			img = ax.imshow(self.trunc_maps[name], origin='lower', cmap='gray', vmin=plot_vmin, vmax=plot_vmax)
# 			cbar = fig.colorbar(img, ax=ax, orientation="vertical", shrink=0.6)
# 			ax.set_title(name)
		
# 			# Set WCS coordinate labels
# 			ax.set_xlabel("AZ Offset")
# 			ra = ax.coords['ra']
# 			ra.set_format_unit(u.arcmin, decimal=True)
# 			ra.set_major_formatter('d.dd')
# 			if i%3==0:
# 				ax.set_ylabel("El Offset")
# 				dec = ax.coords['dec']
# 				dec.set_format_unit(u.arcmin, decimal=True)
# 				dec.set_major_formatter('d.dd')
# 			else:
# 				# suppose ax is one of your WCSAxes subplots
# 				dec = ax.coords['dec']

# 				# 1) clear the axis label
# 				dec.set_axislabel('')

# 				# 2) hide the tick marks
# 				dec.set_ticks_visible(False)

# 				# 3) hide the tick *labels*
# 				dec.set_ticklabel_visible(False)



			
# 			lower_left  = SkyCoord(lowerleft[0]*u.deg,  lowerleft[1]*u.deg, frame='icrs')
# 			x0, y0 = self.trunc_maps['map0'].wcs.world_to_pixel(lower_left)
			
# 			upper_right = SkyCoord(upperright[0]*u.deg,  upperright[1]*u.deg, frame='icrs')
# 			x1, y1 = self.trunc_maps['map0'].wcs.world_to_pixel(upper_right)

			

# 			ax.set_xlim(x0, x1)
# 			ax.set_ylim(y0, y1)
# 		plt.tight_layout()
# 		plt.subplots_adjust(wspace=0.1) 
		
# 		if save_fig_name is not None:
# 			plt.savefig(save_fig_name,facecolor='white',transparent=False,bbox_inches='tight')
		
# 		if noshow:
# 			plt.close()
# 		else:
# 			plt.show()

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
	
	def set_phase(self,c=None,secondary_offset=0.,del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
				  f=17.5,F=525.,D=50.,plot_phase=False):
		
		
		if c is None:
			c = np.zeros(self.zernike_polynomials.shape[0])
		c[0] = 0
		
		Phi = np.zeros([self.zernike_polynomials.shape[1],self.zernike_polynomials.shape[2]])
		for i in range(c.size):
			Phi+=c[i]*self.zernike_polynomials[i,:,:]
		delta_phase = gen_defocus_cassegrain_telescope(self.r,secondary_offset,f=f,F=F,D=D)
		delta_phase2 = gen_phase_error_secondary_lat_displacement(self.x,self.y,del_x,del_y,f=f,F=F,D=D)
		delta_phase3 = gen_phase_error_secondary_tilt(self.x,self.y,del_alph_x,del_alph_y,f=f,F=F,c_minus_a=0.8548,D=D)

		self.phase = Phi+(delta_phase*2.*np.pi/self.wavelength)+(delta_phase2*2.*np.pi/self.wavelength)+(delta_phase3*2.*np.pi/self.wavelength)
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


class Fraunhofer_Beamfit:
	"""
	A class to model and fit far-field beam patterns using Fraunhofer diffraction,
	Zernike polynomials, and real input maps (e.g., from TolTEC/Citlali).

	This class supports the simulation of PSFs through aperture illumination,
	phase aberrations, and Cassegrain defocus, and fits observed beam maps
	to recover surface errors and misalignments.

	Parameters
	----------
	paths2files : list of str
		List of FITS file paths to beam maps.
	wavelength : float
		Observing wavelength in meters.
	mask_radius : float, optional
		Radius (in degrees) of the mask used to isolate the beam (default: 2 arcmin).
	map_center : list of float, optional
		[RA, Dec] of the map center in degrees (default: [0., 0.]).
	padpixels : int, optional
		Number of pixels to pad maps before beam truncation.
	inputfitsfileformat : str, optional
		Format of input FITS files. Currently supports 'citlali'.
	"""
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
			tmpmap = CitlaliMaps(path)
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
		"""
		Truncate the input maps to a square region around the beam center.

		Parameters
		----------
		desired_deltax_size : float
			Physical size (in wavelengths) to use for truncation box.
		center_on_brightest_pix : bool, optional
			Whether to center cutout on peak pixel or map center.
		"""
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
		"""
		Create a circular aperture mask representing the LMT primary mirror.

		Parameters
		----------
		include_legs : bool
			Whether to simulate support legs (quadrupod shadows).
		plot_aperture : bool
			If True, show a plot of the resulting aperture field.
		"""
		
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
	
	def set_phase(self,c=None,secondary_offset=0.,del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
				  f=17.5,F=525.,D=50.,plot_phase=False):
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


		self.phase = Phi+((2.*np.pi)*delta_phase/self.wavelength)+((2.*np.pi)*delta_phase2/self.wavelength)+((2.*np.pi)*delta_phase3/self.wavelength)
		#phase *= A
		
		if plot_phase:
			plt.figure()
			plt.imshow(self.phase,extent=([-self.L/2, self.L/2, -self.L/2, self.L/2]))
			plt.title("Phase")
			plt.xlabel("x [m]")
			plt.ylabel("y [m]")
			plt.colorbar()
			plt.show()

	def make_phase(self,c=None,secondary_offset=0.,del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
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


		tmpphase = Phi+((2.*np.pi)*delta_phase/self.wavelength)+((2.*np.pi)*delta_phase2/self.wavelength)+((2.*np.pi)*delta_phase3/self.wavelength)
		return tmpphase


	def make_psf(self,phase=None,return_psf=False):
		"""
		Construct the modeled PSF using the current aperture, illumination, and phase.
		Stores a normalized enmap PSF in `self.PSF`.
		"""
		if phase is None:
			phase = self.phase

		A_complex = self.Aperture*self.illumination*np.exp(phase*1j)

		field_maps_array = np.empty([self.bp_wavelengths.size,self.trunc_maps['map0'].shape[0],self.trunc_maps['map0'].shape[1]],dtype='complex')

		for i,wavelength in enumerate(self.bp_wavelengths):
			farfield_im_size, U = Fraunhofer(A_complex,wavelength,self.delta_x)
			tmpwcs = build_tangent_wcs(U.shape[0], U.shape[1], farfield_im_size/self.N['map0'])
			tmpUenmap_real = enmap.enmap(np.real(U),wcs=tmpwcs)
			tmpUenmap_imag = enmap.enmap(np.imag(U),wcs=tmpwcs)
			projUenmap_real = enmap.project(tmpUenmap_real,self.trunc_maps['map0'].shape,self.trunc_maps['map0'].wcs)
			projUenmap_imag = enmap.project(tmpUenmap_imag,self.trunc_maps['map0'].shape,self.trunc_maps['map0'].wcs)

			projUenmap_complex = enmap.enmap((projUenmap_real+projUenmap_imag*1.j),wcs=projUenmap_real.wcs)

			field_maps_array[i,:,:] = projUenmap_complex*self.bp_transmission[i]

		#do the integrals
		
		integrated_field = np.trapz(field_maps_array,x=self.bp_wavelengths,axis=0)
		integrated_bp = np.trapz(self.bp_transmission,x=self.bp_wavelengths,axis=0)

		far_field_map = integrated_field/integrated_bp


		PSF,PSF_dB = Convert_field_to_PSF(far_field_map)
		self.PSF = enmap.enmap(PSF/self.normalizing_amplitude,wcs=self.newmapwcs['map0'])
		if return_psf:
			return self.PSF

	def make_psf_monochromatic(self,phase=None,return_psf=False):
		"""
		Construct the modeled PSF using the current aperture, illumination, and phase.
		Stores a normalized enmap PSF in `self.PSF`.
		"""
		if phase is None:
			phase = self.phase

		A_complex = self.Aperture*self.illumination*np.exp(phase*1j)
		farfield_im_size, U = Fraunhofer(A_complex,self.wavelength,self.delta_x)

		PSF,PSF_dB = Convert_field_to_PSF(U)
		self.PSF = enmap.enmap(PSF/self.normalizing_amplitude_monochromatic,wcs=self.newmapwcs['map0'])
		if return_psf:
			return self.PSF


	def make_normalizing_amplitude(self):
		"""
		Calculate the peak PSF amplitude assuming a flat wavefront.
		Used to normalize later PSF models.
		"""
		

		A_complex = self.Aperture*self.illumination

		field_maps_array = np.empty([self.bp_wavelengths.size,self.trunc_maps['map0'].shape[0],self.trunc_maps['map0'].shape[1]],dtype='complex')

		for i,wavelength in enumerate(self.bp_wavelengths):
			farfield_im_size, U = Fraunhofer(A_complex,wavelength,self.delta_x)
			tmpwcs = build_tangent_wcs(U.shape[0], U.shape[1], farfield_im_size/self.N['map0'])
			tmpUenmap_real = enmap.enmap(np.real(U),wcs=tmpwcs)
			tmpUenmap_imag = enmap.enmap(np.imag(U),wcs=tmpwcs)
			projUenmap_real = enmap.project(tmpUenmap_real,self.trunc_maps['map0'].shape,self.trunc_maps['map0'].wcs)
			projUenmap_imag = enmap.project(tmpUenmap_imag,self.trunc_maps['map0'].shape,self.trunc_maps['map0'].wcs)

			projUenmap_complex = enmap.enmap((projUenmap_real+projUenmap_imag*1.j),wcs=projUenmap_real.wcs)

			field_maps_array[i,:,:] = projUenmap_complex*self.bp_transmission[i]

		#do the integrals
		
		integrated_field = np.trapz(field_maps_array,x=self.bp_wavelengths,axis=0)
		integrated_bp = np.trapz(self.bp_transmission,x=self.bp_wavelengths,axis=0)

		far_field_map = integrated_field/integrated_bp


		PSF,PSF_dB = Convert_field_to_PSF(far_field_map)
		tmppsf = enmap.enmap(PSF,wcs=self.newmapwcs['map0'])

		farfield_im_size_mono, U_mono = Fraunhofer(A_complex,self.wavelength,self.delta_x)
		PSF_mono,PSF_dB_mono = Convert_field_to_PSF(U_mono)

		self.normalizing_amplitude = np.amax(tmppsf)
		self.normalizing_amplitude_monochromatic = np.amax(PSF_mono)

	def get_toltec_bandpass(self,bandstr,interpwavelengths,plot_bandpass=False,
							pathtobpfile='/Users/golecjoe/Documents/Projects/beamfit_redux/test_v2/model_passbands.npz'):

		allowed_bands = {'band_150', 'band_220', 'band_280'}
	
		if bandstr not in allowed_bands:
			raise ValueError(f"'{bandstr}' is not a valid band. Choose from {allowed_bands}.")
		path = Path(pathtobpfile)
	
		if not path.exists():
			raise FileNotFoundError(f"File '{pathtobpfile}' does not exist.")

		bpfile = np.load(pathtobpfile)

		wavelength_bp = 1E-3*(300.)/bpfile['f_GHz']

		idx2 = np.argsort(interpwavelengths)
		interpwavelengths = interpwavelengths[idx2]

		idx = np.argsort(wavelength_bp)
		wavelength_bp = wavelength_bp[idx]
		bandpass = bpfile[bandstr][idx]
		bptransinterp = np.interp(interpwavelengths, wavelength_bp, bandpass)
		if plot_bandpass:
			plt.figure()
			plt.plot(interpwavelengths*1E3,bptransinterp,'.',c='r',label='Interpolated Bandpass')
			plt.plot(wavelength_bp*1E3,bandpass,c='k',label='Bandpass')
			plt.xlim(np.amin(interpwavelengths*1E3),np.amax(interpwavelengths*1E3))
			plt.xlabel('Wavelengths (mm)')
			plt.ylabel('Transmission')
			plt.show()


		self.bp_wavelengths = interpwavelengths
		self.bp_transmission = bptransinterp








	def initialize_model(self,
						aperture_plane_resolution = 1.,center_on_brightest_pix=False,
						include_legs=False,plot_aperture=False,
						aperture_fwhm = 48.,edge_taper_diameter=48.,plot_illumination=False,
						n=4,m=4,
						bandstr='band_150', interpwavelengths=300E-3/np.linspace(110,170,20),plot_bandpass=False,
						pathtobpfile='/Users/golecjoe/Documents/Projects/beamfit_redux/test_v2/model_passbands.npz'):
		self.truncate_maps(aperture_plane_resolution,center_on_brightest_pix=center_on_brightest_pix)
		self.set_LMT_aperture(include_legs=include_legs,plot_aperture=plot_aperture)
		self.set_illumination(aperture_fwhm = aperture_fwhm,edge_taper_diameter=edge_taper_diameter,plot_illumination=plot_illumination)
		self.get_zernike_polynomials(n,m)
		self.get_toltec_bandpass(bandstr,interpwavelengths,plot_bandpass=plot_bandpass,pathtobpfile=pathtobpfile)
		self.make_normalizing_amplitude()


	
	def function2minimize(self,x):

		"""
		Objective function for beam model fitting.

		Parameters
		----------
		x : array-like
			Fit parameter vector including source amplitude, tilts, defocus,
			and Zernike coefficients.

		Returns
		-------
		float
			RMS residuals across maps 0, 1, and 2 combined.
		"""

		#list of fit params
		# x[0] = source amplitude
		# x[1] = Primary Focal Length
		# x[2] = M2.X offset
		# x[3] = M2.Y offset
		# x[4] = M2.Z offset

		# x[5] = TILT_Y
		# x[6] = TILT_X

		# x[7] = AST_O
		# x[8] = AST_V
		# x[9] = TRE_V
		# ...

		
		# make map 0 zern params


		c_map0 = np.zeros(self.zernike_polynomials.shape[0])
		c_map0[0] = 0. # piston is always 0
		c_map0[1] = x[5] # TILT_Y is always 0
		c_map0[2] = x[6] # TILT_X is always 0
		c_map0[3] = x[7]
		c_map0[4] = 0 # FOCUS is always 0
		c_map0[5:] = x[8:]

		# make map 1 zern params

		c_map1 = np.zeros(self.zernike_polynomials.shape[0])
		c_map1[0] = 0. # piston is always 0
		c_map1[1] = x[5] # TILT_Y is always 0
		c_map1[2] = x[6] # TILT_X is always 0
		c_map1[3] = x[7]
		c_map1[4] = 0 # FOCUS is always 0
		c_map1[5:] = x[8:]

		# make map 2 zern params

		c_map2 = np.zeros(self.zernike_polynomials.shape[0])
		c_map2[0] = 0. # piston is always 0
		c_map2[1] = x[5] # TILT_Y is always 0
		c_map2[2] = x[6] # TILT_X is always 0
		c_map2[3] = x[7]
		c_map2[4] = 0 # FOCUS is always 0
		c_map2[5:] = x[8:]



		# make map 0 psf
	

		# phase0 = self.make_phase(c=c_map0,secondary_offset=x[4]-1.E-3,del_x=x[2],del_y=x[3],del_alph_x=0.,del_alph_y=0.,
		# 						  f=x[1],F=525.,D=50.)

		phase0 = self.make_phase(c=c_map0,secondary_offset=x[4]-1.E-3,del_x=0.,del_y=0.,del_alph_x=x[2],del_alph_y=x[3],
								  f=x[1],F=525.,D=50.)
		if self.achro_beam_fit:
			model_map0 = self.make_psf(phase=phase0,return_psf=True)*x[0]
		else:
			model_map0 = self.make_psf_monochromatic(phase=phase0,return_psf=True)*x[0]
		resids_map0 = self.trunc_maps['map0'] - model_map0
		rms_map0 = np.sqrt(np.mean(resids_map0**2))

		# make map 1 psf
	
		# phase1 = self.make_phase(c=c_map1,secondary_offset=x[4],del_x=x[3],del_y=x[4],del_alph_x=0.,del_alph_y=0.,
		# 						  f=x[1],F=525.,D=50.)
		phase1 = self.make_phase(c=c_map1,secondary_offset=x[4],del_x=0.,del_y=0.,del_alph_x=x[2],del_alph_y=x[3],
								  f=x[1],F=525.,D=50.)
		if self.achro_beam_fit:
			model_map1 = self.make_psf(phase=phase1,return_psf=True)*x[0]
		else:
			model_map1 = self.make_psf_monochromatic(phase=phase1,return_psf=True)*x[0]
		
		resids_map1 = self.trunc_maps['map1'] - model_map1
		rms_map1 = np.sqrt(np.mean(resids_map1**2))

		# make map 2 psf
	
		# phase2 = self.make_phase(c=c_map2,secondary_offset=x[4]+1E-3,del_x=x[3],del_y=x[4],del_alph_x=0.,del_alph_y=0.,
		# 						  f=x[1],F=525.,D=50.)
		phase2 = self.make_phase(c=c_map2,secondary_offset=x[4]+1E-3,del_x=0.,del_y=0.,del_alph_x=x[2],del_alph_y=x[3],
								  f=x[1],F=525.,D=50.)
		if self.achro_beam_fit:
			model_map2 = self.make_psf(phase=phase2,return_psf=True)*x[0]
		else:
			model_map2 = self.make_psf_monochromatic(phase=phase2,return_psf=True)*x[0]
		
		resids_map2 = self.trunc_maps['map2'] - model_map2
		rms_map2 = np.sqrt(np.mean(resids_map2**2))
	
		
		chisquare = np.sqrt(rms_map0**2+rms_map1**2+rms_map2**2)
		self.cost = np.sum(chisquare)
		self.fit_step_counter+=1
		if self.fit_step_counter%500==0:
			print('On fit step ',self.fit_step_counter,' with cost ',np.sum(chisquare))
			#self.quick_plot_results(model_map0,model_map1,model_map2,plot_vmin=-500,plot_vmax=500.,resids_vmin=None,resids_vmax=None,save_fig_name=None,noshow=False,plot_title=None,lowerleft=[-0.75/60,-0.75/60.],upperright=[0.75/60,0.75/60.])
		return chisquare
		
	def fit_beam(self,c_guess=None,boundvals = None,fit_achromatic_beam=False):

		"""
		Fit the beam model to the input maps by optimizing Zernike phase terms
		and geometric defocus.

		Parameters
		----------
		c_guess : array-like, optional
			Initial guess for Zernike coefficients.
		boundvals : list of tuples
			Bounds for optimization variables, passed to scipy.optimize.minimize.
		"""

		#list of fit params
		# x[0] = source amplitude
		# x[1] = Primary Focal Length
		# x[2] = M2.X offset
		# x[3] = M2.Y offset
		# x[4] = M2.Z offset

		# x[5] = TILT_Y
		# x[6] = TILT_X

		# x[7] = AST_O
		# x[8] = AST_V
		# x[9] = TRE_V
		# ...

		self.achro_beam_fit = fit_achromatic_beam


		x0 = np.zeros(self.zernike_polynomials.shape[0]+3) # need to make an array that has the zernike coeffs plus 1 for theta0
		x0[0] = np.amax(self.trunc_maps['map1'])
		x0[1] = 17.5 # guess for primary focal length
		
		if c_guess is not None:
			x0[8:] = c_guess
		
		self.fit_step_counter = 0
		results = minimize(self.function2minimize,x0,bounds=boundvals)
		self.results = results
		if results.success:
			print('The fit was successful. Cost =', self.cost)
		else:
			print('The fit was not successful, you may need to run again with different guess or bounds. Cost =', self.cost)
		#self.make_zernike_results_dict()

		c_bestfit_map0 = np.zeros(self.zernike_polynomials.shape[0])
		c_bestfit_map0[1] = results.x[5]
		c_bestfit_map0[2] = results.x[6]
		c_bestfit_map0[3] = results.x[7]
		c_bestfit_map0[5:] = results.x[8:]

		c_bestfit_map1 = np.zeros(self.zernike_polynomials.shape[0])
		c_bestfit_map1[1] = results.x[5]
		c_bestfit_map1[2] = results.x[6]
		c_bestfit_map1[3] = results.x[7]
		c_bestfit_map1[5:] = results.x[8:]

		c_bestfit_map2 = np.zeros(self.zernike_polynomials.shape[0])
		c_bestfit_map2[1] = results.x[5]
		c_bestfit_map2[2] = results.x[6]
		c_bestfit_map2[3] = results.x[7]
		c_bestfit_map2[5:] = results.x[8:]

		c_bestfit_ideal = np.zeros(self.zernike_polynomials.shape[0])
		c_bestfit_ideal[3] = results.x[7]
		c_bestfit_ideal[5:] = results.x[8:]

		# make bestfitmaps for resids
		self.bestfit_maps = {}

		# phase0_fit = self.make_phase(c=c_bestfit_map0,secondary_offset=results.x[4]-1.E-3,del_x=results.x[3],del_y=results.x[4],del_alph_x=0.,del_alph_y=0.,
		# 						  f=results.x[1],F=525.,D=50.)
		phase0_fit = self.make_phase(c=c_bestfit_map0,secondary_offset=results.x[4]-1.E-3,del_x=0.,del_y=0.,del_alph_x=results.x[2],del_alph_y=results.x[3],
								  f=results.x[1],F=525.,D=50.)
		if self.achro_beam_fit:
			tmppsf_raw = self.make_psf(phase=phase0_fit,return_psf=True)*results.x[0]

		else:
			tmppsf_raw = self.make_psf_monochromatic(phase=phase0_fit,return_psf=True)*results.x[0]
		self.bestfit_maps['map0'] = tmppsf_raw

		# phase1_fit = self.make_phase(c=c_bestfit_map1,secondary_offset=results.x[4],del_x=results.x[3],del_y=results.x[4],del_alph_x=0.,del_alph_y=0.,
		# 						  f=results.x[1],F=525.,D=50.)
		phase1_fit = self.make_phase(c=c_bestfit_map1,secondary_offset=results.x[4],del_x=0.,del_y=0.,del_alph_x=results.x[2],del_alph_y=results.x[3],
								  f=results.x[1],F=525.,D=50.)
		if self.achro_beam_fit:
			tmppsf_raw = self.make_psf(phase=phase1_fit,return_psf=True)*results.x[0]
		else:
			tmppsf_raw = self.make_psf_monochromatic(phase=phase1_fit,return_psf=True)*results.x[0]
		self.bestfit_maps['map1'] = tmppsf_raw

		# phase2_fit = self.make_phase(c=c_bestfit_map2,secondary_offset=results.x[4]+1.E-3,del_x=results.x[3],del_y=results.x[4],del_alph_x=0.,del_alph_y=0.,
		# 						  f=results.x[1],F=525.,D=50.)
		phase2_fit = self.make_phase(c=c_bestfit_map2,secondary_offset=results.x[4]+1.E-3,del_x=0.,del_y=0.,del_alph_x=results.x[2],del_alph_y=results.x[3],
								  f=results.x[1],F=525.,D=50.)
		if self.achro_beam_fit:
			tmppsf_raw = self.make_psf(phase=phase2_fit,return_psf=True)*results.x[0]
		else:
			tmppsf_raw = self.make_psf_monochromatic(phase=phase2_fit,return_psf=True)*results.x[0]
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
		tmppsf_raw = enmap.enmap(results.x[0]*(PSF/self.normalizing_amplitude_monochromatic),wcs=self.newmapwcs['map0'])
		gain_loss = np.amax(tmppsf_raw/results.x[0])
		self.gain_loss = gain_loss

		self.bestfitbeam = tmppsf_raw

	def make_zernike_results_dict(self):
		zernike_labels = np.array(['AST_V', 'AST_O', 'COMA_H', 'COMA_V', 'TRE_O', 'TRE_V', 'SPH', 'QUAD_V', 'QUAD_O', 'AST2_O', 'AST2_V'])
		zernike_values = np.array([
			self.results.x[8], self.results.x[7], self.results.x[11], self.results.x[10],
			self.results.x[12], self.results.x[9], self.results.x[15],
			self.results.x[17], self.results.x[13], self.results.x[14], self.results.x[16]
		])

		zernike_dict = {
			'labels': zernike_labels.tolist(),  # Convert NumPy array to list
			'values': zernike_values.tolist()   # Convert NumPy array to list
		}

		self.zernike_dict = zernike_dict
		print('Made Zernike dict')



	def plot_bandpass(self):
		plt.figure()
		plt.plot(1E3*self.bp_wavelengths,self.bp_transmission)
		plt.xlabel('wavelengths (mm)')
		plt.ylabel('Transmission')
		plt.show()

	def plot_phase(self,plot_vmin=None,plot_vmax=None,save_fig_name=None,noshow=False):
		"""
		Plot the current phase screen.

		Parameters
		----------
		plot_vmin, plot_vmax : float, optional
			Color limits.
		save_fig_name : str, optional
			Path to save the figure.
		noshow : bool
			If True, suppress display (for script automation).
		"""
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
		"""
		Plot the inferred surface error (in microns) from the fitted phase screen.

		Parameters
		----------
		plot_vmin, plot_vmax : float, optional
			Color limits.
		save_fig_name : str, optional
			Path to save the figure.
		noshow : bool
			If True, suppress display.
		"""
		tmpwavelength = self.wavelength

		phase_primary_only = self.make_phase(c=self.zern_coefficients,secondary_offset=0.,del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
								  f=17.5,F=525.,D=50.)
		
		self.surface_error = 1E6*tmpwavelength*phase_primary_only/(2*np.pi)
		surfrms = np.sqrt(np.mean(self.surface_error[np.nonzero(self.surface_error)]**2))
		print('RMS = ',surfrms,'microns')
		plt.figure()
		plt.imshow(self.surface_error,vmin=plot_vmin,vmax=plot_vmax,extent=([-self.L/2, self.L/2, -self.L/2, self.L/2]))
		cbar = plt.colorbar()
		cbar.set_label(label='microns',rotation=270,labelpad=15,fontsize=14)
		
		plt.title("Surface Error")
		plt.xlabel("x [m]")
		plt.ylabel("y [m]")
		
		plt.xlim(-30,30)
		plt.ylim(-30,30)
		# Add text box with RMS value
		rms_text = f"RMS = {surfrms:.1f} microns"
		plt.text(0.98, 0.95, rms_text,
				 transform=plt.gca().transAxes,
				 fontsize=12,
				 verticalalignment='top',
				 horizontalalignment='right',
				 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

		if save_fig_name is not None:
			plt.savefig(save_fig_name,facecolor='white',transparent=False,bbox_inches='tight')
		
		if noshow:
			plt.close()
		else:
			plt.show()

	def plot_results(self,plot_vmin=None,plot_vmax=None,resids_vmin=None,resids_vmax=None,save_fig_name=None,noshow=False,plot_title=None,lowerleft=[-0.75/60,-0.75/60.],upperright=[0.75/60,0.75/60.]):
		"""
		Show a 3×3 panel plot of input maps, best-fit models, and residuals.

		Parameters
		----------
		plot_vmin, plot_vmax : float, optional
			Color scale limits for data/model maps.
		resids_vmin, resids_vmax : float, optional
			Color scale limits for residual maps.
		save_fig_name : str, optional
			File name to save the figure.
		noshow : bool
			If True, don't display figure.
		plot_title : str, optional
			Title to place above all subplots.
		lowerleft, upperright : list of float
			Sky coordinate box (in degrees) to show in each subplot.
		"""
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
		"""
		Display the final modeled PSF.

		Parameters
		----------
		plot_vmin, plot_vmax : float, optional
			Color limits.
		save_fig_name : str, optional
			File name to save the figure.
		xlims, ylims : tuple, optional
			Axes limits in degrees.
		"""
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

	def plot_inputmaps(self,plot_vmin=None,plot_vmax=None,save_fig_name=None,noshow=False,lowerleft=[-0.75/60,-0.75/60.],upperright=[0.75/60,0.75/60.]):
		"""
		Plot the truncated input maps used in the beam fit.

		Parameters
		----------
		plot_vmin, plot_vmax : float, optional
			Color scale limits.
		save_fig_name : str, optional
			File name to save the figure.
		noshow : bool
			If True, suppress figure display.
		lowerleft, upperright : list of float
			Coordinate box in degrees to crop view.
		"""
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

	def function2minimize_with_pointing_offsets(self,x):

		"""
		Objective function for beam model fitting.

		Parameters
		----------
		x : array-like
			Fit parameter vector including source amplitude, tilts, defocus,
			and Zernike coefficients.

		Returns
		-------
		float
			RMS residuals across maps 0, 1, and 2 combined.
		"""

		#list of fit params
		# x[0] = source amplitude
		# x[1] = M2.Z offset

		# x[2] = TILT_Y_map0
		# x[3] = TILT_X_map0

		# x[4] = TILT_Y_map1
		# x[5] = TILT_X_map1

		# x[6] = TILT_Y_map2
		# x[7] = TILT_X_map2

		# x[8] = AST_O
		# x[9] = AST_V
		# x[10] = TRE_V
		# ...
		
		# make map 0 zern params


		c_map0 = np.zeros(self.zernike_polynomials.shape[0])
		c_map0[0] = 0. # piston is always 0
		c_map0[1] = x[2] # TILT_Y is always 0
		c_map0[2] = x[3] # TILT_X is always 0
		c_map0[3] = x[8]
		c_map0[4] = 0 # FOCUS is always 0
		c_map0[5:] = x[9:]

		# make map 1 zern params

		c_map1 = np.zeros(self.zernike_polynomials.shape[0])
		c_map1[0] = 0. # piston is always 0
		c_map1[1] = x[4] # TILT_Y is always 0
		c_map1[2] = x[5] # TILT_X is always 0
		c_map1[3] = x[8]
		c_map1[4] = 0 # FOCUS is always 0
		c_map1[5:] = x[9:]

		# make map 2 zern params

		c_map2 = np.zeros(self.zernike_polynomials.shape[0])
		c_map2[0] = 0. # piston is always 0
		c_map2[1] = x[6] # TILT_Y is always 0
		c_map2[2] = x[7] # TILT_X is always 0
		c_map2[3] = x[8]
		c_map2[4] = 0 # FOCUS is always 0
		c_map2[5:] = x[9:]



		# make map 0 psf
	

		# phase0 = self.make_phase(c=c_map0,secondary_offset=x[4]-1.E-3,del_x=x[2],del_y=x[3],del_alph_x=0.,del_alph_y=0.,
		# 						  f=x[1],F=525.,D=50.)

		phase0 = self.make_phase(c=c_map0,secondary_offset=x[1]-1.E-3,del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
								  f=17.5,F=525.,D=50.)
		if self.achro_beam_fit:
			model_map0 = self.make_psf(phase=phase0,return_psf=True)*x[0]
		else:
			model_map0 = self.make_psf_monochromatic(phase=phase0,return_psf=True)*x[0]
		resids_map0 = self.trunc_maps['map0'] - model_map0
		rms_map0 = np.sqrt(np.mean(resids_map0**2))

		# make map 1 psf
	
		# phase1 = self.make_phase(c=c_map1,secondary_offset=x[4],del_x=x[3],del_y=x[4],del_alph_x=0.,del_alph_y=0.,
		# 						  f=x[1],F=525.,D=50.)
		phase1 = self.make_phase(c=c_map1,secondary_offset=x[1],del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
								  f=17.5,F=525.,D=50.)
		if self.achro_beam_fit:
			model_map1 = self.make_psf(phase=phase1,return_psf=True)*x[0]
		else:
			model_map1 = self.make_psf_monochromatic(phase=phase1,return_psf=True)*x[0]
		
		resids_map1 = self.trunc_maps['map1'] - model_map1
		rms_map1 = np.sqrt(np.mean(resids_map1**2))

		# make map 2 psf
	
		# phase2 = self.make_phase(c=c_map2,secondary_offset=x[4]+1E-3,del_x=x[3],del_y=x[4],del_alph_x=0.,del_alph_y=0.,
		# 						  f=x[1],F=525.,D=50.)
		phase2 = self.make_phase(c=c_map2,secondary_offset=x[1]+1E-3,del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
								  f=17.5,F=525.,D=50.)
		if self.achro_beam_fit:
			model_map2 = self.make_psf(phase=phase2,return_psf=True)*x[0]
		else:
			model_map2 = self.make_psf_monochromatic(phase=phase2,return_psf=True)*x[0]
		
		resids_map2 = self.trunc_maps['map2'] - model_map2
		rms_map2 = np.sqrt(np.mean(resids_map2**2))
	
		
		chisquare = np.sqrt(rms_map0**2+rms_map1**2+rms_map2**2)
		self.cost = np.sum(chisquare)
		self.fit_step_counter+=1
		if self.fit_step_counter%500==0:
			print('On fit step ',self.fit_step_counter,' with cost ',np.sum(chisquare))
			#self.quick_plot_results(model_map0,model_map1,model_map2,plot_vmin=-500,plot_vmax=500.,resids_vmin=None,resids_vmax=None,save_fig_name=None,noshow=False,plot_title=None,lowerleft=[-0.75/60,-0.75/60.],upperright=[0.75/60,0.75/60.])
		return chisquare
		
	def fit_beam_with_pointing_offsets(self,c_guess=None,boundvals = None,fit_achromatic_beam=False):

		"""
		Fit the beam model to the input maps by optimizing Zernike phase terms
		and geometric defocus.

		Parameters
		----------
		c_guess : array-like, optional
			Initial guess for Zernike coefficients.
		boundvals : list of tuples
			Bounds for optimization variables, passed to scipy.optimize.minimize.
		"""

		#list of fit params
		# x[0] = source amplitude
		# x[1] = M2.Z offset

		# x[2] = TILT_Y_map0
		# x[3] = TILT_X_map0

		# x[4] = TILT_Y_map1
		# x[5] = TILT_X_map1

		# x[6] = TILT_Y_map2
		# x[7] = TILT_X_map2

		# x[8] = AST_O
		# x[9] = AST_V
		# x[10] = TRE_V
		# ...

		self.achro_beam_fit = fit_achromatic_beam


		x0 = np.zeros(self.zernike_polynomials.shape[0]+4) # need to make an array that has the zernike coeffs plus 1 for theta0
		x0[0] = np.amax(self.trunc_maps['map1'])

		
		if c_guess is not None:
			x0[11:] = c_guess
		
		self.fit_step_counter = 0
		results = minimize(self.function2minimize_with_pointing_offsets,x0,bounds=boundvals)
		self.results = results
		if results.success:
			print('The fit was successful. Cost =', self.cost)
		else:
			print('The fit was not successful, you may need to run again with different guess or bounds. Cost =', self.cost)

		self.make_zernike_results_dict()
		c_bestfit_map0 = np.zeros(self.zernike_polynomials.shape[0])
		c_bestfit_map0[1] = results.x[2]
		c_bestfit_map0[2] = results.x[3]
		c_bestfit_map0[3] = results.x[8]
		c_bestfit_map0[5:] = results.x[9:]

		c_bestfit_map1 = np.zeros(self.zernike_polynomials.shape[0])
		c_bestfit_map1[1] = results.x[4]
		c_bestfit_map1[2] = results.x[5]
		c_bestfit_map1[3] = results.x[8]
		c_bestfit_map1[5:] = results.x[9:]

		c_bestfit_map2 = np.zeros(self.zernike_polynomials.shape[0])
		c_bestfit_map2[1] = results.x[6]
		c_bestfit_map2[2] = results.x[7]
		c_bestfit_map2[3] = results.x[8]
		c_bestfit_map2[5:] = results.x[9:]

		c_bestfit_ideal = np.zeros(self.zernike_polynomials.shape[0])
		c_bestfit_ideal[3] = results.x[8]
		c_bestfit_ideal[5:] = results.x[9:]

		# make bestfitmaps for resids
		self.bestfit_maps = {}

		# phase0_fit = self.make_phase(c=c_bestfit_map0,secondary_offset=results.x[4]-1.E-3,del_x=results.x[3],del_y=results.x[4],del_alph_x=0.,del_alph_y=0.,
		# 						  f=results.x[1],F=525.,D=50.)
		phase0_fit = self.make_phase(c=c_bestfit_map0,secondary_offset=results.x[1]-1.E-3,del_x=0.,del_y=0.,del_alph_x=0,del_alph_y=0,
								  f=17.5,F=525.,D=50.)
		if self.achro_beam_fit:
			tmppsf_raw = self.make_psf(phase=phase0_fit,return_psf=True)*results.x[0]

		else:
			tmppsf_raw = self.make_psf_monochromatic(phase=phase0_fit,return_psf=True)*results.x[0]
		self.bestfit_maps['map0'] = tmppsf_raw

		# phase1_fit = self.make_phase(c=c_bestfit_map1,secondary_offset=results.x[4],del_x=results.x[3],del_y=results.x[4],del_alph_x=0.,del_alph_y=0.,
		# 						  f=results.x[1],F=525.,D=50.)
		phase1_fit = self.make_phase(c=c_bestfit_map1,secondary_offset=results.x[1],del_x=0.,del_y=0.,del_alph_x=0,del_alph_y=0,
								  f=17.5,F=525.,D=50.)
		if self.achro_beam_fit:
			tmppsf_raw = self.make_psf(phase=phase1_fit,return_psf=True)*results.x[0]
		else:
			tmppsf_raw = self.make_psf_monochromatic(phase=phase1_fit,return_psf=True)*results.x[0]
		self.bestfit_maps['map1'] = tmppsf_raw

		# phase2_fit = self.make_phase(c=c_bestfit_map2,secondary_offset=results.x[4]+1.E-3,del_x=results.x[3],del_y=results.x[4],del_alph_x=0.,del_alph_y=0.,
		# 						  f=results.x[1],F=525.,D=50.)
		phase2_fit = self.make_phase(c=c_bestfit_map2,secondary_offset=results.x[1]+1.E-3,del_x=0.,del_y=0.,del_alph_x=0,del_alph_y=0,
								  f=17.5,F=525.,D=50.)
		if self.achro_beam_fit:
			tmppsf_raw = self.make_psf(phase=phase2_fit,return_psf=True)*results.x[0]
		else:
			tmppsf_raw = self.make_psf_monochromatic(phase=phase2_fit,return_psf=True)*results.x[0]
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
		tmppsf_raw = enmap.enmap(results.x[0]*(PSF/self.normalizing_amplitude_monochromatic),wcs=self.newmapwcs['map0'])
		gain_loss = np.amax(tmppsf_raw/results.x[0])
		self.gain_loss = gain_loss

		self.bestfitbeam = tmppsf_raw
