import os
import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.wcs import WCS
from pixell import enmap, enplot, reproject, utils, curvedsky,wcsutils 


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

				if len(np.shape(hdu.data))==4:
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
				else:
					tmpenmap = enmap.enmap(hdu.data[:,:],wcs=WCS(header=hdu.header))
				
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
	phi = np.arctan2(-y,x)
	return(x,y,r,phi)

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

