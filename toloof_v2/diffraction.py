import numpy as np

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
	return PSF