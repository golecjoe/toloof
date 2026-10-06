import numpy as np
from math import factorial

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
		angular_part = np.sin(m*phi)
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
	tmpdefocus = -dz*(((1-a**2)/(1+a**2))+((1-b**2)/(1+b**2)))
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
