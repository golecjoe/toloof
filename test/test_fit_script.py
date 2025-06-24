import sys

sys.path.append('../')

from toloof import *

fit_a1100 = True
fit_a2000 = True 

fit_n_order = 4
fit_m_order = 4


first_obsnum = 134272
obsnums = []
obsnums.append(str(first_obsnum))
obsnums.append(str(first_obsnum+1))
obsnums.append(str(first_obsnum+2))

print('Reducting OOF obsnums ',obsnums)

paths_a2000 = []
paths_a1100 = []

for i in obsnums:

	paths_a2000.append('test_data/'+i+'/raw/toltec_commissioning_a2000_pointing_'+i+'_citlali.fits')
	paths_a1100.append('test_data/'+i+'/raw/toltec_commissioning_a1100_pointing_'+i+'_citlali.fits')

pathlist_a1100 = np.array(paths_a1100)
pathlist_a2000 = np.array(paths_a2000)

if fit_a1100:
	print('FITTING a1100')
	beamfit_a1100 = Fraunhofer_Beamfit(pathlist_a1100,1.1E-3,mask_radius=2./60.,padpixels = 60)
	beamfit_a1100.truncate_maps(1.0)
	beamfit_a1100.set_LMT_aperture()
	beamfit_a1100.get_zernike_polynomials(fit_n_order,fit_m_order)
	beamfit_a1100.set_phase()
	beamfit_a1100.set_illumination(aperture_fwhm = 43.,edge_taper_diameter=43.)
	beamfit_a1100.make_normalizing_amplitude()
	beamfit_a1100.make_psf()
	
	beamfit_a1100.fit_beam()
	beamfit_a1100.plot_surface_error(plot_vmin=None,plot_vmax=None,noshow=False)
	print('---------------------------------------------------','\n')


	print('FIT RESULTS a1100:')
	print('Source Amplitude = ',beamfit_a1100.results.x[0])
	print('Primary Focal Length = ',beamfit_a1100.results.x[1])
	print('M2.X Offset = ',beamfit_a1100.results.x[2])
	print('M2.Y Offset = ',beamfit_a1100.results.x[3])
	print('M2.Z Offset = ',beamfit_a1100.results.x[4])
	print('M2.alph_X Offset = ',beamfit_a1100.results.x[5])
	print('M2.alph_Y Offset = ',beamfit_a1100.results.x[6])


	print('ZERNIKES:')
	print('AST_O =',1E6*beamfit_a1100.results.x[7]*(beamfit_a1100.wavelength/(2.*np.pi))/np.sqrt(2))
	print('AST_V =',1E6*beamfit_a1100.results.x[8]*(beamfit_a1100.wavelength/(2.*np.pi))/np.sqrt(2))
	print('COMA_H =',1E6*beamfit_a1100.results.x[11]*(beamfit_a1100.wavelength/(2.*np.pi))/np.sqrt(2))
	print('COMA_V =',1E6*beamfit_a1100.results.x[10]*(beamfit_a1100.wavelength/(2.*np.pi))/np.sqrt(2))
	print('TRE_O =',1E6*beamfit_a1100.results.x[12]*(beamfit_a1100.wavelength/(2.*np.pi))/np.sqrt(2))
	print('TRE_V =',1E6*beamfit_a1100.results.x[9]*(beamfit_a1100.wavelength/(2.*np.pi))/np.sqrt(2))
	print('SPH =',1E6*beamfit_a1100.results.x[15]*(beamfit_a1100.wavelength/(2.*np.pi))/np.sqrt(2))
	print('AST2_V =',1E6*beamfit_a1100.results.x[16]*(beamfit_a1100.wavelength/(2.*np.pi))/np.sqrt(2))
	print('AST2_O =',1E6*beamfit_a1100.results.x[14]*(beamfit_a1100.wavelength/(2.*np.pi))/np.sqrt(2))
	print('QUAD_V=',1E6*beamfit_a1100.results.x[17]*(beamfit_a1100.wavelength/(2.*np.pi))/np.sqrt(2))
	print('QUAD_O =',1E6*beamfit_a1100.results.x[13]*(beamfit_a1100.wavelength/(2.*np.pi))/np.sqrt(2),'\n')

	print('DERIVED:')
	print('Strehl Ratio: ',beamfit_a1100.gain_loss,'\n')
	print('---------------------------------------------------')

	beamfit_a1100.plot_results(plot_vmin=-500,plot_vmax=500
		                               ,noshow=False)

if fit_a2000:
	print('FITTING a2000')
	beamfit_a2000 = Fraunhofer_Beamfit(pathlist_a2000,2.0E-3,mask_radius=2./60.,padpixels = 60)
	beamfit_a2000.truncate_maps(1.0)
	beamfit_a2000.set_LMT_aperture()
	beamfit_a2000.get_zernike_polynomials(fit_n_order,fit_m_order)
	beamfit_a2000.set_phase()
	#beamfit_a2000.set_phase(5)
	beamfit_a2000.set_illumination(aperture_fwhm = 43.,edge_taper_diameter=43.)
	beamfit_a2000.make_normalizing_amplitude()
	beamfit_a2000.make_psf()
	
	beamfit_a2000.fit_beam()
	beamfit_a2000.plot_surface_error(plot_vmin=None,plot_vmax=None,noshow=False)
	
	print('---------------------------------------------------','\n')


	print('FIT RESULTS a2000:')
	print('Source Amplitude = ',beamfit_a2000.results.x[0])
	print('Primary Focal Length = ',beamfit_a2000.results.x[1])
	print('M2.X Offset = ',beamfit_a2000.results.x[2])
	print('M2.Y Offset = ',beamfit_a2000.results.x[3])
	print('M2.Z Offset = ',beamfit_a2000.results.x[4])
	print('M2.alph_X Offset = ',beamfit_a2000.results.x[5])
	print('M2.alph_Y Offset = ',beamfit_a2000.results.x[6])


	print('ZERNIKES:')
	print('AST_O =',1E6*beamfit_a2000.results.x[7]*(beamfit_a2000.wavelength/(2.*np.pi))/np.sqrt(2))
	print('AST_V =',1E6*beamfit_a2000.results.x[8]*(beamfit_a2000.wavelength/(2.*np.pi))/np.sqrt(2))
	print('COMA_H =',1E6*beamfit_a2000.results.x[11]*(beamfit_a2000.wavelength/(2.*np.pi))/np.sqrt(2))
	print('COMA_V =',1E6*beamfit_a2000.results.x[10]*(beamfit_a2000.wavelength/(2.*np.pi))/np.sqrt(2))
	print('TRE_O =',1E6*beamfit_a2000.results.x[12]*(beamfit_a2000.wavelength/(2.*np.pi))/np.sqrt(2))
	print('TRE_V =',1E6*beamfit_a2000.results.x[9]*(beamfit_a2000.wavelength/(2.*np.pi))/np.sqrt(2))
	print('SPH =',1E6*beamfit_a2000.results.x[15]*(beamfit_a2000.wavelength/(2.*np.pi))/np.sqrt(2))
	print('AST2_V =',1E6*beamfit_a2000.results.x[16]*(beamfit_a2000.wavelength/(2.*np.pi))/np.sqrt(2))
	print('AST2_O =',1E6*beamfit_a2000.results.x[14]*(beamfit_a2000.wavelength/(2.*np.pi))/np.sqrt(2))
	print('QUAD_V=',1E6*beamfit_a2000.results.x[17]*(beamfit_a2000.wavelength/(2.*np.pi))/np.sqrt(2))
	print('QUAD_O =',1E6*beamfit_a2000.results.x[13]*(beamfit_a2000.wavelength/(2.*np.pi))/np.sqrt(2),'\n')

	print('DERIVED:')
	print('Strehl Ratio: ',beamfit_a2000.gain_loss,'\n')
	print('---------------------------------------------------')

	beamfit_a2000.plot_results(plot_vmin=-500,plot_vmax=500,noshow=False)


