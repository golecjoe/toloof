import sys

sys.path.append('../')

from toloof import *

fit_a1100 = True
fit_a1400 = True
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
paths_a1400 = []
paths_a1100 = []

for i in obsnums:

	paths_a2000.append('test_data/'+i+'/raw/toltec_commissioning_a2000_pointing_'+i+'_citlali.fits')
	paths_a1400.append('test_data/'+i+'/raw/toltec_commissioning_a1400_pointing_'+i+'_citlali.fits')
	paths_a1100.append('test_data/'+i+'/raw/toltec_commissioning_a1100_pointing_'+i+'_citlali.fits')

pathlist_a1100 = np.array(paths_a1100)
pathlist_a2000 = np.array(paths_a2000)

if fit_a1100:
	print('FITTING a1100')
	beamfit_a1100 = Fraunhofer_Beamfit(paths_a1100,1.1E-3,mask_radius=2./60.,padpixels = 60)
	beamfit_a1100.initialize_model(aperture_plane_resolution = 1.0,center_on_brightest_pix=True,
		            bandstr='band_280',interpwavelengths=300E-3/np.linspace(250,310,5),
					aperture_fwhm = 43.,edge_taper_diameter=43.,pathtobpfile='model_passbands.npz')
	beamfit_a1100.fit_beam_with_pointing_offsets(c_guess=None,boundvals = None,fit_achromatic_beam=False)
	print('---------------------------------------------------','\n')

	print('FIT RESULTS a1100:')
	print('SOURCE AMPLITUDE = ',beamfit_a1100.results.x[0])
	print('M2.Z Offset = ',beamfit_a1100.results.x[1])

	print('map 0 TILT Y = ',beamfit_a1100.results.x[2]*-5, ' arcsec')
	print('map 0 TILT X = ',beamfit_a1100.results.x[3]*5, ' arcsec')

	print('map 1 TILT Y = ',beamfit_a1100.results.x[4]*-5, ' arcsec')
	print('map 1 TILT X = ',beamfit_a1100.results.x[5]*5, ' arcsec')

	print('map 2 TILT Y = ',beamfit_a1100.results.x[6]*-5, ' arcsec')
	print('map 2 TILT X = ',beamfit_a1100.results.x[7]*5, ' arcsec')

	print('AST_O = ',beamfit_a1100.results.x[8]*(beamfit_a1100.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('AST_V = ',beamfit_a1100.results.x[9]*(beamfit_a1100.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('TRE_V = ',beamfit_a1100.results.x[10]*(beamfit_a1100.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('COMA_V = ',beamfit_a1100.results.x[11]*(beamfit_a1100.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('COMA_H = ',beamfit_a1100.results.x[12]*(beamfit_a1100.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('TRE_O = ',beamfit_a1100.results.x[13]*(beamfit_a1100.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('QUAD_O = ',beamfit_a1100.results.x[14]*(beamfit_a1100.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('AST2_O = ',beamfit_a1100.results.x[15]*(beamfit_a1100.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('SPH = ',beamfit_a1100.results.x[16]*(beamfit_a1100.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('AST2_V = ',beamfit_a1100.results.x[17]*(beamfit_a1100.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('QUAD_V = ',beamfit_a1100.results.x[18]*(beamfit_a1100.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
		
	print('DERIVED:')
	print('Gain Percent: ',beamfit_a1100.gain_loss,'\n')
	print('---------------------------------------------------')

	# beamfit_a1100.plot_results(plot_vmin=-500,plot_vmax=500,save_fig_name=plot_results_name_a1100
	# 							   ,noshow=True,plot_title=plot_results_title_a1100)
	# beamfit_a1100.plot_surface_error(plot_vmin=surface_error_vmin,plot_vmax=surface_error_vmax,save_fig_name=plot_surferror_name_a1100,noshow=True)
	# np.savetxt(save_results_name_a1100,beamfit_a1100.results.x[:],delimiter=',')

if fit_a1400:
	print('FITTING a1400')
	beamfit_a1400 = Fraunhofer_Beamfit(paths_a1400,1.4E-3,mask_radius=1./60.,padpixels = 60)
	beamfit_a1400.initialize_model(aperture_plane_resolution = 1.0,center_on_brightest_pix=True,
		            bandstr='band_220',interpwavelengths=300E-3/np.linspace(195,242,5),
					aperture_fwhm = 43.,edge_taper_diameter=43.,pathtobpfile='model_passbands.npz')
	beamfit_a1400.fit_beam_with_pointing_offsets(c_guess=None,boundvals = None,fit_achromatic_beam=False)
	print('---------------------------------------------------','\n')

	print('FIT RESULTS a1400:')
	print('SOURCE AMPLITUDE = ',beamfit_a1400.results.x[0])
	print('M2.Z Offset = ',beamfit_a1400.results.x[1])

	print('map 0 TILT Y = ',beamfit_a1400.results.x[2]*-5, ' arcsec')
	print('map 0 TILT X = ',beamfit_a1400.results.x[3]*5, ' arcsec')

	print('map 1 TILT Y = ',beamfit_a1400.results.x[4]*-5, ' arcsec')
	print('map 1 TILT X = ',beamfit_a1400.results.x[5]*5, ' arcsec')

	print('map 2 TILT Y = ',beamfit_a1400.results.x[6]*-5, ' arcsec')
	print('map 2 TILT X = ',beamfit_a1400.results.x[7]*5, ' arcsec')

	print('AST_O = ',beamfit_a1400.results.x[8]*(beamfit_a1400.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('AST_V = ',beamfit_a1400.results.x[9]*(beamfit_a1400.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('TRE_V = ',beamfit_a1400.results.x[10]*(beamfit_a1400.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('COMA_V = ',beamfit_a1400.results.x[11]*(beamfit_a1400.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('COMA_H = ',beamfit_a1400.results.x[12]*(beamfit_a1400.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('TRE_O = ',beamfit_a1400.results.x[13]*(beamfit_a1400.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('QUAD_O = ',beamfit_a1400.results.x[14]*(beamfit_a1400.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('AST2_O = ',beamfit_a1400.results.x[15]*(beamfit_a1400.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('SPH = ',beamfit_a1400.results.x[16]*(beamfit_a1400.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('AST2_V = ',beamfit_a1400.results.x[17]*(beamfit_a1400.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('QUAD_V = ',beamfit_a1400.results.x[18]*(beamfit_a1400.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
		
	print('DERIVED:')
	print('Gain Percent: ',beamfit_a1400.gain_loss,'\n')
	print('---------------------------------------------------')

	# beamfit_a1400.plot_results(plot_vmin=-500,plot_vmax=500,save_fig_name=plot_results_name_a1400
	# 							   ,noshow=True,plot_title=plot_results_title_a1400)
	# beamfit_a1400.plot_surface_error(plot_vmin=surface_error_vmin,plot_vmax=surface_error_vmax,save_fig_name=plot_surferror_name_a1400,noshow=True)

	# np.savetxt(save_results_name_a1400,beamfit_a1400.results.x[:],delimiter=',')



if fit_a2000:
	print('FITTING a2000')
	beamfit_a2000 = Fraunhofer_Beamfit(paths_a2000,2.0E-3,mask_radius=2./60.,padpixels = 60)
	beamfit_a2000.initialize_model(aperture_plane_resolution = 1.0,center_on_brightest_pix=True,
		            bandstr='band_150',interpwavelengths=300E-3/np.linspace(128,170,5),
					aperture_fwhm = 43.,edge_taper_diameter=43.,pathtobpfile='model_passbands.npz')


	beamfit_a2000.fit_beam_with_pointing_offsets(c_guess=None,boundvals = None,fit_achromatic_beam=False)
	print('---------------------------------------------------','\n')

	print('FIT RESULTS a2000:')
	print('SOURCE AMPLITUDE = ',beamfit_a2000.results.x[0])
	print('M2.Z Offset = ',beamfit_a2000.results.x[1])

	print('map 0 TILT Y = ',beamfit_a2000.results.x[2]*-5, ' arcsec')
	print('map 0 TILT X = ',beamfit_a2000.results.x[3]*5, ' arcsec')

	print('map 1 TILT Y = ',beamfit_a2000.results.x[4]*-5, ' arcsec')
	print('map 1 TILT X = ',beamfit_a2000.results.x[5]*5, ' arcsec')

	print('map 2 TILT Y = ',beamfit_a2000.results.x[6]*-5, ' arcsec')
	print('map 2 TILT X = ',beamfit_a2000.results.x[7]*5, ' arcsec')

	print('AST_O = ',beamfit_a2000.results.x[8]*(beamfit_a2000.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('AST_V = ',beamfit_a2000.results.x[9]*(beamfit_a2000.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('TRE_V = ',beamfit_a2000.results.x[10]*(beamfit_a2000.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('COMA_V = ',beamfit_a2000.results.x[11]*(beamfit_a2000.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('COMA_H = ',beamfit_a2000.results.x[12]*(beamfit_a2000.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('TRE_O = ',beamfit_a2000.results.x[13]*(beamfit_a2000.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('QUAD_O = ',beamfit_a2000.results.x[14]*(beamfit_a2000.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('AST2_O = ',beamfit_a2000.results.x[15]*(beamfit_a2000.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('SPH = ',beamfit_a2000.results.x[16]*(beamfit_a2000.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('AST2_V = ',beamfit_a2000.results.x[17]*(beamfit_a2000.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
	print('QUAD_V = ',beamfit_a2000.results.x[18]*(beamfit_a2000.wavelength*1E6)/(2.*np.pi*np.sqrt(2)))
		
	print('DERIVED:')
	print('Gain Percent: ',beamfit_a2000.gain_loss,'\n')
	print('---------------------------------------------------')

	# beamfit_a2000.plot_inputmaps(save_fig_name=plot_initmaps_name_a2000,noshow=True,lowerleft=[-0.75/60,-0.75/60.],upperright=[0.75/60,0.75/60.])

	# beamfit_a2000.plot_results(plot_vmin=-500,plot_vmax=500,save_fig_name=plot_results_name_a2000
	# 							   ,noshow=True,plot_title=plot_results_title_a2000)
	# beamfit_a2000.plot_surface_error(plot_vmin=surface_error_vmin,plot_vmax=surface_error_vmax,save_fig_name=plot_surferror_name_a2000,noshow=True)

	# np.savetxt(save_results_name_a2000,beamfit_a2000.results.x[:],delimiter=',')

