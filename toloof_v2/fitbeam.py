from scipy.optimize import curve_fit,minimize
import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap, enplot, reproject, utils, curvedsky,wcsutils
import json

from beamclass import Beam


class fit_beam_with_pointing_offsets:

	def __init__(self,beam_class):
		print('Initializing the fit beam class ')

		mapcounter= 0 
		for i in beam_class.trunc_maps:
			mapcounter+=1
		print(f'There are {mapcounter} maps')
		print('The vector of model parameters will have the form:')
		print('x[0] = source_amplitude')
		print('x[1] = M2.Z Offset')
		vectposoffsetcounter = 0+2
		self.tilt_offset_start_index = vectposoffsetcounter
		for i in range(mapcounter):
			print(f'x[{vectposoffsetcounter}] = TILT_Y_map{i}')
			vectposoffsetcounter+=1
			print(f'x[{vectposoffsetcounter}] = TILT_X_map{i}')
			vectposoffsetcounter+=1
		self.tilt_offset_end_index = vectposoffsetcounter
		print(f'x[{vectposoffsetcounter}] = AST_O')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = AST_V')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = TRE_V')
		vectposoffsetcounter+=1
		print('...')
		# print(f'x[{vectposoffsetcounter}] = COMA_V')
		# vectposoffsetcounter+=1
		# print(f'x[{vectposoffsetcounter}] = COMA_H')
		# vectposoffsetcounter+=1
		# print(f'x[{vectposoffsetcounter}] = TRE_O')
		# vectposoffsetcounter+=1
		# print(f'x[{vectposoffsetcounter}] = QUAD_O')
		# vectposoffsetcounter+=1
		# print(f'x[{vectposoffsetcounter}] = AST2_O')
		# vectposoffsetcounter+=1
		# print(f'x[{vectposoffsetcounter}] = SPH')
		# vectposoffsetcounter+=1
		# print(f'x[{vectposoffsetcounter}] = AST2_V')
		# vectposoffsetcounter+=1
		# print(f'x[{vectposoffsetcounter}] = QUAD_V')

		self.number_of_maps = mapcounter
		self.fit_vec_size = 2+(2*mapcounter)+beam_class.zernike_polynomials.shape[0]-4 #vectposoffsetcounter+1

		self.tmpbeamclass = beam_class

		x0 = np.zeros(self.fit_vec_size)

		map_maxes = np.zeros(self.number_of_maps)

		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			tmpmax = np.amax(self.tmpbeamclass.trunc_maps[i])
			map_maxes[count] = tmpmax
		ampguess = max(map_maxes)

		x0[0] = ampguess
		x0[1] = beam_class.m2z_vals[f'map{int(self.number_of_maps/2.)}']

		self.x0 = x0

		self.fitting_counter = 0
		self.temp_cost = -999

	def chisquared(self,x):

		if self.fitting_counter%500==0:
			print('On fitting iteration = ',self.fitting_counter, ' with Cost = ',self.temp_cost)

		chi_squared = 0

		source_amp = x[0]
		M2z_offset = x[1]

		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

			ctmp[1] = x[tilt_counter]
			tilt_counter+=1
			ctmp[2] = x[tilt_counter]
			tilt_counter+=1

			ctmp[3] = x[self.tilt_offset_end_index]
			ctmp[5:] = x[self.tilt_offset_end_index+1:]

			modelbeam = source_amp*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+M2z_offset,
							   del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.)


			residual = self.tmpbeamclass.trunc_maps[i] - modelbeam

			chi_squared+= np.mean(residual**2)
		self.fitting_counter+=1
		self.temp_cost = chi_squared
		self.strehl_ratio = np.amax(self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=0,
							   del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.))
		return np.sqrt(chi_squared)


def plot_fit_results(beamclass,fitclass,results,vmax_frac_of_source_flux = 0.2,resids_stretch=5,
	                 title=None,savefigname=None,showplot=False):

	plt.figure(figsize=(15,8))
	# plt.figure()
	subplotcounter=1
	for count,i in enumerate(beamclass.trunc_maps):
		plt.subplot(3,fitclass.number_of_maps,subplotcounter)
		corners_tmp = np.rad2deg(enmap.corners(beamclass.trunc_maps[i].shape,beamclass.trunc_maps[i].wcs))
		imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
		plt.imshow(beamclass.trunc_maps[i],extent=imextent_tmp,origin='lower',vmin=-50,vmax=vmax_frac_of_source_flux*results.x[0])
		plt.xticks([])
		post_stamp_size = 1.5*60.#1.5/60.
		plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
		plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
		if (subplotcounter)%fitclass.number_of_maps!=1:
			plt.yticks([])
		subplotcounter+=1
	tilt_counter = 0
	for count,i in enumerate(beamclass.trunc_maps):
		plt.subplot(3,fitclass.number_of_maps,subplotcounter)
		ctmp = np.zeros(beamclass.zernike_polynomials.shape[0])
		ctmp[1] = results.x[fitclass.tilt_offset_start_index+tilt_counter]
		tilt_counter+=1
		ctmp[2] = results.x[fitclass.tilt_offset_start_index+tilt_counter]
		tilt_counter+=1

		ctmp[3] = results.x[fitclass.tilt_offset_end_index]
		ctmp[5:] = results.x[fitclass.tilt_offset_end_index+1:]

		tmpmodelbeam = results.x[0]*beamclass.make_psf(c=ctmp,secondary_offset=beamclass.m2z_vals[i]+results.x[1])
		
		corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
		imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
		plt.imshow(tmpmodelbeam,extent=imextent_tmp,origin='lower',vmin=-50,vmax=vmax_frac_of_source_flux*results.x[0])
		plt.xticks([])
		plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
		plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
		if (subplotcounter)%fitclass.number_of_maps!=1:
			plt.yticks([])
		subplotcounter+=1
	tilt_counter = 0
	for count,i in enumerate(beamclass.trunc_maps):
		plt.subplot(3,fitclass.number_of_maps,subplotcounter)
		ctmp = np.zeros(beamclass.zernike_polynomials.shape[0])
		ctmp[1] = results.x[fitclass.tilt_offset_start_index+tilt_counter]
		tilt_counter+=1
		ctmp[2] = results.x[fitclass.tilt_offset_start_index+tilt_counter]
		tilt_counter+=1

		ctmp[3] = results.x[fitclass.tilt_offset_end_index]
		ctmp[5:] = results.x[fitclass.tilt_offset_end_index+1:]

		tmpmodelbeam = results.x[0]*beamclass.make_psf(c=ctmp,secondary_offset=beamclass.m2z_vals[i]+results.x[1])
		
		corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
		imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
		plt.imshow(100*(beamclass.trunc_maps[i]-tmpmodelbeam)/results.x[0],extent=imextent_tmp,origin='lower',vmin=-resids_stretch,vmax=resids_stretch)
		plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
		plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
		if (subplotcounter)%fitclass.number_of_maps!=1:
			plt.yticks([])

		subplotcounter+=1

	fig = plt.gcf()

	nrows = 3
	ncols = fitclass.number_of_maps

	# Optional: leave some room on the right for colorbars
	fig.subplots_adjust(right=0.88)

	cbar_labels = ['mJy/beam','mJy/beam','% Source Flux']
	row_labels = ["data", "model", "residuals"]

	for row in range(nrows):



		# index of last subplot in this row (1-based for plt.subplot)
		last_idx = (row + 1) * ncols
		ax = plt.subplot(nrows, ncols, last_idx)

		# Get position of this last axes
		pos = ax.get_position()

		# Make a new axes for the colorbar, just to the right of it
		cax = fig.add_axes([
			pos.x1 + 0.005,   # a bit to the right of the last axes
			pos.y0,           # same bottom
			0.01,             # narrow width
			pos.height        # same height
		])

		# Get the image plotted in this axes (the first imshow)
		im = ax.images[0]

		# Make a colorbar for that row
		cb = fig.colorbar(im, cax=cax)
		cb.set_label(cbar_labels[row],rotation=-90,labelpad=20)

		first_idx = ((row) * ncols)+1
		ax = plt.subplot(nrows, ncols, first_idx)
		pos = ax.get_position()


		# y-coordinate of the center of this row
		y = pos.y0 + pos.height/2

		# Put text slightly left of this row
		fig.text(
			pos.x0 - 0.03,     # shift a little left of the first subplot
			y,
			row_labels[row],
			va='center', ha='right', fontsize=14
		)
	plt.suptitle(title)
	if savefigname is not None:
		plt.savefig(savefigname,bbox_inches='tight')

	if showplot:
		plt.show()
	else:
		plt.close()


def save_results(results,fitclass,savefilename):
	zernike_labels = ['AST_O','AST_V','TRE_V','COMA_V','COMA_H','TRE_O','QUAD_O','AST2_O','SPH','AST2_V','QUAD_V']
	results_dict = {}
	results_dict['source_amp'] = results.x[0]
	results_dict['M2.Z_offset'] = results.x[1]
	results_dict['strehl_ratio'] = fitclass.strehl_ratio
	map_counter = 0
	tmpind = fitclass.tilt_offset_start_index
	while tmpind<fitclass.tilt_offset_end_index:
		results_dict[f'Tilt_Y_map{map_counter}'] = results.x[tmpind]
		tmpind+=1
		results_dict[f'Tilt_X_map{map_counter}'] = results.x[tmpind]
		tmpind+=1
		map_counter+=1
	for i,val in enumerate(results.x[fitclass.tilt_offset_end_index:]):
		if i < len(zernike_labels):
			zernlabel = zernike_labels[i]
			# zernlabel = f'Noll {i}'
		else:
			zernlabel = f'OSA Index {i+4}'
			# zernlabel = zernike_labels[i]
		results_dict[zernlabel] = val*1E6*(np.mean(fitclass.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))
	for i in results_dict:
		print(i+' ', results_dict[i])
	with open(savefilename, "w") as f:
		json.dump(results_dict, f, indent=4)

















