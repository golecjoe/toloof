from scipy.optimize import curve_fit,minimize
import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap, enplot, reproject, utils, curvedsky,wcsutils
import json
from scipy.optimize import minimize, least_squares
import os
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec



from beamclass import Beam



class fit_beam_with_M2_offsets:

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
		print(f'x[{vectposoffsetcounter}] = del M2.X')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = del M2.Y')
		vectposoffsetcounter+=1
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

			ctmp[3] = x[self.tilt_offset_end_index+2]
			ctmp[5] = x[self.tilt_offset_end_index+3]
			ctmp[6] = x[self.tilt_offset_end_index+4]
			ctmp[9:] = x[self.tilt_offset_end_index+5:]

			modelbeam = source_amp*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+M2z_offset,
							   del_x=x[self.tilt_offset_end_index],del_y=x[self.tilt_offset_end_index+1],del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.)


			residual = self.tmpbeamclass.trunc_maps[i] - modelbeam

			chi_squared+= np.mean(residual**2)
		self.fitting_counter+=1
		self.temp_cost = chi_squared
		self.strehl_ratio = np.amax(self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=0,
							   del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.))
		return np.sqrt(chi_squared)

	def run_fitter(self):

	

		results = minimize(self.chisquared,x0=self.x0)
		self.results = results

	def plot_fit_results(self,vmax_frac_of_source_flux = 0.2,resids_stretch=5,vmin_val = -50,
					 title=None,savefigname=None,showplot=False):

		plt.figure(figsize=(15,8))
		# plt.figure()
		subplotcounter=1
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			corners_tmp = np.rad2deg(enmap.corners(self.tmpbeamclass.trunc_maps[i].shape,self.tmpbeamclass.trunc_maps[i].wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(self.tmpbeamclass.trunc_maps[i],extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			post_stamp_size = 1.5*60.#1.5/60.
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
			ctmp[1] = self.results.x[tilt_counter]
			tilt_counter+=1
			ctmp[2] = self.results.x[tilt_counter]
			tilt_counter+=1

			ctmp[3] = self.results.x[self.tilt_offset_end_index+2]
			ctmp[5] = self.results.x[self.tilt_offset_end_index+3]
			ctmp[6] = self.results.x[self.tilt_offset_end_index+4]
			ctmp[9:] = self.results.x[self.tilt_offset_end_index+5:]

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
																		 del_x=self.results.x[self.tilt_offset_end_index],del_y=self.results.x[self.tilt_offset_end_index+1])
			
			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(tmpmodelbeam,extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate((self.tmpbeamclass.trunc_maps)):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
			ctmp[1] = self.results.x[tilt_counter]
			tilt_counter+=1
			ctmp[2] = self.results.x[tilt_counter]
			tilt_counter+=1

			ctmp[3] = self.results.x[self.tilt_offset_end_index+2]
			ctmp[5] = self.results.x[self.tilt_offset_end_index+3]
			ctmp[6] = self.results.x[self.tilt_offset_end_index+4]
			ctmp[9:] = self.results.x[self.tilt_offset_end_index+5:]

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
																	del_x=self.results.x[self.tilt_offset_end_index],del_y=self.results.x[self.tilt_offset_end_index+1])
			
			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(100*(self.tmpbeamclass.trunc_maps[i]-tmpmodelbeam)/self.results.x[0],extent=imextent_tmp,origin='lower',vmin=-resids_stretch,vmax=resids_stretch)
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])

			subplotcounter+=1

		fig = plt.gcf()

		nrows = 3
		ncols = self.number_of_maps

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

	def save_results(self,savefilename):
		zernike_labels = ['AST_O','AST_V','TRE_V','TRE_O','QUAD_O','AST2_O','SPH','AST2_V','QUAD_V']
		results_dict = {}
		results_dict['source_amp'] = self.results.x[0]
		results_dict['M2.Z_offset'] = self.results.x[1]
		results_dict['strehl_ratio'] = self.strehl_ratio
		map_counter = 0
		tmpind = self.tilt_offset_start_index
		while tmpind<self.tilt_offset_end_index:
			results_dict[f'Tilt_Y_map{map_counter}'] = self.results.x[tmpind]
			tmpind+=1
			results_dict[f'Tilt_X_map{map_counter}'] = self.results.x[tmpind]
			tmpind+=1
			map_counter+=1

		results_dict['M2.X_offset'] = self.results.x[self.tilt_offset_end_index]
		results_dict['M2.Y_offset'] = self.results.x[self.tilt_offset_end_index+1]

		results_dict['AST_O'] = self.results.x[self.tilt_offset_end_index+2]*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))
		results_dict['AST_V'] = self.results.x[self.tilt_offset_end_index+3]*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))
		results_dict['TRE_V'] = self.results.x[self.tilt_offset_end_index+4]*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))

		for i in np.arange(self.results.x[self.tilt_offset_end_index+5:].size):
			if (i+3)<len(zernike_labels):
				zernlabel = zernike_labels[i+3]
			else:
				zernlabel = f'OSA Index {i+3+4}'

			results_dict[zernlabel] = self.results.x[self.tilt_offset_end_index+5+i]*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))

		for i in results_dict:
			print(i+' ', results_dict[i])

		for key, value in results_dict.items():
			if isinstance(value, np.ndarray):
				results_dict[key] = value.tolist()
		with open(savefilename, "w") as f:
			json.dump(results_dict, f, indent=4)



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
			print(f'x[{vectposoffsetcounter}] = AZ_Off_map{i}')
			vectposoffsetcounter+=1
			print(f'x[{vectposoffsetcounter}] = EL_Off_map{i}')
			vectposoffsetcounter+=1
		self.tilt_offset_end_index = vectposoffsetcounter
		print(f'x[{vectposoffsetcounter}] = AST_O')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = AST_V')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = TRE_V')
		vectposoffsetcounter+=1
		print('...')


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

			# ctmp[1] = x[tilt_counter]
			# tilt_counter+=1
			# ctmp[2] = x[tilt_counter]
			# tilt_counter+=1

			ctmp[3] = x[self.tilt_offset_end_index]
			ctmp[5:] = x[self.tilt_offset_end_index+1:]

			modelbeam = source_amp*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+M2z_offset,
							   del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.)

			az_off_deg = x[tilt_counter]/3600.
			tilt_counter+=1
			el_off_deg = x[tilt_counter]/3600.
			tilt_counter+=1

			pix1_offset = el_off_deg/abs(modelbeam.wcs.wcs.cdelt[1])
			pix2_offset = az_off_deg/abs(modelbeam.wcs.wcs.cdelt[0])

			modelbeam = enmap.fractional_shift(modelbeam,[pix1_offset,pix2_offset],keepwcs=True)

			residual = self.tmpbeamclass.trunc_maps[i] - modelbeam

			chi_squared+= np.mean(residual**2)
		self.fitting_counter+=1
		self.temp_cost = chi_squared
		self.strehl_ratio = np.amax(self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=0,
							   del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.))
		return np.sqrt(chi_squared)

	def run_fitter(self):
		# results = minimize(self.chisquared,x0=self.x0)
		# self.results = results
		self.results = minimize(
			self.chisquared,
			x0=self.x0,
			method="Powell",
			options={
				#"maxiter": 300,   # raise if you need tighter convergence
				"xtol": 1e-3,     # parameter tolerance
				"ftol": 1e-3,     # cost tolerance
				"disp": True,     # print progress
			},
		)


	def plot_fit_results(self,vmax_frac_of_source_flux = 0.2,resids_stretch=5,vmin_val = -50,
					 title=None,savefigname=None,showplot=False):

		plt.figure(figsize=(15,8))
		# plt.figure()
		subplotcounter=1
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			corners_tmp = np.rad2deg(enmap.corners(self.tmpbeamclass.trunc_maps[i].shape,self.tmpbeamclass.trunc_maps[i].wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(self.tmpbeamclass.trunc_maps[i],extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			post_stamp_size = 1.5*60.#1.5/60.
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])


			az_off_deg = self.results.x[tilt_counter]/3600.
			tilt_counter+=1
			el_off_deg = self.results.x[tilt_counter]/3600.
			tilt_counter+=1

			ctmp[3] = self.results.x[self.tilt_offset_end_index]
			ctmp[5:] = self.results.x[self.tilt_offset_end_index+1:]

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1])
			
			pix1_offset = el_off_deg/abs(tmpmodelbeam.wcs.wcs.cdelt[1])
			pix2_offset = az_off_deg/abs(tmpmodelbeam.wcs.wcs.cdelt[0])

			tmpmodelbeam = enmap.fractional_shift(tmpmodelbeam,[pix1_offset,pix2_offset],keepwcs=True)

			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(tmpmodelbeam,extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
			
			az_off_deg = self.results.x[tilt_counter]/3600.
			tilt_counter+=1
			el_off_deg = self.results.x[tilt_counter]/3600.
			tilt_counter+=1

			ctmp[3] = self.results.x[self.tilt_offset_end_index]
			ctmp[5:] = self.results.x[self.tilt_offset_end_index+1:]

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1])
			
			pix1_offset = el_off_deg/abs(tmpmodelbeam.wcs.wcs.cdelt[1])
			pix2_offset = az_off_deg/abs(tmpmodelbeam.wcs.wcs.cdelt[0])

			tmpmodelbeam = enmap.fractional_shift(tmpmodelbeam,[pix1_offset,pix2_offset],keepwcs=True)

			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(100*(self.tmpbeamclass.trunc_maps[i]-tmpmodelbeam)/self.results.x[0],extent=imextent_tmp,origin='lower',vmin=-resids_stretch,vmax=resids_stretch)
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])

			subplotcounter+=1

		fig = plt.gcf()

		nrows = 3
		ncols = self.number_of_maps

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


	def save_results(self,savefilename):
		zernike_labels = ['AST_O','AST_V','TRE_V','COMA_V','COMA_H','TRE_O','QUAD_O','AST2_O','SPH','AST2_V','QUAD_V']
		results_dict = {}
		results_dict['source_amp'] = self.results.x[0]
		results_dict['M2.Z_offset'] = self.results.x[1]
		results_dict['strehl_ratio'] = self.strehl_ratio
		map_counter = 0
		tmpind = self.tilt_offset_start_index
		while tmpind<self.tilt_offset_end_index:
			results_dict[f'Az_Off_map{map_counter}'] = self.results.x[tmpind]
			tmpind+=1
			results_dict[f'El_Off_map{map_counter}'] = self.results.x[tmpind]
			tmpind+=1
			map_counter+=1
		for i,val in enumerate(self.results.x[self.tilt_offset_end_index:]):
			if i < len(zernike_labels):
				zernlabel = zernike_labels[i]
				# zernlabel = f'Noll {i}'
			else:
				zernlabel = f'OSA Index {i+4}'
				# zernlabel = zernike_labels[i]
			results_dict[zernlabel] = val*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))
		for i in results_dict:
			print(i+' ', results_dict[i])
		for key, value in results_dict.items():
				if isinstance(value, np.ndarray):
					results_dict[key] = value.tolist()
		with open(savefilename, "w") as f:
			json.dump(results_dict, f, indent=4)

class fit_beam_with_pointing_tilt_offsets:

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
			print(f'x[{vectposoffsetcounter}] = AZ_Off_map{i}')
			vectposoffsetcounter+=1
			print(f'x[{vectposoffsetcounter}] = EL_Off_map{i}')
			vectposoffsetcounter+=1
		self.tilt_offset_end_index = vectposoffsetcounter
		print(f'x[{vectposoffsetcounter}] = AST_O')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = AST_V')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = TRE_V')
		vectposoffsetcounter+=1
		print('...')


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

			# az_off_deg = x[tilt_counter]/3600.
			# tilt_counter+=1
			# el_off_deg = x[tilt_counter]/3600.
			# tilt_counter+=1

			# pix1_offset = el_off_deg/abs(modelbeam.wcs.wcs.cdelt[1])
			# pix2_offset = az_off_deg/abs(modelbeam.wcs.wcs.cdelt[0])

			# modelbeam = enmap.fractional_shift(modelbeam,[pix1_offset,pix2_offset],keepwcs=True)

			residual = self.tmpbeamclass.trunc_maps[i] - modelbeam

			chi_squared+= np.mean(residual**2)
		self.fitting_counter+=1
		self.temp_cost = np.sqrt(chi_squared)
		
		return np.sqrt(chi_squared)

	def make_bounds(self,tilt_plusminus,zern_plusminus):
		tmpbounds = []

		for i in range(len(self.x0)):
			if i==0:
				tmpbounds.append((0.75*self.x0[0],1.25*self.x0[0]))
			elif i==1:
				tmpbounds.append((-4E-3,4E-3))
			elif i in [2,3,4]:
				tmpbounds.append((-tilt_plusminus,tilt_plusminus))
			else:
				tmpbounds.append((-zern_plusminus,zern_plusminus))
		tmpbounds = tuple(tmpbounds)
		return tmpbounds



	def run_fitter(self,stop_delta_cost=None,patience=None):
		# results = minimize(self.chisquared,x0=self.x0)
		# self.results = results

		tmpbound = self.make_bounds(800,300)

		self.temp_cost = 1E6
		if stop_delta_cost is None:
			stop_delta_cost = 0.010  # mJy/beam = 10 uJy/beam
		if patience is None:
			patience = 50             # require x consecutive small changes
		previous_cost = [None]
		small_change_counter = [0]

		def callback(xk):
			current_cost = self.temp_cost

			if previous_cost[0] is not None:
				delta_cost = abs(previous_cost[0] - current_cost)

				if delta_cost < stop_delta_cost:
					small_change_counter[0] += 1

					print(
						f"Small cost change {small_change_counter[0]}/{patience}: "
						f"delta = {delta_cost:.6g}: "
						f"cost = {current_cost:.6g}: "
					)

					if small_change_counter[0] >= patience:
						print(
							f"Stopping: cost change has been less than "
							f"{stop_delta_cost:.6g} for {patience} consecutive iterations."
						)
						raise StopIteration

				else:
					small_change_counter[0] = 0

			previous_cost[0] = current_cost

		self.results = minimize(
			self.chisquared,
			x0=self.x0,
			callback=callback,
			# bounds=tmpbound,
			# tol=1E-9,
			#method="L-BFGS-B",
			options={
				#"maxiter": 9000,   # raise if you need tighter convergence
				#"xtol": 1e-3,     # parameter tolerance
				#"ftol": 1.e-7,     # cost tolerance
				#"disp": True,     # print progress
			},
		)
		ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
		ctmp[3] = self.results.x[self.tilt_offset_end_index]
		ctmp[5:] = self.results.x[self.tilt_offset_end_index+1:]
		self.strehl_ratio = np.amax(self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.results.x[1],
							   del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.))


	def plot_fit_results(self,vmax_frac_of_source_flux = 0.2,resids_stretch=5,vmin_val = -50,
					 title=None,savefigname=None,showplot=False):

		#plt.figure(figsize=(15,8))
		fig = plt.figure(figsize=(3.0*self.number_of_maps + 1.2, 8))
		# plt.figure()
		subplotcounter=1
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			corners_tmp = np.rad2deg(enmap.corners(self.tmpbeamclass.trunc_maps[i].shape,self.tmpbeamclass.trunc_maps[i].wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(self.tmpbeamclass.trunc_maps[i],extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			post_stamp_size = 1.5*60.#1.5/60.
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

			ctmp[1] = self.results.x[tilt_counter]
			tilt_counter+=1
			ctmp[2] = self.results.x[tilt_counter]
			tilt_counter+=1


			ctmp[3] = self.results.x[self.tilt_offset_end_index]
			ctmp[5:] = self.results.x[self.tilt_offset_end_index+1:]

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1])
			

			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(tmpmodelbeam,extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			if subplotcounter == self.number_of_maps + 1:
				plt.ylabel('El Offset (arcsec)')
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

			ctmp[1] = self.results.x[tilt_counter]
			tilt_counter+=1
			ctmp[2] = self.results.x[tilt_counter]
			tilt_counter+=1
			

			ctmp[3] = self.results.x[self.tilt_offset_end_index]
			ctmp[5:] = self.results.x[self.tilt_offset_end_index+1:]

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1])
			

			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(100*(self.tmpbeamclass.trunc_maps[i]-tmpmodelbeam)/self.results.x[0],extent=imextent_tmp,origin='lower',vmin=-resids_stretch,vmax=resids_stretch)
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if subplotcounter == 2*self.number_of_maps + 2:
				plt.xlabel('Az Offset (arcsec)')
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])

			subplotcounter+=1

		#fig = plt.gcf()

		nrows = 3
		ncols = self.number_of_maps

		# Optional: leave some room on the right for colorbars
		#fig.subplots_adjust(right=0.88)

		fig.subplots_adjust(
			left=0.10,
			right=0.88,
			bottom=0.06,
			top=0.92,
			wspace=0.03,
			hspace=0.20
		)

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
				pos.x0 - 0.06,     # shift a little left of the first subplot
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


	def save_results(self,savefilename):
		zernike_labels = ['AST_O','AST_V','TRE_V','COMA_V','COMA_H','TRE_O','QUAD_O','AST2_O','SPH','AST2_V','QUAD_V']
		results_dict = {}
		results_dict['source_amp'] = self.results.x[0]
		results_dict['M2.Z_offset'] = self.results.x[1]
		results_dict['strehl_ratio'] = self.strehl_ratio
		map_counter = 0
		tmpind = self.tilt_offset_start_index
		while tmpind<self.tilt_offset_end_index:
			results_dict[f'Az_Off_map{map_counter}'] = self.results.x[tmpind]
			tmpind+=1
			results_dict[f'El_Off_map{map_counter}'] = self.results.x[tmpind]
			tmpind+=1
			map_counter+=1
		for i,val in enumerate(self.results.x[self.tilt_offset_end_index:]):
			if i < len(zernike_labels):
				zernlabel = zernike_labels[i]
				# zernlabel = f'Noll {i}'
			else:
				zernlabel = f'OSA Index {i+4}'
				# zernlabel = zernike_labels[i]
			results_dict[zernlabel] = val#*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))
		for i in results_dict:
			print(i+' ', results_dict[i])
		for key, value in results_dict.items():
				if isinstance(value, np.ndarray):
					results_dict[key] = value.tolist()
		with open(savefilename, "w") as f:
			json.dump(results_dict, f, indent=4)

	def surface_plot(self,savefilename,vmin=-400,vmax=400):
		cfit = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
		cfit[3] = self.results.x[self.tilt_offset_end_index]
		cfit[5:] = self.results.x[self.tilt_offset_end_index+1:]
		OPD = np.tensordot(cfit, self.tmpbeamclass.zernike_polynomials, axes=([0],[0]))
		plt.figure()
		plt.imshow(OPD,extent=([-self.tmpbeamclass.L/2.,self.tmpbeamclass.L/2.,-self.tmpbeamclass.L/2.,self.tmpbeamclass.L/2.]),
			vmin=vmin,vmax=vmax)
		#plt.imshow(OPD)

		plt.xlim(-30,30)
		plt.ylim(-30,30)
		plt.xlabel('x (meters)')
		plt.ylabel('y (meters)')
		cbar = plt.colorbar()
		cbar.set_label('OPD (microns)',rotation=-90,labelpad=20)
		plt.savefig(savefilename,bbox_inches='tight')
		plt.close()



class fit_beam_with_M2_offsets_globTilt:

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
		for i in range(1):
			print(f'x[{vectposoffsetcounter}] = TILT_Y_map{i}')
			vectposoffsetcounter+=1
			print(f'x[{vectposoffsetcounter}] = TILT_X_map{i}')
			vectposoffsetcounter+=1
		self.tilt_offset_end_index = vectposoffsetcounter
		print(f'x[{vectposoffsetcounter}] = del M2.X')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = del M2.Y')
		vectposoffsetcounter+=1
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
		self.fit_vec_size = 2+(2*1)+beam_class.zernike_polynomials.shape[0]-4 #vectposoffsetcounter+1

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
			#tilt_counter+=1
			ctmp[2] = x[tilt_counter+1]
			#tilt_counter+=1

			ctmp[3] = x[self.tilt_offset_end_index+2]
			ctmp[5] = x[self.tilt_offset_end_index+3]
			ctmp[6] = x[self.tilt_offset_end_index+4]
			ctmp[9:] = x[self.tilt_offset_end_index+5:]

			modelbeam = source_amp*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+M2z_offset,
							   del_x=x[self.tilt_offset_end_index],del_y=x[self.tilt_offset_end_index+1],del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.)


			residual = self.tmpbeamclass.trunc_maps[i] - modelbeam

			chi_squared+= np.mean(residual**2)
		self.fitting_counter+=1
		self.temp_cost = chi_squared
		self.strehl_ratio = np.amax(self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=0,
							   del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.))
		return np.sqrt(chi_squared)/3.

	def run_fitter(self):
		self.x0[1] = 0.

		self.x0[4] = 0.0#0.2E-3
		self.x0[5] = 0.0#2.5E-3

		self.x0[6] = 0.0#-53.5*1E-6*np.sqrt(2)*(2.*np.pi/np.mean(self.tmpbeamclass.wavelengths))
		self.x0[7] = 0.0#-45*1E-6*np.sqrt(2)*(2.*np.pi/np.mean(self.tmpbeamclass.wavelengths))
		self.x0[8] = 0.0#20*1E-6*np.sqrt(2)*(2.*np.pi/np.mean(self.tmpbeamclass.wavelengths))
		self.x0[9] = 0.0#-43*1E-6*np.sqrt(2)*(2.*np.pi/np.mean(self.tmpbeamclass.wavelengths))
		self.x0[10] = 0.0#19*1E-6*np.sqrt(2)*(2.*np.pi/np.mean(self.tmpbeamclass.wavelengths))

		lower_bounds = np.full_like(self.x0, -np.inf, dtype=float)
		upper_bounds = np.full_like(self.x0,  np.inf, dtype=float)

		lower_bounds[0] = 0.75*self.x0[0]
		upper_bounds[0] = 1.25*self.x0[0]

		lower_bounds[1] = -4.E-3
		upper_bounds[1] = 4.E-3

		lower_bounds[self.tilt_offset_end_index] = -4.E-3
		upper_bounds[self.tilt_offset_end_index] = 4.E-3

		lower_bounds[self.tilt_offset_end_index+1] = -4.E-3
		upper_bounds[self.tilt_offset_end_index+1] = 4.E-3

		lower_bounds[self.tilt_offset_end_index+2:] = -200.*1E-6*np.sqrt(2)*(2.*np.pi/np.mean(self.tmpbeamclass.wavelengths))
		upper_bounds[self.tilt_offset_end_index+2:] = 200.*1E-6*np.sqrt(2)*(2.*np.pi/np.mean(self.tmpbeamclass.wavelengths))

		bounds = list(zip(lower_bounds, upper_bounds))
		results = minimize(self.chisquared,x0=self.x0,bounds=bounds,
							method="L-BFGS-B")#,
							# options ={
							# 		"maxiter": 500,
							# 		"maxfun": 2000,
							# 		#"ftol": 1e-7,
							# 		#"gtol": 1e-5,
							# 		#"disp": True
							# })
		self.results = results

	def plot_fit_results(self,vmax_frac_of_source_flux = 0.2,resids_stretch=5,vmin_val = -50,
					 title=None,savefigname=None,showplot=False):

		plt.figure(figsize=(15,8))
		# plt.figure()
		subplotcounter=1
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			corners_tmp = np.rad2deg(enmap.corners(self.tmpbeamclass.trunc_maps[i].shape,self.tmpbeamclass.trunc_maps[i].wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(self.tmpbeamclass.trunc_maps[i],extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			post_stamp_size = 1.5*60.#1.5/60.
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
			ctmp[1] = self.results.x[tilt_counter]
			#tilt_counter+=1
			ctmp[2] = self.results.x[tilt_counter+1]
			#tilt_counter+=1

			ctmp[3] = self.results.x[self.tilt_offset_end_index+2]
			ctmp[5] = self.results.x[self.tilt_offset_end_index+3]
			ctmp[6] = self.results.x[self.tilt_offset_end_index+4]
			ctmp[9:] = self.results.x[self.tilt_offset_end_index+5:]

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
																		 del_x=self.results.x[self.tilt_offset_end_index],del_y=self.results.x[self.tilt_offset_end_index+1])
			
			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(tmpmodelbeam,extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate((self.tmpbeamclass.trunc_maps)):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
			ctmp[1] = self.results.x[tilt_counter]
			#tilt_counter+=1
			ctmp[2] = self.results.x[tilt_counter+1]
			#tilt_counter+=1

			ctmp[3] = self.results.x[self.tilt_offset_end_index+2]
			ctmp[5] = self.results.x[self.tilt_offset_end_index+3]
			ctmp[6] = self.results.x[self.tilt_offset_end_index+4]
			ctmp[9:] = self.results.x[self.tilt_offset_end_index+5:]

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
																	del_x=self.results.x[self.tilt_offset_end_index],del_y=self.results.x[self.tilt_offset_end_index+1])
			
			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(100*(self.tmpbeamclass.trunc_maps[i]-tmpmodelbeam)/self.results.x[0],extent=imextent_tmp,origin='lower',vmin=-resids_stretch,vmax=resids_stretch)
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])

			subplotcounter+=1

		fig = plt.gcf()

		nrows = 3
		ncols = self.number_of_maps

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
	def surface_plot(self,savefilename,vmin=-400,vmax=400):
		cfit = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
		cfit[3] = self.results.x[self.tilt_offset_end_index]
		cfit[5:] = self.results.x[self.tilt_offset_end_index+1:]
		OPD = np.tensordot(cfit, self.tmpbeamclass.zernike_polynomials, axes=([0],[0]))
		plt.figure()
		plt.imshow(OPD,extent=([-self.tmpbeamclass.L/2.,self.tmpbeamclass.L/2.,-self.tmpbeamclass.L/2.,self.tmpbeamclass.L/2.]),
			vmin=vmin,vmax=vmax)
		#plt.imshow(OPD)

		plt.xlim(-30,30)
		plt.ylim(-30,30)
		plt.xlabel('x (meters)')
		plt.ylabel('y (meters)')
		cbar = plt.colorbar()
		cbar.set_label('OPD (microns)',rotation=-90,labelpad=20)
		plt.savefig(savefilename,bbox_inches='tight')
		plt.close()

	def save_results(self,savefilename):
		zernike_labels = ['AST_O','AST_V','TRE_V','TRE_O','QUAD_O','AST2_O','SPH','AST2_V','QUAD_V']
		results_dict = {}
		results_dict['source_amp'] = self.results.x[0]
		results_dict['M2.Z_offset'] = self.results.x[1]
		results_dict['strehl_ratio'] = self.strehl_ratio
		map_counter = 0
		tmpind = self.tilt_offset_start_index
		while tmpind<self.tilt_offset_end_index:
			results_dict[f'Tilt_Y_map{map_counter}'] = self.results.x[tmpind]
			tmpind+=1
			results_dict[f'Tilt_X_map{map_counter}'] = self.results.x[tmpind]
			tmpind+=1
			map_counter+=1

		results_dict['M2.X_offset'] = self.results.x[self.tilt_offset_end_index]
		results_dict['M2.Y_offset'] = self.results.x[self.tilt_offset_end_index+1]

		results_dict['AST_O'] = self.results.x[self.tilt_offset_end_index+2]*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))
		results_dict['AST_V'] = self.results.x[self.tilt_offset_end_index+3]*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))
		results_dict['TRE_V'] = self.results.x[self.tilt_offset_end_index+4]*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))

		for i in np.arange(self.results.x[self.tilt_offset_end_index+5:].size):
			if (i+3)<len(zernike_labels):
				zernlabel = zernike_labels[i+3]
			else:
				zernlabel = f'OSA Index {i+3+4}'

			results_dict[zernlabel] = self.results.x[self.tilt_offset_end_index+5+i]*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))

		for i in results_dict:
			print(i+' ', results_dict[i])

		for key, value in results_dict.items():
			if isinstance(value, np.ndarray):
				results_dict[key] = value.tolist()
		with open(savefilename, "w") as f:
			json.dump(results_dict, f, indent=4)


class fit_beam_with_pointing_tilt_offsets_wIllumination_leastsquares:

	def __init__(self,beam_class):

		print('Initializing the fit beam class ')


		mapcounter= 0 
		for i in beam_class.trunc_maps:
			mapcounter+=1
		print(f'There are {mapcounter} maps')
		print('The vector of model parameters will have the form:')
		print('x[0] = source_amplitude')
		print('x[1] = M2.Z Offset')
		print('x[2] = Illumination')
		vectposoffsetcounter = 0+3
		self.tilt_offset_start_index = vectposoffsetcounter
		for i in range(mapcounter):
			print(f'x[{vectposoffsetcounter}] = AZ_Off_map{i}')
			vectposoffsetcounter+=1
			print(f'x[{vectposoffsetcounter}] = EL_Off_map{i}')
			vectposoffsetcounter+=1
		self.tilt_offset_end_index = vectposoffsetcounter
		print(f'x[{vectposoffsetcounter}] = AST_O')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = AST_V')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = TRE_V')
		vectposoffsetcounter+=1
		print('...')


		self.number_of_maps = mapcounter
		self.fit_vec_size = 3+(2*mapcounter)+beam_class.zernike_polynomials.shape[0]-4 #vectposoffsetcounter+1

		self.tmpbeamclass = beam_class

		x0 = np.zeros(self.fit_vec_size)

		map_maxes = np.zeros(self.number_of_maps)

		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			tmpmax = np.amax(self.tmpbeamclass.trunc_maps[i])
			map_maxes[count] = tmpmax
		ampguess = max(map_maxes)

		x0[0] = ampguess
		x0[1] = beam_class.m2z_vals[f'map{int(self.number_of_maps/2.)}']
		x0[2] = 45.

		self.x0 = x0

		self.fitting_counter = 0
		self.temp_cost = -999

	def chisquared(self,x):

		if self.fitting_counter%500==0:
			print('On fitting iteration = ',self.fitting_counter, ' with Cost = ',self.temp_cost)

		chi_squared = 0

		source_amp = x[0]
		M2z_offset = x[1]

		self.tmpbeamclass.initialize_model(aperture_plane_resolution = 1.0,center_on_brightest_pix=True,
							  include_legs=True,plot_aperture=False,save_aperture=None,
							  aperture_fwhm = x[2],edge_taper_diameter=x[2],plot_illumination=False,
							  n=4,m=4)

		tilt_counter = self.tilt_offset_start_index
		residual_maps = {}
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

			# az_off_deg = x[tilt_counter]/3600.
			# tilt_counter+=1
			# el_off_deg = x[tilt_counter]/3600.
			# tilt_counter+=1

			# pix1_offset = el_off_deg/abs(modelbeam.wcs.wcs.cdelt[1])
			# pix2_offset = az_off_deg/abs(modelbeam.wcs.wcs.cdelt[0])

			# modelbeam = enmap.fractional_shift(modelbeam,[pix1_offset,pix2_offset],keepwcs=True)

			residual = self.tmpbeamclass.trunc_maps[i] - modelbeam
			residual_maps[i] = residual

			chi_squared+= residual**2

		tmpreturn = np.concatenate([residual.ravel() for residual in residual_maps.values()])
		self.fitting_counter+=1
		self.temp_cost = np.sqrt(np.mean(chi_squared))
		
		return tmpreturn

	def make_bounds(self,tilt_plusminus,zern_plusminus):
		tmpbounds = []

		for i in range(len(self.x0)):
			if i==0:
				tmpbounds.append((0.75*self.x0[0],1.25*self.x0[0]))
			elif i==1:
				tmpbounds.append((-4E-3,4E-3))
			elif i in [2,3,4]:
				tmpbounds.append((-tilt_plusminus,tilt_plusminus))
			else:
				tmpbounds.append((-zern_plusminus,zern_plusminus))
		tmpbounds = tuple(tmpbounds)
		return tmpbounds



	def run_fitter(self,stop_delta_cost=None,patience=None):
		# results = minimize(self.chisquared,x0=self.x0)
		# self.results = results

		tmpbound = self.make_bounds(800,300)

		self.temp_cost = 1E6
		if stop_delta_cost is None:
			stop_delta_cost = 0.010  # mJy/beam = 10 uJy/beam
		if patience is None:
			patience = 50             # require x consecutive small changes
		previous_cost = [None]
		small_change_counter = [0]

		def callback(xk):
			current_cost = self.temp_cost

			if previous_cost[0] is not None:
				delta_cost = abs(previous_cost[0] - current_cost)

				if delta_cost < stop_delta_cost:
					small_change_counter[0] += 1

					print(
						f"Small cost change {small_change_counter[0]}/{patience}: "
						f"delta = {delta_cost:.6g}: "
						f"cost = {current_cost:.6g}: "
					)

					if small_change_counter[0] >= patience:
						print(
							f"Stopping: cost change has been less than "
							f"{stop_delta_cost:.6g} for {patience} consecutive iterations."
						)
						raise StopIteration

				else:
					small_change_counter[0] = 0

			previous_cost[0] = current_cost

		self.results = least_squares(
			self.chisquared,
			self.x0,
			x_scale='jac'
		)
		self.tmpbeamclass.initialize_model(aperture_plane_resolution = 1.0,center_on_brightest_pix=True,
							  include_legs=True,plot_aperture=False,save_aperture=None,
							  aperture_fwhm = self.results.x[2],edge_taper_diameter=self.results.x[2],plot_illumination=False,
							  n=4,m=4)
		ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
		ctmp[3] = self.results.x[self.tilt_offset_end_index]
		ctmp[5:] = self.results.x[self.tilt_offset_end_index+1:]
		self.strehl_ratio = np.amax(self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.results.x[1],
							   del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.))


	def plot_fit_results(self,vmax_frac_of_source_flux = 0.2,resids_stretch=5,vmin_val = -50,
					 title=None,savefigname=None,showplot=False):

		#plt.figure(figsize=(15,8))
		fig = plt.figure(figsize=(3.0*self.number_of_maps + 1.2, 8))
		# plt.figure()
		subplotcounter=1
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			corners_tmp = np.rad2deg(enmap.corners(self.tmpbeamclass.trunc_maps[i].shape,self.tmpbeamclass.trunc_maps[i].wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(self.tmpbeamclass.trunc_maps[i],extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			post_stamp_size = 1.5*60.#1.5/60.
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

			ctmp[1] = self.results.x[tilt_counter]
			tilt_counter+=1
			ctmp[2] = self.results.x[tilt_counter]
			tilt_counter+=1


			ctmp[3] = self.results.x[self.tilt_offset_end_index]
			ctmp[5:] = self.results.x[self.tilt_offset_end_index+1:]

			self.tmpbeamclass.initialize_model(aperture_plane_resolution = 1.0,center_on_brightest_pix=True,
							  include_legs=True,plot_aperture=False,save_aperture=None,
							  aperture_fwhm = self.results.x[2],edge_taper_diameter=self.results.x[2],plot_illumination=False,
							  n=4,m=4)

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1])
			

			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(tmpmodelbeam,extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			if subplotcounter == self.number_of_maps + 1:
				plt.ylabel('El Offset (arcsec)')
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

			ctmp[1] = self.results.x[tilt_counter]
			tilt_counter+=1
			ctmp[2] = self.results.x[tilt_counter]
			tilt_counter+=1
			

			ctmp[3] = self.results.x[self.tilt_offset_end_index]
			ctmp[5:] = self.results.x[self.tilt_offset_end_index+1:]

			self.tmpbeamclass.initialize_model(aperture_plane_resolution = 1.0,center_on_brightest_pix=True,
							  include_legs=True,plot_aperture=False,save_aperture=None,
							  aperture_fwhm = self.results.x[2],edge_taper_diameter=self.results.x[2],plot_illumination=False,
							  n=4,m=4)

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1])
			

			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(100*(self.tmpbeamclass.trunc_maps[i]-tmpmodelbeam)/self.results.x[0],extent=imextent_tmp,origin='lower',vmin=-resids_stretch,vmax=resids_stretch)
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if subplotcounter == 2*self.number_of_maps + 2:
				plt.xlabel('Az Offset (arcsec)')
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])

			subplotcounter+=1

		#fig = plt.gcf()

		nrows = 3
		ncols = self.number_of_maps

		# Optional: leave some room on the right for colorbars
		#fig.subplots_adjust(right=0.88)

		fig.subplots_adjust(
			left=0.10,
			right=0.88,
			bottom=0.06,
			top=0.92,
			wspace=0.03,
			hspace=0.20
		)

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
				pos.x0 - 0.06,     # shift a little left of the first subplot
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


	def save_results(self,savefilename):
		zernike_labels = ['AST_O','AST_V','TRE_V','COMA_V','COMA_H','TRE_O','QUAD_O','AST2_O','SPH','AST2_V','QUAD_V']
		results_dict = {}
		results_dict['source_amp'] = self.results.x[0]
		results_dict['M2.Z_offset'] = self.results.x[1]
		results_dict['Illumination'] = self.results.x[2]
		results_dict['strehl_ratio'] = self.strehl_ratio
		map_counter = 0
		tmpind = self.tilt_offset_start_index
		while tmpind<self.tilt_offset_end_index:
			results_dict[f'Az_Off_map{map_counter}'] = self.results.x[tmpind]
			tmpind+=1
			results_dict[f'El_Off_map{map_counter}'] = self.results.x[tmpind]
			tmpind+=1
			map_counter+=1
		for i,val in enumerate(self.results.x[self.tilt_offset_end_index:]):
			if i < len(zernike_labels):
				zernlabel = zernike_labels[i]
				# zernlabel = f'Noll {i}'
			else:
				zernlabel = f'OSA Index {i+4}'
				# zernlabel = zernike_labels[i]
			results_dict[zernlabel] = val#*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))
		for i in results_dict:
			print(i+' ', results_dict[i])
		for key, value in results_dict.items():
				if isinstance(value, np.ndarray):
					results_dict[key] = value.tolist()
		with open(savefilename, "w") as f:
			json.dump(results_dict, f, indent=4)

	def surface_plot(self,savefilename,vmin=-400,vmax=400):
		cfit = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
		cfit[3] = self.results.x[self.tilt_offset_end_index]
		cfit[5:] = self.results.x[self.tilt_offset_end_index+1:]
		OPD = np.tensordot(cfit, self.tmpbeamclass.zernike_polynomials, axes=([0],[0]))
		plt.figure()
		plt.imshow(OPD,extent=([-self.tmpbeamclass.L/2.,self.tmpbeamclass.L/2.,-self.tmpbeamclass.L/2.,self.tmpbeamclass.L/2.]),
			vmin=vmin,vmax=vmax)
		#plt.imshow(OPD)

		plt.xlim(-30,30)
		plt.ylim(-30,30)
		plt.xlabel('x (meters)')
		plt.ylabel('y (meters)')
		cbar = plt.colorbar()
		cbar.set_label('OPD (microns)',rotation=-90,labelpad=20)
		plt.savefig(savefilename,bbox_inches='tight')
		plt.close()

class _LeastSquaresUncertainties:
	"""Common covariance estimation and exports for the least-squares fitters."""

	def save_uncertainties(self, saveuncertfilename):
		"""Save local Jacobian covariance and result variances after run_fitter.

		The supplied filename (with or without an extension) sets the common
		stem for .csv, .png and .json outputs. CSV rows/columns follow results.x
		and are labeled by parameter name. JSON mirrors save_results keys;
		values are variances, NOT standard deviations, in squared native units
		(meters, meters/m, degrees, degrees/m, microns, or source-amplitude units).

		Covariance is (J.T J)^-1 * sum(residual**2)/(N-p), computed by SVD
		with column scaling. This assumes independent equal-variance residuals
		and a locally linear model with inactive parameter bounds; it does not
		account for correlated map noise or telescope coordinate errors. Rank deficiency and N <= p are errors.
		Derived results use first-order covariance propagation. The recorded
		middle coordinate has zero variance; Strehl uses a numerical gradient
		of the same middle-map PSF peak as run_fitter (a local approximation).
		"""
		if not hasattr(self, 'results'):
			raise ValueError('Run the fitter before saving uncertainties')
		x = np.asarray(self.results.x, dtype=float)
		jac = np.asarray(self.results.jac, dtype=float)
		residuals = np.asarray(self.results.fun, dtype=float)
		if (jac.ndim != 2 or jac.shape != (residuals.size, x.size)
				or not np.all(np.isfinite(jac))
				or not np.all(np.isfinite(residuals))):
			raise ValueError('Expected a finite residual vector and matching Jacobian')
		dof = residuals.size - x.size
		if dof <= 0:
			raise ValueError('Covariance requires more residuals than fitted parameters')

		# Normalize columns so differing parameter units do not dominate the
		# rank test. Transform the resulting covariance back to native units.
		scales = np.linalg.norm(jac, axis=0)
		if np.any(scales == 0):
			raise ValueError('Jacobian is rank deficient; parameter covariance is undefined')
		_, singular, vt = np.linalg.svd(jac / scales, full_matrices=False)
		threshold = np.finfo(float).eps * max(jac.shape) * singular[0]
		if np.any(singular <= threshold):
			raise ValueError('Jacobian is rank deficient; parameter covariance is undefined')
		inverse = (vt.T / singular) / scales[:, None]
		covariance = (inverse @ inverse.T) * (np.sum(residuals**2) / dof)

		names, gradients = self._uncertainty_model(x)
		variances = {name: float(max(0., gradient @ covariance @ gradient))
			for name, gradient in gradients.items()}

		stem = os.path.splitext(os.fspath(saveuncertfilename))[0]
		# Use csv.writer so generic labels and future map names are quoted safely.
		import csv
		with open(stem + '.csv', 'w', newline='') as f:
			writer = csv.writer(f)
			writer.writerow(['parameter'] + names)
			for name, row in zip(names, covariance):
				writer.writerow([name] + row.tolist())
		with open(stem + '.json', 'w') as f:
			json.dump(variances, f, indent=4, allow_nan=False)

		# A symmetric logarithmic scale keeps covariances with different units
		# visible while preserving sign. The CSV always contains raw values.
		from matplotlib.colors import SymLogNorm
		limit = float(np.max(np.abs(covariance)))
		limit = limit if limit > 0 else 1.
		fig, ax = plt.subplots(figsize=(max(8, .55 * x.size), max(7, .5 * x.size)))
		try:
			im = ax.imshow(covariance, cmap='RdBu_r',
				norm=SymLogNorm(linthresh=limit * 1e-6, vmin=-limit, vmax=limit))
			ax.set_xticks(np.arange(x.size))
			ax.set_yticks(np.arange(x.size))
			ax.set_xticklabels(names, rotation=90)
			ax.set_yticklabels(names)
			ax.set_title('Jacobian covariance (native parameter units; symmetric log scale)')
			fig.colorbar(im, ax=ax,
				ticks=limit * np.array([-1., -1e-2, -1e-4, 0., 1e-4, 1e-2, 1.]),
				format='%.1e', label='Covariance (row unit × column unit)')
			fig.savefig(stem + '.png', dpi=200, bbox_inches='tight')
		finally:
			plt.close(fig)

		self.covariance_matrix = covariance
		self.uncertaintiesdict = variances
		self.uncertaintytable_fname = stem + '.png'


class fit_beam_with_pointing_tilt_offsets_leastsquares(_LeastSquaresUncertainties):
	"""Fit per-map pointing tilts with a shared axial focus offset.

	Modeled defocus is recorded absolute M2.Z + x[1], in meters.
	The absolute focus target is -x[1]; the reported move is from the
	middle map (in fitted map order) to that target.
	"""

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
		for map_name in beam_class.trunc_maps:
			print(f'x[{vectposoffsetcounter}] = Az_Off_{map_name}')
			vectposoffsetcounter+=1
			print(f'x[{vectposoffsetcounter}] = El_Off_{map_name}')
			vectposoffsetcounter+=1
		self.tilt_offset_end_index = vectposoffsetcounter
		print(f'x[{vectposoffsetcounter}] = AST_O')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = AST_V')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = TRE_V')
		vectposoffsetcounter+=1
		print('...')


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

		chi_squared = np.zeros(self.tmpbeamclass.trunc_maps['map0'].shape)

		source_amp = x[0]
		M2z_offset = x[1]

		tilt_counter = self.tilt_offset_start_index
		residual_maps = {}
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

			# az_off_deg = x[tilt_counter]/3600.
			# tilt_counter+=1
			# el_off_deg = x[tilt_counter]/3600.
			# tilt_counter+=1

			# pix1_offset = el_off_deg/abs(modelbeam.wcs.wcs.cdelt[1])
			# pix2_offset = az_off_deg/abs(modelbeam.wcs.wcs.cdelt[0])

			# modelbeam = enmap.fractional_shift(modelbeam,[pix1_offset,pix2_offset],keepwcs=True)

			residual = self.tmpbeamclass.trunc_maps[i] - modelbeam
			residual_maps[i] = residual

			chi_squared+= residual**2

		tmpreturn = np.concatenate([residual.ravel() for residual in residual_maps.values()])
		self.fitting_counter+=1
		self.temp_cost = np.sqrt(np.mean(chi_squared))
		
		return tmpreturn

	def make_bounds(self,tilt_plusminus,zern_plusminus):
		tmpbounds = []

		lowerbounds = []
		upperbounds = []

		for i in range(len(self.x0)):
			if i==0:
				# lowerbounds.append(0.7*self.x0[0])
				# upperbounds.append(1.3*self.x0[0])
				lowerbounds.append(0)
				upperbounds.append(np.inf)
			elif i==1:
				# lowerbounds.append(-5E-3)
				# upperbounds.append(5E-3)
				lowerbounds.append(-np.inf)
				upperbounds.append(np.inf)
			elif i in [2,3,4]:
				# lowerbounds.append(-tilt_plusminus)
				# upperbounds.append(tilt_plusminus)
				lowerbounds.append(-np.inf)
				upperbounds.append(np.inf)
			elif i==self.tilt_offset_end_index+8:
				# lowerbounds.append(-tilt_plusminus)
				# upperbounds.append(tilt_plusminus)
				lowerbounds.append(-250)
				upperbounds.append(250)
			else:
				# lowerbounds.append(-zern_plusminus)
				# upperbounds.append(zern_plusminus)
				lowerbounds.append(-np.inf)
				upperbounds.append(np.inf)
		lowerbounds = tuple(lowerbounds)
		upperbounds = tuple(upperbounds)
		return (lowerbounds,upperbounds)



	def run_fitter(self,stop_delta_cost=None,patience=None,tilt_plusminus=None,zern_plusminus=None,
					newx0 = None):
		# results = minimize(self.chisquared,x0=self.x0)
		# self.results = results

		if newx0 is not None:
			self.x0 = newx0

		tmpbound = self.make_bounds(1000,350)

		self.temp_cost = 1E6
		if stop_delta_cost is None:
			stop_delta_cost = 0.010  # mJy/beam = 10 uJy/beam
		if patience is None:
			patience = 50             # require x consecutive small changes
		previous_cost = [None]
		small_change_counter = [0]

		def callback(xk):
			current_cost = self.temp_cost

			if previous_cost[0] is not None:
				delta_cost = abs(previous_cost[0] - current_cost)

				if delta_cost < stop_delta_cost:
					small_change_counter[0] += 1

					print(
						f"Small cost change {small_change_counter[0]}/{patience}: "
						f"delta = {delta_cost:.6g}: "
						f"cost = {current_cost:.6g}: "
					)

					if small_change_counter[0] >= patience:
						print(
							f"Stopping: cost change has been less than "
							f"{stop_delta_cost:.6g} for {patience} consecutive iterations."
						)
						raise StopIteration

				else:
					small_change_counter[0] = 0

			previous_cost[0] = current_cost

		self.results = least_squares(
			self.chisquared,
			self.x0,
			x_scale='jac',
			bounds=tmpbound
		)
		ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
		ctmp[3] = self.results.x[self.tilt_offset_end_index]
		ctmp[5:] = self.results.x[self.tilt_offset_end_index+1:]
		map_names = list(self.tmpbeamclass.trunc_maps)
		middle_map = map_names[len(map_names) // 2]
		z_middle = float(self.tmpbeamclass.m2z_vals[middle_map])
		self.strehl_ratio = float(np.amax(self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=z_middle + self.results.x[1],
							   del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.)))


	def plot_fit_results(self,vmax_frac_of_source_flux = 0.2,resids_stretch=5,vmin_val = -50,
					 title=None,savefigname=None,showplot=False):

		#plt.figure(figsize=(15,8))
		fig = plt.figure(figsize=(3.0*self.number_of_maps + 1.2, 8))
		# plt.figure()
		subplotcounter=1
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			corners_tmp = np.rad2deg(enmap.corners(self.tmpbeamclass.trunc_maps[i].shape,self.tmpbeamclass.trunc_maps[i].wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(self.tmpbeamclass.trunc_maps[i],extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			post_stamp_size = 1.5*60.#1.5/60.
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

			ctmp[1] = self.results.x[tilt_counter]
			tilt_counter+=1
			ctmp[2] = self.results.x[tilt_counter]
			tilt_counter+=1


			ctmp[3] = self.results.x[self.tilt_offset_end_index]
			ctmp[5:] = self.results.x[self.tilt_offset_end_index+1:]

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1])
			

			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(tmpmodelbeam,extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			if subplotcounter == self.number_of_maps + 1:
				plt.ylabel('El Offset (arcsec)')
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

			ctmp[1] = self.results.x[tilt_counter]
			tilt_counter+=1
			ctmp[2] = self.results.x[tilt_counter]
			tilt_counter+=1
			

			ctmp[3] = self.results.x[self.tilt_offset_end_index]
			ctmp[5:] = self.results.x[self.tilt_offset_end_index+1:]

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1])
			

			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(100*(self.tmpbeamclass.trunc_maps[i]-tmpmodelbeam)/self.results.x[0],extent=imextent_tmp,origin='lower',vmin=-resids_stretch,vmax=resids_stretch)
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if subplotcounter == 2*self.number_of_maps + 2:
				plt.xlabel('Az Offset (arcsec)')
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])

			subplotcounter+=1

		#fig = plt.gcf()

		nrows = 3
		ncols = self.number_of_maps

		# Optional: leave some room on the right for colorbars
		#fig.subplots_adjust(right=0.88)

		fig.subplots_adjust(
			left=0.10,
			right=0.88,
			bottom=0.06,
			top=0.92,
			wspace=0.03,
			hspace=0.20
		)

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
				pos.x0 - 0.06,     # shift a little left of the first subplot
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
		self.datamodelresid_fname = savefigname


	def _fitted_zernikes(self, include_tilt=True, map_name=None):
		"""Reconstruct fitted Zernikes, using the middle map's pointing by default."""
		x = self.results.x
		c = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
		if include_tilt:
			map_names = list(self.tmpbeamclass.trunc_maps)
			map_index = len(map_names) // 2 if map_name is None else map_names.index(map_name)
			tilt_index = self.tilt_offset_start_index + 2 * map_index
			c[1:3] = x[tilt_index:tilt_index + 2]
		c[3] = x[self.tilt_offset_end_index]
		c[5:] = x[self.tilt_offset_end_index + 1:]
		return c

	def save_results(self, savefilename):

		savefilename = os.fspath(savefilename)

		# Friendly names for the Zernike modes you commonly use.
		# These correspond to OSA/ANSI indices 4 through 14.
		zernike_labels = {
			4:  'AST_O',
			5:  'AST_V',
			6:  'TRE_V',
			7:  'COMA_V',
			8:  'COMA_H',
			9:  'TRE_O',
			10: 'QUAD_O',
			11: 'AST2_O',
			12: 'SPH',
			13: 'AST2_V',
			14: 'QUAD_V',
		}

		results_dict = {}

		# --------------------------------------------------
		# General fit parameters
		# --------------------------------------------------

		results_dict['source_amp'] = self.results.x[0]
		# Model defocus = recorded absolute M2.Z + fitted offset.
		results_dict['M2.Z_offset'] = float(self.results.x[1])
		results_dict['M2.Z_target'] = -float(self.results.x[1])
		map_names = list(self.tmpbeamclass.trunc_maps)
		middle_map = map_names[len(map_names) // 2]
		results_dict['M2.Z_middle'] = float(self.tmpbeamclass.m2z_vals[middle_map])
		results_dict['M2.Z_move'] = (results_dict['M2.Z_target']
			- results_dict['M2.Z_middle'])
		results_dict['strehl_ratio'] = self.strehl_ratio

		# --------------------------------------------------
		# Pointing offsets
		# --------------------------------------------------

		for map_counter, map_name in enumerate(self.tmpbeamclass.trunc_maps):
			tmpind = self.tilt_offset_start_index + 2 * map_counter
			results_dict[f'Az_Off_{map_name}'] = self.results.x[tmpind]
			results_dict[f'El_Off_{map_name}'] = self.results.x[tmpind + 1]

		# --------------------------------------------------
		# Zernike coefficients
		# --------------------------------------------------

		zernike_results = self.results.x[self.tilt_offset_end_index:]

		for i, val in enumerate(zernike_results):

			osa_index = i + 4

			# Use your descriptive name when available.
			# Otherwise fall back to the OSA index.
			zernlabel = zernike_labels.get(
				osa_index,
				f'OSA Index {osa_index}'
			)

			results_dict[zernlabel] = val

		# --------------------------------------------------
		# Print results
		# --------------------------------------------------

		for key, value in results_dict.items():
			print(key, value)

		# --------------------------------------------------
		# Convert numpy arrays/scalars for JSON
		# --------------------------------------------------

		for key, value in results_dict.items():

			if isinstance(value, np.ndarray):
				results_dict[key] = value.tolist()

			elif isinstance(value, np.generic):
				results_dict[key] = value.item()

		# --------------------------------------------------
		# Save JSON
		# --------------------------------------------------

		with open(savefilename, "w") as f:
			json.dump(results_dict, f, indent=4)

		self.resultsdict = results_dict

		# --------------------------------------------------
		# Save results as a PNG table
		# --------------------------------------------------

		table_rows = []

		for label, value in results_dict.items():

			# Determine units from the parameter itself rather
			# than from its position in the dictionary.
			if label == 'source_amp':
				unit = 'mJy/beam'

			elif label in ('M2.Z_offset', 'M2.Z_target', 'M2.Z_middle', 'M2.Z_move'):
				unit = 'mm'

			elif label == 'strehl_ratio':
				unit = ''

			elif label.startswith('Az_Off_'):
				unit = 'microns'

			elif label.startswith('El_Off_'):
				unit = 'microns'

			else:
				# Remaining parameters are Zernike coefficients
				unit = 'microns'

			# --------------------------------------------------
			# Format values
			# --------------------------------------------------

			if (
				isinstance(value, (int, float, np.integer, np.floating))
				and label in ('M2.Z_offset', 'M2.Z_target', 'M2.Z_middle', 'M2.Z_move')
			):
				# Convert meters -> mm for display
				display_value = value * 1E3
				value_str = f"{display_value:.4f}"

			elif isinstance(
				value,
				(int, float, np.integer, np.floating)
			):
				value_str = f"{value:.4f}"

			else:
				value_str = str(value)

			table_rows.append([
				label,
				value_str,
				unit
			])

		# --------------------------------------------------
		# Make table figure
		# --------------------------------------------------

		fig_height = max(
			2.0,
			0.35 * (len(table_rows) + 1)
		)

		fig, ax = plt.subplots(
			figsize=(6, fig_height)
		)

		ax.axis("off")

		table = ax.table(
			cellText=table_rows,
			colLabels=["LABEL", "VALUE", "UNITS"],
			cellLoc="center",
			colLoc="center",
			loc="center",
		)

		table.auto_set_font_size(False)
		table.set_fontsize(11)
		table.scale(1.0, 1.4)

		# Style cells
		for (row, col), cell in table.get_celld().items():

			cell.set_edgecolor("black")
			cell.set_linewidth(1.0)

			if row == 0:
				cell.set_facecolor("0.9")
			else:
				cell.set_facecolor("0.96")

		# Same filename as JSON, but .png
		pngfilename = (
			os.path.splitext(savefilename)[0]
			+ ".png"
		)

		plt.savefig(
			pngfilename,
			dpi=200,
			bbox_inches="tight"
		)

		plt.close(fig)

		self.resulttable_fname = pngfilename

	def _uncertainty_model(self, x):
		"""Parameter names and result gradients for per-map pointing tilts."""
		zernike_labels = {
			3: 'AST_O', 5: 'AST_V', 6: 'TRE_V', 7: 'COMA_V', 8: 'COMA_H',
			9: 'TRE_O', 10: 'QUAD_O', 11: 'AST2_O', 12: 'SPH',
			13: 'AST2_V', 14: 'QUAD_V',
		}
		indices = [3] + list(range(5, self.tmpbeamclass.zernike_polynomials.shape[0]))
		map_names = list(self.tmpbeamclass.trunc_maps)
		names = ['source_amp', 'M2.Z_offset']
		for map_name in map_names:
			names += [f'Az_Off_{map_name}', f'El_Off_{map_name}']
		names += [zernike_labels.get(idx, f'OSA Index {idx}') for idx in indices]
		basis = np.eye(x.size)
		z_middle = float(self.tmpbeamclass.m2z_vals[map_names[len(map_names) // 2]])
		gradients = {
			'source_amp': basis[0],
			'M2.Z_offset': basis[1],
			'M2.Z_target': -basis[1],
			'M2.Z_middle': np.zeros(x.size),
			'M2.Z_move': -basis[1],
		}

		def strehl_at(parameters):
			c = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
			c[indices] = parameters[self.tilt_offset_end_index:]
			return float(np.amax(self.tmpbeamclass.make_psf(
				c=c, secondary_offset=z_middle + parameters[1],
				del_x=0., del_y=0., del_alph_x=0., del_alph_y=0.,
				f=17.5, F=525., D=50.,
			)))

		strehl_gradient = np.zeros(x.size)
		for idx in [1] + list(range(self.tilt_offset_end_index, x.size)):
			step = np.cbrt(np.finfo(float).eps) * max(abs(x[idx]), 1e-3 if idx == 1 else 1.)
			delta = step * basis[idx]
			strehl_gradient[idx] = (strehl_at(x + delta) - strehl_at(x - delta)) / (2 * step)
		gradients['strehl_ratio'] = strehl_gradient
		for idx in range(2, x.size):
			gradients[names[idx]] = basis[idx]
		return names, gradients

	def surface_plot(self,savefilename,vmin=-400,vmax=400):
		cfit = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
		cfit[3] = self.results.x[self.tilt_offset_end_index]
		cfit[5:] = self.results.x[self.tilt_offset_end_index+1:]
		OPD = np.tensordot(cfit, self.tmpbeamclass.zernike_polynomials, axes=([0],[0]))
		plt.figure()
		plt.imshow(OPD,extent=([-self.tmpbeamclass.L/2.,self.tmpbeamclass.L/2.,-self.tmpbeamclass.L/2.,self.tmpbeamclass.L/2.]),
			vmin=vmin,vmax=vmax)
		#plt.imshow(OPD)

		plt.xlim(-30,30)
		plt.ylim(-30,30)
		plt.xlabel('x (meters)')
		plt.ylabel('y (meters)')
		cbar = plt.colorbar()
		cbar.set_label('OPD (microns)',rotation=-90,labelpad=20)
		plt.savefig(savefilename,bbox_inches='tight')
		plt.close()
		self.surfaceplot_fname = savefilename




	def combine_result_pngs(self,
							outfile="combined_results.png",
							dpi=200):

		main_png = self.datamodelresid_fname
		table_png = self.resulttable_fname
		opd_png = self.surfaceplot_fname
		"""
		Combine three PNGs into one summary figure.

		Layout:
		- top row: main_png spans full width
		- bottom left: opd_png
		- bottom right: table_png
		"""

		img_main = mpimg.imread(main_png)
		img_table = mpimg.imread(table_png)
		img_opd = mpimg.imread(opd_png)

		fig = plt.figure(figsize=(12, 14), constrained_layout=True)
		gs = GridSpec(
			2, 2,
			figure=fig,
			height_ratios=[1.2, 1.0],
			width_ratios=[1.0, 1.1]
		)

		ax_main = fig.add_subplot(gs[0, :])
		ax_opd = fig.add_subplot(gs[1, 0])
		ax_table = fig.add_subplot(gs[1, 1])

		ax_main.imshow(img_main)
		ax_opd.imshow(img_opd)
		ax_table.imshow(img_table)

		ax_main.axis("off")
		ax_opd.axis("off")
		ax_table.axis("off")

		plt.savefig(outfile, dpi=dpi, bbox_inches="tight")
		plt.close(fig)

		print(f"Saved combined figure to {outfile}")

	def save_zernike_dat(self, filename):
		"""
		Save self.resultsdict in the same format as the LMT OOF
		zernike .dat file.

		Supports both the named low-order Zernike parameters and
		higher-order parameters stored as "OSA Index N".
		"""

		# --------------------------------------------------
		# Full ordered list used by the LMT .dat format
		# --------------------------------------------------

		zernike_rows = [
			(0,  "BIAS"),
			(1,  "TILT_H"),
			(2,  "TILT_V"),
			(3,  "FOCUS"),
			(4,  "AST_V"),
			(5,  "AST_O"),
			(6,  "COMA_H"),
			(7,  "COMA_V"),
			(8,  "TRE_O"),
			(9,  "TRE_V"),
			(10, "SPH"),
			(11, "2AST_V"),
			(12, "2AST_O"),
			(13, "TET_V"),
			(14, "TET_O"),
			(15, "2COMA_H"),
			(16, "2COMA_V"),
			(17, "2TRE_O"),
			(18, "2TRE_V"),
			(19, "PEN_O"),
			(20, "PEN_V"),
			(21, "2SPH"),
			(22, "3AST_V"),
			(23, "3AST_O"),
			(24, "2TET_V"),
			(25, "2TET_O"),
			(26, "HEX_V"),
			(27, "HEX_O"),
			(28, "3COMA_H"),
			(29, "3COMA_V"),
			(30, "3TRE_O"),
			(31, "3TRE_V"),
			(32, "2PEN_O"),
			(33, "2PEN_V"),
			(34, "HEPT_O"),
			(35, "HEPT_V"),
			(36, "3SPH"),
			(37, "4AST_V"),
			(38, "4AST_O"),
			(39, "3TET_V"),
			(40, "3TET_O"),
			(41, "2HEX_V"),
			(42, "2HEX_O"),
			(43, "OCT_V"),
			(44, "OCT_O"),
		]

		# --------------------------------------------------
		# Map .dat names -> names used in self.resultsdict
		#
		# These preserve your existing naming convention.
		# --------------------------------------------------

		result_mapping = {
			"AST_V":   "AST_V",
			"AST_O":   "AST_O",
			"COMA_H":  "COMA_H",
			"COMA_V":  "COMA_V",
			"TRE_O":   "TRE_O",
			"TRE_V":   "TRE_V",
			"SPH":     "SPH",
			"2AST_V":  "AST2_V",
			"2AST_O":  "AST2_O",
			"TET_V":   "QUAD_V",
			"TET_O":   "QUAD_O",
		}

		# --------------------------------------------------
		# Write file
		# --------------------------------------------------

		with open(filename, "w") as f:

			for index, dat_label in zernike_rows:

				value = 0.0
				flag = 0

				# ------------------------------------------
				# First check for one of your explicitly
				# named Zernike coefficients.
				# ------------------------------------------

				if dat_label in result_mapping:

					result_label = result_mapping[dat_label]

					if result_label in self.resultsdict:
						value = float(self.resultsdict[result_label])
						flag = 1

				# ------------------------------------------
				# Otherwise check whether this mode was
				# stored generically as "OSA Index N".
				# ------------------------------------------

				else:

					osa_label = f"OSA Index {index}"

					if osa_label in self.resultsdict:
						value = float(self.resultsdict[osa_label])
						flag = 1

				f.write(
					f"{index:d} {flag:d} {dat_label} {value:.6f}\n"
				)

	def save_subref_dat(self, filename, reference_m2z=0.0):
		"""
		Save the M2 position offsets in the same format as the LMT OOF
		subref.dat file.

		M2Z is -x[1], the absolute coordinate where modeled defocus is zero.
		Writes M2Z in microns; lateral offsets are not fitted.
		reference_m2z is accepted for compatibility and does not affect this export.
		"""

		# Convert meters -> microns
		m2z_microns = -float(self.results.x[1]) * 1e6

		with open(filename, "w") as f:
			f.write(f"0 1 M2Z {m2z_microns:.6f}\n")
			f.write(f"1 0 M2X {0.0:.6f}\n")
			f.write(f"2 0 M2Y {0.0:.6f}\n")



class fit_beam_with_M2_offsets_globTilt_leastsquares:

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
		for i in range(1):
			print(f'x[{vectposoffsetcounter}] = TILT_Y_map{i}')
			vectposoffsetcounter+=1
			print(f'x[{vectposoffsetcounter}] = TILT_X_map{i}')
			vectposoffsetcounter+=1
		self.tilt_offset_end_index = vectposoffsetcounter
		print(f'x[{vectposoffsetcounter}] = del M2.X')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = del M2.Y')
		vectposoffsetcounter+=1
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
		self.fit_vec_size = 2+(2*1)+beam_class.zernike_polynomials.shape[0]-4 #vectposoffsetcounter+1

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
		residual_maps = {}
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

			ctmp[1] = x[tilt_counter]
			#tilt_counter+=1
			ctmp[2] = x[tilt_counter+1]
			#tilt_counter+=1

			ctmp[3] = x[self.tilt_offset_end_index+2]
			ctmp[5] = x[self.tilt_offset_end_index+3]
			ctmp[6] = x[self.tilt_offset_end_index+4]
			ctmp[9:] = x[self.tilt_offset_end_index+5:]

			modelbeam = source_amp*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+M2z_offset,
							   del_x=x[self.tilt_offset_end_index],del_y=x[self.tilt_offset_end_index+1],del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.)


			residual = self.tmpbeamclass.trunc_maps[i] - modelbeam
			residual_maps[i] = residual

			chi_squared+= residual**2

		tmpreturn = np.concatenate([residual.ravel() for residual in residual_maps.values()])
		self.fitting_counter+=1
		self.temp_cost = np.sqrt(np.mean(chi_squared))
		
		return tmpreturn

	def run_fitter(self):
		self.x0[1] = 0.

		self.x0[4] = 0.0#0.2E-3
		self.x0[5] = 0.0#2.5E-3

		self.x0[6] = 0.0#-53.5*1E-6*np.sqrt(2)*(2.*np.pi/np.mean(self.tmpbeamclass.wavelengths))
		self.x0[7] = 0.0#-45*1E-6*np.sqrt(2)*(2.*np.pi/np.mean(self.tmpbeamclass.wavelengths))
		self.x0[8] = 0.0#20*1E-6*np.sqrt(2)*(2.*np.pi/np.mean(self.tmpbeamclass.wavelengths))
		self.x0[9] = 0.0#-43*1E-6*np.sqrt(2)*(2.*np.pi/np.mean(self.tmpbeamclass.wavelengths))
		self.x0[10] = 0.0#19*1E-6*np.sqrt(2)*(2.*np.pi/np.mean(self.tmpbeamclass.wavelengths))

		
		self.results = least_squares(
			self.chisquared,
			self.x0,
			x_scale='jac'
		)
		ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
		ctmp[5] = self.results.x[self.tilt_offset_end_index+3]
		ctmp[6] = self.results.x[self.tilt_offset_end_index+4]
		ctmp[9:] = self.results.x[self.tilt_offset_end_index+5:]
		self.strehl_ratio = np.amax(self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.results.x[1],
							   del_x=self.results.x[self.tilt_offset_end_index],del_y=self.results.x[self.tilt_offset_end_index+1],del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.))

	def plot_fit_results(self,vmax_frac_of_source_flux = 0.2,resids_stretch=5,vmin_val = -50,
					 title=None,savefigname=None,showplot=False):

		plt.figure(figsize=(15,8))
		# plt.figure()
		subplotcounter=1
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			corners_tmp = np.rad2deg(enmap.corners(self.tmpbeamclass.trunc_maps[i].shape,self.tmpbeamclass.trunc_maps[i].wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(self.tmpbeamclass.trunc_maps[i],extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			post_stamp_size = 1.5*60.#1.5/60.
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
			ctmp[1] = self.results.x[tilt_counter]
			#tilt_counter+=1
			ctmp[2] = self.results.x[tilt_counter+1]
			#tilt_counter+=1

			ctmp[3] = self.results.x[self.tilt_offset_end_index+2]
			ctmp[5] = self.results.x[self.tilt_offset_end_index+3]
			ctmp[6] = self.results.x[self.tilt_offset_end_index+4]
			ctmp[9:] = self.results.x[self.tilt_offset_end_index+5:]

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
																		 del_x=self.results.x[self.tilt_offset_end_index],del_y=self.results.x[self.tilt_offset_end_index+1])
			
			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(tmpmodelbeam,extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate((self.tmpbeamclass.trunc_maps)):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
			ctmp[1] = self.results.x[tilt_counter]
			#tilt_counter+=1
			ctmp[2] = self.results.x[tilt_counter+1]
			#tilt_counter+=1

			ctmp[3] = self.results.x[self.tilt_offset_end_index+2]
			ctmp[5] = self.results.x[self.tilt_offset_end_index+3]
			ctmp[6] = self.results.x[self.tilt_offset_end_index+4]
			ctmp[9:] = self.results.x[self.tilt_offset_end_index+5:]

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
																	del_x=self.results.x[self.tilt_offset_end_index],del_y=self.results.x[self.tilt_offset_end_index+1])
			
			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(100*(self.tmpbeamclass.trunc_maps[i]-tmpmodelbeam)/self.results.x[0],extent=imextent_tmp,origin='lower',vmin=-resids_stretch,vmax=resids_stretch)
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])

			subplotcounter+=1

		fig = plt.gcf()

		nrows = 3
		ncols = self.number_of_maps

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
	def surface_plot(self,savefilename,vmin=-400,vmax=400):
		cfit = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
		cfit[3] = self.results.x[self.tilt_offset_end_index]
		cfit[5:] = self.results.x[self.tilt_offset_end_index+1:]
		OPD = np.tensordot(cfit, self.tmpbeamclass.zernike_polynomials, axes=([0],[0]))
		plt.figure()
		plt.imshow(OPD,extent=([-self.tmpbeamclass.L/2.,self.tmpbeamclass.L/2.,-self.tmpbeamclass.L/2.,self.tmpbeamclass.L/2.]),
			vmin=vmin,vmax=vmax)
		#plt.imshow(OPD)

		plt.xlim(-30,30)
		plt.ylim(-30,30)
		plt.xlabel('x (meters)')
		plt.ylabel('y (meters)')
		cbar = plt.colorbar()
		cbar.set_label('OPD (microns)',rotation=-90,labelpad=20)
		plt.savefig(savefilename,bbox_inches='tight')
		plt.close()

	def save_results(self,savefilename):
		zernike_labels = ['AST_O','AST_V','TRE_V','TRE_O','QUAD_O','AST2_O','SPH','AST2_V','QUAD_V']
		results_dict = {}
		results_dict['source_amp'] = self.results.x[0]
		results_dict['M2.Z_offset'] = self.results.x[1]
		results_dict['strehl_ratio'] = self.strehl_ratio
		map_counter = 0
		tmpind = self.tilt_offset_start_index
		while tmpind<self.tilt_offset_end_index:
			results_dict[f'Tilt_Y_map{map_counter}'] = self.results.x[tmpind]
			tmpind+=1
			results_dict[f'Tilt_X_map{map_counter}'] = self.results.x[tmpind]
			tmpind+=1
			map_counter+=1

		results_dict['M2.X_offset'] = self.results.x[self.tilt_offset_end_index]
		results_dict['M2.Y_offset'] = self.results.x[self.tilt_offset_end_index+1]

		results_dict['AST_O'] = self.results.x[self.tilt_offset_end_index+2]#*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))
		results_dict['AST_V'] = self.results.x[self.tilt_offset_end_index+3]#*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))
		results_dict['TRE_V'] = self.results.x[self.tilt_offset_end_index+4]#*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))

		for i in np.arange(self.results.x[self.tilt_offset_end_index+5:].size):
			if (i+3)<len(zernike_labels):
				zernlabel = zernike_labels[i+3]
			else:
				zernlabel = f'OSA Index {i+3+4}'

			results_dict[zernlabel] = self.results.x[self.tilt_offset_end_index+5+i]#*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))

		for i in results_dict:
			print(i+' ', results_dict[i])

		for key, value in results_dict.items():
			if isinstance(value, np.ndarray):
				results_dict[key] = value.tolist()
		with open(savefilename, "w") as f:
			json.dump(results_dict, f, indent=4)


class fit_beam_with_M2_offsets_noCOMA_leastsquares:

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
			print(f'x[{vectposoffsetcounter}] = AZ_Off_map{i}')
			vectposoffsetcounter+=1
			print(f'x[{vectposoffsetcounter}] = EL_Off_map{i}')
			vectposoffsetcounter+=1
		self.tilt_offset_end_index = vectposoffsetcounter
		print(f'x[{vectposoffsetcounter}] = del M2.X')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = del M2.Y')
		vectposoffsetcounter+=1
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
		self.fit_vec_size = 2+(2*mapcounter)+beam_class.zernike_polynomials.shape[0]-4

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
		residual_maps = {}
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

			ctmp[1] = x[tilt_counter]
			#tilt_counter+=1
			ctmp[2] = x[tilt_counter+1]
			#tilt_counter+=1

			ctmp[3] = x[self.tilt_offset_end_index+2]
			ctmp[5] = x[self.tilt_offset_end_index+3]
			ctmp[6] = x[self.tilt_offset_end_index+4]
			ctmp[9:] = x[self.tilt_offset_end_index+5:]

			modelbeam = source_amp*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+M2z_offset,
							   del_x=x[self.tilt_offset_end_index],del_y=x[self.tilt_offset_end_index+1],del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.)


			residual = self.tmpbeamclass.trunc_maps[i] - modelbeam
			residual_maps[i] = residual

			chi_squared+= residual**2

		tmpreturn = np.concatenate([residual.ravel() for residual in residual_maps.values()])
		self.fitting_counter+=1
		self.temp_cost = np.sqrt(np.mean(chi_squared))
		
		return tmpreturn

	def make_bounds(self,tilt_plusminus,zern_plusminus):
		tmpbounds = []

		lowerbounds = []
		upperbounds = []

		for i in range(len(self.x0)):
			if i==0:
				# lowerbounds.append(0.7*self.x0[0])
				# upperbounds.append(1.3*self.x0[0])
				lowerbounds.append(0)
				upperbounds.append(np.inf)
			elif i==1:
				# lowerbounds.append(-5E-3)
				# upperbounds.append(5E-3)
				lowerbounds.append(-np.inf)
				upperbounds.append(np.inf)
			elif i in range(2,self.tilt_offset_end_index+1):
				# lowerbounds.append(-tilt_plusminus)
				# upperbounds.append(tilt_plusminus)
				lowerbounds.append(-np.inf)
				upperbounds.append(np.inf)
			elif i==self.tilt_offset_end_index+8+2:
				# lowerbounds.append(-tilt_plusminus)
				# upperbounds.append(tilt_plusminus)
				lowerbounds.append(-150)
				upperbounds.append(150)
			else:
				# lowerbounds.append(-zern_plusminus)
				# upperbounds.append(zern_plusminus)
				lowerbounds.append(-np.inf)
				upperbounds.append(np.inf)
		lowerbounds = tuple(lowerbounds)
		upperbounds = tuple(upperbounds)
		return (lowerbounds,upperbounds)

	def run_fitter(self):

		
		self.results = least_squares(
			self.chisquared,
			self.x0,
			x_scale='jac'
		)
		ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
		ctmp[5] = self.results.x[self.tilt_offset_end_index+3]
		ctmp[6] = self.results.x[self.tilt_offset_end_index+4]
		ctmp[9:] = self.results.x[self.tilt_offset_end_index+5:]
		self.strehl_ratio = np.amax(self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.results.x[1],
							   del_x=self.results.x[self.tilt_offset_end_index],del_y=self.results.x[self.tilt_offset_end_index+1],del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.))

	def plot_fit_results(self,vmax_frac_of_source_flux = 0.2,resids_stretch=5,vmin_val = -50,
					 title=None,savefigname=None,showplot=False):

		plt.figure(figsize=(15,8))
		# plt.figure()
		subplotcounter=1
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			corners_tmp = np.rad2deg(enmap.corners(self.tmpbeamclass.trunc_maps[i].shape,self.tmpbeamclass.trunc_maps[i].wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(self.tmpbeamclass.trunc_maps[i],extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			post_stamp_size = 1.5*60.#1.5/60.
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
			ctmp[1] = self.results.x[tilt_counter]
			#tilt_counter+=1
			ctmp[2] = self.results.x[tilt_counter+1]
			#tilt_counter+=1

			ctmp[3] = self.results.x[self.tilt_offset_end_index+2]
			ctmp[5] = self.results.x[self.tilt_offset_end_index+3]
			ctmp[6] = self.results.x[self.tilt_offset_end_index+4]
			ctmp[9:] = self.results.x[self.tilt_offset_end_index+5:]

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
																		 del_x=self.results.x[self.tilt_offset_end_index],del_y=self.results.x[self.tilt_offset_end_index+1])
			
			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(tmpmodelbeam,extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate((self.tmpbeamclass.trunc_maps)):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
			ctmp[1] = self.results.x[tilt_counter]
			#tilt_counter+=1
			ctmp[2] = self.results.x[tilt_counter+1]
			#tilt_counter+=1

			ctmp[3] = self.results.x[self.tilt_offset_end_index+2]
			ctmp[5] = self.results.x[self.tilt_offset_end_index+3]
			ctmp[6] = self.results.x[self.tilt_offset_end_index+4]
			ctmp[9:] = self.results.x[self.tilt_offset_end_index+5:]

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
																	del_x=self.results.x[self.tilt_offset_end_index],del_y=self.results.x[self.tilt_offset_end_index+1])
			
			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(100*(self.tmpbeamclass.trunc_maps[i]-tmpmodelbeam)/self.results.x[0],extent=imextent_tmp,origin='lower',vmin=-resids_stretch,vmax=resids_stretch)
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])

			subplotcounter+=1

		fig = plt.gcf()

		nrows = 3
		ncols = self.number_of_maps

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
	def surface_plot(self,savefilename,vmin=-400,vmax=400):

		cfit = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

		#tilt_counter+=1

		cfit[3] = self.results.x[self.tilt_offset_end_index+2]
		cfit[5] = self.results.x[self.tilt_offset_end_index+3]
		cfit[6] = self.results.x[self.tilt_offset_end_index+4]
		cfit[9:] = self.results.x[self.tilt_offset_end_index+5:]

		OPD = np.tensordot(cfit, self.tmpbeamclass.zernike_polynomials, axes=([0],[0]))
		plt.figure()
		plt.imshow(OPD,extent=([-self.tmpbeamclass.L/2.,self.tmpbeamclass.L/2.,-self.tmpbeamclass.L/2.,self.tmpbeamclass.L/2.]),
			vmin=vmin,vmax=vmax)
		#plt.imshow(OPD)

		plt.xlim(-30,30)
		plt.ylim(-30,30)
		plt.xlabel('x (meters)')
		plt.ylabel('y (meters)')
		cbar = plt.colorbar()
		cbar.set_label('OPD (microns)',rotation=-90,labelpad=20)
		plt.savefig(savefilename,bbox_inches='tight')
		plt.close()

	def save_results(self,savefilename):
		zernike_labels = ['AST_O','AST_V','TRE_V','TRE_O','QUAD_O','AST2_O','SPH','AST2_V','QUAD_V']
		results_dict = {}
		results_dict['source_amp'] = self.results.x[0]
		results_dict['M2.Z_offset'] = self.results.x[1]
		results_dict['strehl_ratio'] = self.strehl_ratio
		map_counter = 0
		tmpind = self.tilt_offset_start_index
		while tmpind<self.tilt_offset_end_index:
			results_dict[f'Tilt_Y_map{map_counter}'] = self.results.x[tmpind]
			tmpind+=1
			results_dict[f'Tilt_X_map{map_counter}'] = self.results.x[tmpind]
			tmpind+=1
			map_counter+=1

		results_dict['M2.X_offset'] = self.results.x[self.tilt_offset_end_index]
		results_dict['M2.Y_offset'] = self.results.x[self.tilt_offset_end_index+1]

		results_dict['AST_O'] = self.results.x[self.tilt_offset_end_index+2]#*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))
		results_dict['AST_V'] = self.results.x[self.tilt_offset_end_index+3]#*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))
		results_dict['TRE_V'] = self.results.x[self.tilt_offset_end_index+4]#*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))

		for i in np.arange(self.results.x[self.tilt_offset_end_index+5:].size):
			if (i+3)<len(zernike_labels):
				zernlabel = zernike_labels[i+3]
			else:
				zernlabel = f'OSA Index {i+3+4}'

			results_dict[zernlabel] = self.results.x[self.tilt_offset_end_index+5+i]#*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))

		for i in results_dict:
			print(i+' ', results_dict[i])

		for key, value in results_dict.items():
			if isinstance(value, np.ndarray):
				results_dict[key] = value.tolist()
		with open(savefilename, "w") as f:
			json.dump(results_dict, f, indent=4)


class fit_beam_with_M2_offsets_wCOMA_leastsquares:

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
			print(f'x[{vectposoffsetcounter}] = AZ_Off_map{i}')
			vectposoffsetcounter+=1
			print(f'x[{vectposoffsetcounter}] = EL_Off_map{i}')
			vectposoffsetcounter+=1
		self.tilt_offset_end_index = vectposoffsetcounter
		print(f'x[{vectposoffsetcounter}] = del M2.X')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = del M2.Y')
		vectposoffsetcounter+=1
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
		self.fit_vec_size = 2+(2*mapcounter)+beam_class.zernike_polynomials.shape[0]-2

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
		residual_maps = {}
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

			ctmp[1] = x[tilt_counter]
			#tilt_counter+=1
			ctmp[2] = x[tilt_counter+1]
			#tilt_counter+=1

			ctmp[3] = x[self.tilt_offset_end_index+2]
			ctmp[5] = x[self.tilt_offset_end_index+3]
			ctmp[6:] = x[self.tilt_offset_end_index+4:]
			#ctmp[9:] = x[self.tilt_offset_end_index+5:]

			modelbeam = source_amp*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+M2z_offset,
							   del_x=x[self.tilt_offset_end_index],del_y=x[self.tilt_offset_end_index+1],del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.)


			residual = self.tmpbeamclass.trunc_maps[i] - modelbeam
			residual_maps[i] = residual

			chi_squared+= residual**2

		tmpreturn = np.concatenate([residual.ravel() for residual in residual_maps.values()])
		self.fitting_counter+=1
		self.temp_cost = np.sqrt(np.mean(chi_squared))
		
		return tmpreturn

	def make_bounds(self,tilt_plusminus,zern_plusminus):
		tmpbounds = []

		lowerbounds = []
		upperbounds = []

		for i in range(len(self.x0)):
			if i==0:
				lowerbounds.append(0.7*self.x0[0])
				upperbounds.append(1.3*self.x0[0])
				# lowerbounds.append(0)
				# upperbounds.append(np.inf)
			elif i==1:
				# lowerbounds.append(-5E-3)
				# upperbounds.append(5E-3)
				lowerbounds.append(-np.inf)
				upperbounds.append(np.inf)
			elif i in range(2,self.tilt_offset_end_index+1):
				# lowerbounds.append(-tilt_plusminus)
				# upperbounds.append(tilt_plusminus)
				lowerbounds.append(-np.inf)
				upperbounds.append(np.inf)

			elif i in range(self.tilt_offset_end_index+1,self.tilt_offset_end_index+2):
				# lowerbounds.append(-tilt_plusminus)
				# upperbounds.append(tilt_plusminus)
				lowerbounds.append(-5E-3)
				upperbounds.append(5E-3)
			elif i in range(self.tilt_offset_end_index+2,len(self.x0)+1):
				# lowerbounds.append(-tilt_plusminus)
				# upperbounds.append(tilt_plusminus)
				lowerbounds.append(-300)
				upperbounds.append(300)
			# elif i==self.tilt_offset_end_index+
			# 	# lowerbounds.append(-zern_plusminus)
			# 	# upperbounds.append(zern_plusminus)
			# 	lowerbounds.append(-np.inf)
			# 	upperbounds.append(np.inf)
		lowerbounds = tuple(lowerbounds)
		upperbounds = tuple(upperbounds)
		return (lowerbounds,upperbounds)

	def run_fitter(self):

		tmpbounds = self.make_bounds(None,None)

		
		self.results = least_squares(
			self.chisquared,
			self.x0,
			bounds=tmpbounds,
			x_scale='jac'
		)
		ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
		ctmp[5] = self.results.x[self.tilt_offset_end_index+3]
		ctmp[6:] = self.results.x[self.tilt_offset_end_index+4:]
		#ctmp[9:] = self.results.x[self.tilt_offset_end_index+5:]
		self.strehl_ratio = np.amax(self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.results.x[1],
							   del_x=self.results.x[self.tilt_offset_end_index],del_y=self.results.x[self.tilt_offset_end_index+1],del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.))

	def plot_fit_results(self,vmax_frac_of_source_flux = 0.2,resids_stretch=5,vmin_val = -50,
					 title=None,savefigname=None,showplot=False):

		plt.figure(figsize=(15,8))
		# plt.figure()
		subplotcounter=1
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			corners_tmp = np.rad2deg(enmap.corners(self.tmpbeamclass.trunc_maps[i].shape,self.tmpbeamclass.trunc_maps[i].wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(self.tmpbeamclass.trunc_maps[i],extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			post_stamp_size = 1.5*60.#1.5/60.
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
			ctmp[1] = self.results.x[tilt_counter]
			#tilt_counter+=1
			ctmp[2] = self.results.x[tilt_counter+1]
			#tilt_counter+=1

			ctmp[3] = self.results.x[self.tilt_offset_end_index+2]
			ctmp[5] = self.results.x[self.tilt_offset_end_index+3]
			ctmp[6:] = self.results.x[self.tilt_offset_end_index+4:]
			#ctmp[9:] = self.results.x[self.tilt_offset_end_index+5:]

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
																		 del_x=self.results.x[self.tilt_offset_end_index],del_y=self.results.x[self.tilt_offset_end_index+1])
			
			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(tmpmodelbeam,extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate((self.tmpbeamclass.trunc_maps)):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
			ctmp[1] = self.results.x[tilt_counter]
			#tilt_counter+=1
			ctmp[2] = self.results.x[tilt_counter+1]
			#tilt_counter+=1

			ctmp[3] = self.results.x[self.tilt_offset_end_index+2]
			ctmp[5] = self.results.x[self.tilt_offset_end_index+3]
			ctmp[6:] = self.results.x[self.tilt_offset_end_index+4:]
			#ctmp[9:] = self.results.x[self.tilt_offset_end_index+5:]

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
																	del_x=self.results.x[self.tilt_offset_end_index],del_y=self.results.x[self.tilt_offset_end_index+1])
			
			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(100*(self.tmpbeamclass.trunc_maps[i]-tmpmodelbeam)/self.results.x[0],extent=imextent_tmp,origin='lower',vmin=-resids_stretch,vmax=resids_stretch)
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])

			subplotcounter+=1

		fig = plt.gcf()

		nrows = 3
		ncols = self.number_of_maps

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
	def surface_plot(self,savefilename,vmin=-400,vmax=400):

		cfit = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

		#tilt_counter+=1

		cfit[3] = self.results.x[self.tilt_offset_end_index+2]
		cfit[5] = self.results.x[self.tilt_offset_end_index+3]
		cfit[6:] = self.results.x[self.tilt_offset_end_index+4:]
		#cfit[9:] = self.results.x[self.tilt_offset_end_index+5:]

		OPD = np.tensordot(cfit, self.tmpbeamclass.zernike_polynomials, axes=([0],[0]))
		plt.figure()
		plt.imshow(OPD,extent=([-self.tmpbeamclass.L/2.,self.tmpbeamclass.L/2.,-self.tmpbeamclass.L/2.,self.tmpbeamclass.L/2.]),
			vmin=vmin,vmax=vmax)
		#plt.imshow(OPD)

		plt.xlim(-30,30)
		plt.ylim(-30,30)
		plt.xlabel('x (meters)')
		plt.ylabel('y (meters)')
		cbar = plt.colorbar()
		cbar.set_label('OPD (microns)',rotation=-90,labelpad=20)
		plt.savefig(savefilename,bbox_inches='tight')
		plt.close()

	def save_results(self,savefilename):
		zernike_labels = ['AST_O','AST_V','TRE_V','COMA_V','COMA_H','TRE_O','QUAD_O','AST2_O','SPH','AST2_V','QUAD_V']
		results_dict = {}
		results_dict['source_amp'] = self.results.x[0]
		results_dict['M2.Z_offset'] = self.results.x[1]
		results_dict['strehl_ratio'] = self.strehl_ratio
		map_counter = 0
		tmpind = self.tilt_offset_start_index
		while tmpind<self.tilt_offset_end_index:
			results_dict[f'Tilt_Y_map{map_counter}'] = self.results.x[tmpind]
			tmpind+=1
			results_dict[f'Tilt_X_map{map_counter}'] = self.results.x[tmpind]
			tmpind+=1
			map_counter+=1

		results_dict['M2.X_offset'] = self.results.x[self.tilt_offset_end_index]
		results_dict['M2.Y_offset'] = self.results.x[self.tilt_offset_end_index+1]

		results_dict['AST_O'] = self.results.x[self.tilt_offset_end_index+2]#*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))
		results_dict['AST_V'] = self.results.x[self.tilt_offset_end_index+3]#*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))
		results_dict['TRE_V'] = self.results.x[self.tilt_offset_end_index+4]#*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))

		for i in np.arange(self.results.x[self.tilt_offset_end_index+5:].size):
			if (i+3)<len(zernike_labels):
				zernlabel = zernike_labels[i+3]
			else:
				zernlabel = f'OSA Index {i+3+4}'

			results_dict[zernlabel] = self.results.x[self.tilt_offset_end_index+5+i]#*1E6*(np.mean(self.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))

		for i in results_dict:
			print(i+' ', results_dict[i])

		for key, value in results_dict.items():
			if isinstance(value, np.ndarray):
				results_dict[key] = value.tolist()
		with open(savefilename, "w") as f:
			json.dump(results_dict, f, indent=4)


# class fit_beam_with_glob_tilt_M2offsets_leastsquares:

# 	def __init__(self,beam_class):

# 		print('Initializing the fit beam class ')


# 		mapcounter= 0 
# 		for i in beam_class.trunc_maps:
# 			mapcounter+=1
# 		print(f'There are {mapcounter} maps')
# 		print('The vector of model parameters will have the form:')
# 		print('x[0] = source_amplitude')
# 		print('x[1] = M2.Z Offset')
# 		vectposoffsetcounter = 0+2
# 		self.tilt_offset_start_index = vectposoffsetcounter
# 		for i in range(mapcounter):
# 			print(f'x[{vectposoffsetcounter}] = AZ_Off_map{i}')
# 			vectposoffsetcounter+=1
# 			print(f'x[{vectposoffsetcounter}] = EL_Off_map{i}')
# 			vectposoffsetcounter+=1
# 		self.tilt_offset_end_index = 8
# 		print(f'x[{vectposoffsetcounter}] = AST_O')
# 		vectposoffsetcounter+=1
# 		print(f'x[{vectposoffsetcounter}] = AST_V')
# 		vectposoffsetcounter+=1
# 		print(f'x[{vectposoffsetcounter}] = TRE_V')
# 		vectposoffsetcounter+=1
# 		print('...')


# 		self.number_of_maps = mapcounter
# 		self.fit_vec_size = 2+(2)+4+beam_class.zernike_polynomials.shape[0]-4-2 #vectposoffsetcounter+1

# 		self.tmpbeamclass = beam_class

# 		x0 = np.zeros(self.fit_vec_size)

# 		map_maxes = np.zeros(self.number_of_maps)

# 		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
# 			tmpmax = np.amax(self.tmpbeamclass.trunc_maps[i])
# 			map_maxes[count] = tmpmax
# 		ampguess = max(map_maxes)

# 		x0[0] = ampguess
# 		x0[1] = beam_class.m2z_vals[f'map{int(self.number_of_maps/2.)}']

# 		self.x0 = x0

# 		self.fitting_counter = 0
# 		self.temp_cost = -999

# 	def chisquared(self,x):

# 		if self.fitting_counter%500==0:
# 			print('On fitting iteration = ',self.fitting_counter, ' with Cost = ',self.temp_cost)

# 		chi_squared = np.zeros(self.tmpbeamclass.trunc_maps['map0'].shape)

# 		source_amp = x[0]
# 		M2z_offset = x[1]

# 		a_m2x = x[2]
# 		b_m2x = x[3]

# 		a_m2y = x[4]
# 		b_m2y = x[5]

# 		tilt_counter = self.tilt_offset_start_index
# 		residual_maps = {}
# 		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
# 			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

# 			ctmp[1] = x[6]
# 			tilt_counter+=1
# 			ctmp[2] = x[7]
# 			tilt_counter+=1

# 			ctmp[3] = x[self.tilt_offset_end_index]
# 			ctmp[5] = x[self.tilt_offset_end_index+1]
# 			ctmp[6] = x[self.tilt_offset_end_index+2]
# 			ctmp[9:] = x[self.tilt_offset_end_index+3:]

# 			tmpdelx = a_m2x+(b_m2x*self.tmpbeamclass.m2z_vals[i])
# 			tmpdely = a_m2y+(b_m2y*self.tmpbeamclass.m2z_vals[i])

# 			modelbeam = source_amp*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+M2z_offset,
# 							   del_x=tmpdelx,del_y=tmpdely,del_alph_x=0.,del_alph_y=0.,
# 							   f=17.5,F=525.,D=50.)

# 			residual = self.tmpbeamclass.trunc_maps[i] - modelbeam
# 			residual_maps[i] = residual

# 			chi_squared+= residual**2

# 		tmpreturn = np.concatenate([residual.ravel() for residual in residual_maps.values()])
# 		self.fitting_counter+=1
# 		self.temp_cost = np.sqrt(np.mean(chi_squared))
		
# 		return tmpreturn

# 	def make_bounds(self,tilt_plusminus,zern_plusminus):
# 		tmpbounds = []

# 		lowerbounds = []
# 		upperbounds = []

# 		for i in range(len(self.x0)):
# 			if i==0:
# 				# lowerbounds.append(0.7*self.x0[0])
# 				# upperbounds.append(1.3*self.x0[0])
# 				lowerbounds.append(0)
# 				upperbounds.append(np.inf)
# 			elif i==1:
# 				# lowerbounds.append(-5E-3)
# 				# upperbounds.append(5E-3)
# 				lowerbounds.append(-np.inf)
# 				upperbounds.append(np.inf)
# 			elif i in [2,3,4]:
# 				# lowerbounds.append(-tilt_plusminus)
# 				# upperbounds.append(tilt_plusminus)
# 				lowerbounds.append(-np.inf)
# 				upperbounds.append(np.inf)
# 			elif i==self.tilt_offset_end_index+8:
# 				# lowerbounds.append(-tilt_plusminus)
# 				# upperbounds.append(tilt_plusminus)
# 				lowerbounds.append(-150)
# 				upperbounds.append(150)
# 			else:
# 				# lowerbounds.append(-zern_plusminus)
# 				# upperbounds.append(zern_plusminus)
# 				lowerbounds.append(-np.inf)
# 				upperbounds.append(np.inf)
# 		lowerbounds = tuple(lowerbounds)
# 		upperbounds = tuple(upperbounds)
# 		return (lowerbounds,upperbounds)



# 	def run_fitter(self,tilt_plusminus=None,zern_plusminus=None,
# 					newx0 = None):
# 		# results = minimize(self.chisquared,x0=self.x0)
# 		# self.results = results

# 		if newx0 is not None:
# 			self.x0 = newx0

# 		tmpbound = self.make_bounds(1000,350)

# 		self.temp_cost = 1E6
		
# 		previous_cost = [None]
# 		small_change_counter = [0]

		

# 		self.results = least_squares(
# 			self.chisquared,
# 			self.x0,
# 			x_scale='jac'
# 		)
# 		ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
# 		ctmp[3] = self.results.x[self.tilt_offset_end_index]
# 		ctmp[5] = self.results.x[self.tilt_offset_end_index+1]
# 		ctmp[6] = self.results.x[self.tilt_offset_end_index+2]
# 		ctmp[9:] = self.results.x[self.tilt_offset_end_index+3:]
# 		self.strehl_ratio = np.amax(self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.results.x[1],
# 							   del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
# 							   f=17.5,F=525.,D=50.))


# 	def plot_fit_results(self,vmax_frac_of_source_flux = 0.2,resids_stretch=5,vmin_val = -50,
# 					 title=None,savefigname=None,showplot=False):

# 		#plt.figure(figsize=(15,8))
# 		fig = plt.figure(figsize=(3.0*self.number_of_maps + 1.2, 8))
# 		# plt.figure()
# 		subplotcounter=1
# 		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
# 			plt.subplot(3,self.number_of_maps,subplotcounter)
# 			corners_tmp = np.rad2deg(enmap.corners(self.tmpbeamclass.trunc_maps[i].shape,self.tmpbeamclass.trunc_maps[i].wcs))
# 			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
# 			plt.imshow(self.tmpbeamclass.trunc_maps[i],extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
# 			plt.xticks([])
# 			post_stamp_size = 1.5*60.#1.5/60.
# 			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
# 			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
# 			if (subplotcounter)%self.number_of_maps!=1:
# 				plt.yticks([])
# 			subplotcounter+=1
# 		tilt_counter = self.tilt_offset_start_index
# 		fit_a_m2x = self.results.x[2]
# 		fit_b_m2x = self.results.x[3]

# 		fit_a_m2y = self.results.x[4]
# 		fit_b_m2y = self.results.x[5]

# 		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
# 			plt.subplot(3,self.number_of_maps,subplotcounter)
# 			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

# 			ctmp[1] = self.results.x[6]
# 			tilt_counter+=1
# 			ctmp[2] = self.results.x[7]
# 			tilt_counter+=1


# 			ctmp[3] = self.results.x[self.tilt_offset_end_index]
# 			ctmp[5] = self.results.x[self.tilt_offset_end_index+1]
# 			ctmp[6] = self.results.x[self.tilt_offset_end_index+2]
# 			ctmp[9:] = self.results.x[self.tilt_offset_end_index+3:]

# 			tmpdelx = fit_a_m2x+(fit_b_m2x*self.tmpbeamclass.m2z_vals[i])
# 			tmpdely = fit_a_m2y+(fit_b_m2y*self.tmpbeamclass.m2z_vals[i])

# 			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,
# 				secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
# 				del_x=tmpdelx,del_y=tmpdely)
			

# 			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
# 			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
# 			plt.imshow(tmpmodelbeam,extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
# 			plt.xticks([])
# 			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
# 			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
# 			if (subplotcounter)%self.number_of_maps!=1:
# 				plt.yticks([])
# 			if subplotcounter == self.number_of_maps + 1:
# 				plt.ylabel('El Offset (arcsec)')
# 			subplotcounter+=1
# 		tilt_counter = self.tilt_offset_start_index
# 		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
# 			plt.subplot(3,self.number_of_maps,subplotcounter)
# 			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

# 			ctmp[1] = self.results.x[6]
# 			tilt_counter+=1
# 			ctmp[2] = self.results.x[7]
# 			tilt_counter+=1
			

# 			ctmp[3] = self.results.x[self.tilt_offset_end_index]
# 			ctmp[5] = self.results.x[self.tilt_offset_end_index+1]
# 			ctmp[6] = self.results.x[self.tilt_offset_end_index+2]
# 			ctmp[9:] = self.results.x[self.tilt_offset_end_index+3:]

# 			tmpdelx = fit_a_m2x+(fit_b_m2x*self.tmpbeamclass.m2z_vals[i])
# 			tmpdely = fit_a_m2y+(fit_b_m2y*self.tmpbeamclass.m2z_vals[i])

# 			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,
# 				secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
# 				del_x=tmpdelx,del_y=tmpdely)
			

# 			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
# 			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
# 			plt.imshow(100*(self.tmpbeamclass.trunc_maps[i]-tmpmodelbeam)/self.results.x[0],extent=imextent_tmp,origin='lower',vmin=-resids_stretch,vmax=resids_stretch)
# 			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
# 			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
# 			if subplotcounter == 2*self.number_of_maps + 2:
# 				plt.xlabel('Az Offset (arcsec)')
# 			if (subplotcounter)%self.number_of_maps!=1:
# 				plt.yticks([])

# 			subplotcounter+=1

# 		#fig = plt.gcf()

# 		nrows = 3
# 		ncols = self.number_of_maps

# 		# Optional: leave some room on the right for colorbars
# 		#fig.subplots_adjust(right=0.88)

# 		fig.subplots_adjust(
# 			left=0.10,
# 			right=0.88,
# 			bottom=0.06,
# 			top=0.92,
# 			wspace=0.03,
# 			hspace=0.20
# 		)

# 		cbar_labels = ['mJy/beam','mJy/beam','% Source Flux']
# 		row_labels = ["data", "model", "residuals"]

# 		for row in range(nrows):



# 			# index of last subplot in this row (1-based for plt.subplot)
# 			last_idx = (row + 1) * ncols
# 			ax = plt.subplot(nrows, ncols, last_idx)

# 			# Get position of this last axes
# 			pos = ax.get_position()

# 			# Make a new axes for the colorbar, just to the right of it
# 			cax = fig.add_axes([
# 				pos.x1 + 0.005,   # a bit to the right of the last axes
# 				pos.y0,           # same bottom
# 				0.01,             # narrow width
# 				pos.height        # same height
# 			])

# 			# Get the image plotted in this axes (the first imshow)
# 			im = ax.images[0]

# 			# Make a colorbar for that row
# 			cb = fig.colorbar(im, cax=cax)
# 			cb.set_label(cbar_labels[row],rotation=-90,labelpad=20)

# 			first_idx = ((row) * ncols)+1
# 			ax = plt.subplot(nrows, ncols, first_idx)
# 			pos = ax.get_position()


# 			# y-coordinate of the center of this row
# 			y = pos.y0 + pos.height/2

# 			# Put text slightly left of this row
# 			fig.text(
# 				pos.x0 - 0.06,     # shift a little left of the first subplot
# 				y,
# 				row_labels[row],
# 				va='center', ha='right', fontsize=14
# 			)
# 		plt.suptitle(title)
# 		if savefigname is not None:
# 			plt.savefig(savefigname,bbox_inches='tight')

# 		if showplot:
# 			plt.show()
# 		else:
# 			plt.close()
# 		self.datamodelresid_fname = savefigname


# 	def save_results(self, savefilename):

# 		# Friendly names for the Zernike modes you commonly use.
# 		# These correspond to OSA/ANSI indices 4 through 14.
# 		zernike_labels = {
# 			4:  'AST_O',
# 			5:  'AST_V',
# 			6:  'TRE_V',
# 			7:  'COMA_V',
# 			8:  'COMA_H',
# 			9:  'TRE_O',
# 			10: 'QUAD_O',
# 			11: 'AST2_O',
# 			12: 'SPH',
# 			13: 'AST2_V',
# 			14: 'QUAD_V',
# 		}

# 		results_dict = {}

# 		# --------------------------------------------------
# 		# General fit parameters
# 		# --------------------------------------------------

# 		results_dict['source_amp'] = self.results.x[0]
# 		results_dict['M2.Z_offset'] = self.results.x[1]
# 		results_dict['strehl_ratio'] = self.strehl_ratio

# 		# --------------------------------------------------
# 		# Pointing offsets
# 		# --------------------------------------------------

# 		map_counter = 0
# 		tmpind = self.tilt_offset_start_index

# 		while tmpind < self.tilt_offset_end_index:

# 			results_dict[f'Az_Off_map{map_counter}'] = self.results.x[tmpind]
# 			tmpind += 1

# 			results_dict[f'El_Off_map{map_counter}'] = self.results.x[tmpind]
# 			tmpind += 1

# 			map_counter += 1

# 		# --------------------------------------------------
# 		# Zernike coefficients
# 		# --------------------------------------------------

# 		zernike_results = self.results.x[self.tilt_offset_end_index:]

# 		for i, val in enumerate(zernike_results):

# 			osa_index = i + 4

# 			# Use your descriptive name when available.
# 			# Otherwise fall back to the OSA index.
# 			zernlabel = zernike_labels.get(
# 				osa_index,
# 				f'OSA Index {osa_index}'
# 			)

# 			results_dict[zernlabel] = val

# 		# --------------------------------------------------
# 		# Print results
# 		# --------------------------------------------------

# 		for key, value in results_dict.items():
# 			print(key, value)

# 		# --------------------------------------------------
# 		# Convert numpy arrays/scalars for JSON
# 		# --------------------------------------------------

# 		for key, value in results_dict.items():

# 			if isinstance(value, np.ndarray):
# 				results_dict[key] = value.tolist()

# 			elif isinstance(value, np.generic):
# 				results_dict[key] = value.item()

# 		# --------------------------------------------------
# 		# Save JSON
# 		# --------------------------------------------------

# 		with open(savefilename, "w") as f:
# 			json.dump(results_dict, f, indent=4)

# 		self.resultsdict = results_dict

# 		# --------------------------------------------------
# 		# Save results as a PNG table
# 		# --------------------------------------------------

# 		table_rows = []

# 		for label, value in results_dict.items():

# 			# Determine units from the parameter itself rather
# 			# than from its position in the dictionary.
# 			if label == 'source_amp':
# 				unit = 'mJy/beam'

# 			elif label == 'M2.Z_offset':
# 				unit = 'mm'

# 			elif label == 'strehl_ratio':
# 				unit = ''

# 			elif label.startswith('Az_Off_map'):
# 				unit = 'microns'

# 			elif label.startswith('El_Off_map'):
# 				unit = 'microns'

# 			else:
# 				# Remaining parameters are Zernike coefficients
# 				unit = 'microns'

# 			# --------------------------------------------------
# 			# Format values
# 			# --------------------------------------------------

# 			if (
# 				isinstance(value, (int, float, np.integer, np.floating))
# 				and label == 'M2.Z_offset'
# 			):
# 				# Convert meters -> mm for display
# 				display_value = value * 1E3
# 				value_str = f"{display_value:.6f}"

# 			elif isinstance(
# 				value,
# 				(int, float, np.integer, np.floating)
# 			):
# 				value_str = f"{value:.2f}"

# 			else:
# 				value_str = str(value)

# 			table_rows.append([
# 				label,
# 				value_str,
# 				unit
# 			])

# 		# --------------------------------------------------
# 		# Make table figure
# 		# --------------------------------------------------

# 		fig_height = max(
# 			2.0,
# 			0.35 * (len(table_rows) + 1)
# 		)

# 		fig, ax = plt.subplots(
# 			figsize=(6, fig_height)
# 		)

# 		ax.axis("off")

# 		table = ax.table(
# 			cellText=table_rows,
# 			colLabels=["LABEL", "VALUE", "UNITS"],
# 			cellLoc="center",
# 			colLoc="center",
# 			loc="center",
# 		)

# 		table.auto_set_font_size(False)
# 		table.set_fontsize(11)
# 		table.scale(1.0, 1.4)

# 		# Style cells
# 		for (row, col), cell in table.get_celld().items():

# 			cell.set_edgecolor("black")
# 			cell.set_linewidth(1.0)

# 			if row == 0:
# 				cell.set_facecolor("0.9")
# 			else:
# 				cell.set_facecolor("0.96")

# 		# Same filename as JSON, but .png
# 		pngfilename = (
# 			os.path.splitext(savefilename)[0]
# 			+ ".png"
# 		)

# 		plt.savefig(
# 			pngfilename,
# 			dpi=200,
# 			bbox_inches="tight"
# 		)

# 		plt.close(fig)

# 		self.resulttable_fname = pngfilename

# 	def surface_plot(self,savefilename,vmin=-400,vmax=400):
# 		cfit = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
# 		cfit[3] = self.results.x[self.tilt_offset_end_index]
# 		cfit[5:] = self.results.x[self.tilt_offset_end_index+1:]
# 		OPD = np.tensordot(cfit, self.tmpbeamclass.zernike_polynomials, axes=([0],[0]))
# 		plt.figure()
# 		plt.imshow(OPD,extent=([-self.tmpbeamclass.L/2.,self.tmpbeamclass.L/2.,-self.tmpbeamclass.L/2.,self.tmpbeamclass.L/2.]),
# 			vmin=vmin,vmax=vmax)
# 		#plt.imshow(OPD)

# 		plt.xlim(-30,30)
# 		plt.ylim(-30,30)
# 		plt.xlabel('x (meters)')
# 		plt.ylabel('y (meters)')
# 		cbar = plt.colorbar()
# 		cbar.set_label('OPD (microns)',rotation=-90,labelpad=20)
# 		plt.savefig(savefilename,bbox_inches='tight')
# 		plt.close()
# 		self.surfaceplot_fname = savefilename




# 	def combine_result_pngs(self,
# 							outfile="combined_results.png",
# 							dpi=200):

# 		main_png = self.datamodelresid_fname
# 		table_png = self.resulttable_fname
# 		opd_png = self.surfaceplot_fname
# 		"""
# 		Combine three PNGs into one summary figure.

# 		Layout:
# 		- top row: main_png spans full width
# 		- bottom left: opd_png
# 		- bottom right: table_png
# 		"""

# 		img_main = mpimg.imread(main_png)
# 		img_table = mpimg.imread(table_png)
# 		img_opd = mpimg.imread(opd_png)

# 		fig = plt.figure(figsize=(12, 14), constrained_layout=True)
# 		gs = GridSpec(
# 			2, 2,
# 			figure=fig,
# 			height_ratios=[1.2, 1.0],
# 			width_ratios=[1.0, 1.1]
# 		)

# 		ax_main = fig.add_subplot(gs[0, :])
# 		ax_opd = fig.add_subplot(gs[1, 0])
# 		ax_table = fig.add_subplot(gs[1, 1])

# 		ax_main.imshow(img_main)
# 		ax_opd.imshow(img_opd)
# 		ax_table.imshow(img_table)

# 		ax_main.axis("off")
# 		ax_opd.axis("off")
# 		ax_table.axis("off")

# 		plt.savefig(outfile, dpi=dpi, bbox_inches="tight")
# 		plt.close(fig)

# 		print(f"Saved combined figure to {outfile}")

# 	def save_zernike_dat(self, filename):
# 		"""
# 		Save self.resultsdict in the same format as the LMT OOF
# 		zernike .dat file.

# 		Supports both the named low-order Zernike parameters and
# 		higher-order parameters stored as "OSA Index N".
# 		"""

# 		# --------------------------------------------------
# 		# Full ordered list used by the LMT .dat format
# 		# --------------------------------------------------

# 		zernike_rows = [
# 			(0,  "BIAS"),
# 			(1,  "TILT_H"),
# 			(2,  "TILT_V"),
# 			(3,  "FOCUS"),
# 			(4,  "AST_V"),
# 			(5,  "AST_O"),
# 			(6,  "COMA_H"),
# 			(7,  "COMA_V"),
# 			(8,  "TRE_O"),
# 			(9,  "TRE_V"),
# 			(10, "SPH"),
# 			(11, "2AST_V"),
# 			(12, "2AST_O"),
# 			(13, "TET_V"),
# 			(14, "TET_O"),
# 			(15, "2COMA_H"),
# 			(16, "2COMA_V"),
# 			(17, "2TRE_O"),
# 			(18, "2TRE_V"),
# 			(19, "PEN_O"),
# 			(20, "PEN_V"),
# 			(21, "2SPH"),
# 			(22, "3AST_V"),
# 			(23, "3AST_O"),
# 			(24, "2TET_V"),
# 			(25, "2TET_O"),
# 			(26, "HEX_V"),
# 			(27, "HEX_O"),
# 			(28, "3COMA_H"),
# 			(29, "3COMA_V"),
# 			(30, "3TRE_O"),
# 			(31, "3TRE_V"),
# 			(32, "2PEN_O"),
# 			(33, "2PEN_V"),
# 			(34, "HEPT_O"),
# 			(35, "HEPT_V"),
# 			(36, "3SPH"),
# 			(37, "4AST_V"),
# 			(38, "4AST_O"),
# 			(39, "3TET_V"),
# 			(40, "3TET_O"),
# 			(41, "2HEX_V"),
# 			(42, "2HEX_O"),
# 			(43, "OCT_V"),
# 			(44, "OCT_O"),
# 		]

# 		# --------------------------------------------------
# 		# Map .dat names -> names used in self.resultsdict
# 		#
# 		# These preserve your existing naming convention.
# 		# --------------------------------------------------

# 		result_mapping = {
# 			"AST_V":   "AST_V",
# 			"AST_O":   "AST_O",
# 			"COMA_H":  "COMA_H",
# 			"COMA_V":  "COMA_V",
# 			"TRE_O":   "TRE_O",
# 			"TRE_V":   "TRE_V",
# 			"SPH":     "SPH",
# 			"2AST_V":  "AST2_V",
# 			"2AST_O":  "AST2_O",
# 			"TET_V":   "QUAD_V",
# 			"TET_O":   "QUAD_O",
# 		}

# 		# --------------------------------------------------
# 		# Write file
# 		# --------------------------------------------------

# 		with open(filename, "w") as f:

# 			for index, dat_label in zernike_rows:

# 				value = 0.0
# 				flag = 0

# 				# ------------------------------------------
# 				# First check for one of your explicitly
# 				# named Zernike coefficients.
# 				# ------------------------------------------

# 				if dat_label in result_mapping:

# 					result_label = result_mapping[dat_label]

# 					if result_label in self.resultsdict:
# 						value = float(self.resultsdict[result_label])
# 						flag = 1

# 				# ------------------------------------------
# 				# Otherwise check whether this mode was
# 				# stored generically as "OSA Index N".
# 				# ------------------------------------------

# 				else:

# 					osa_label = f"OSA Index {index}"

# 					if osa_label in self.resultsdict:
# 						value = float(self.resultsdict[osa_label])
# 						flag = 1

# 				f.write(
# 					f"{index:d} {flag:d} {dat_label} {value:.6f}\n"
# 				)

# 	def save_subref_dat(self, filename):
# 		"""
# 		Save the M2 position offsets in the same format as the LMT OOF
# 		subref.dat file.

# 		Assumes self.resultsdict['M2.Z_offset'] is in meters.
# 		Writes M2Z in microns.
# 		"""

# 		# Convert meters -> microns
# 		m2z_microns = float(self.resultsdict["M2.Z_offset"]) * 1e6

# 		with open(filename, "w") as f:
# 			f.write(f"0 1 M2Z {m2z_microns:.6f}\n")
# 			f.write(f"1 0 M2X {0.0:.6f}\n")
# 			f.write(f"2 0 M2Y {0.0:.6f}\n")



######################################


class fit_beam_with_glob_tilt_M2offsets_leastsquares(_LeastSquaresUncertainties):

	def __init__(self,beam_class):

		print('Initializing the fit beam class ')


		mapcounter = len(beam_class.trunc_maps)
		print(f'There are {mapcounter} maps')
		print('The vector of model parameters will have the form:')
		parameter_names = [
			'source_amplitude', 'M2.Z_offset',
			'a_M2X', 'b_M2X', 'a_M2Y', 'b_M2Y',
			'TILT_H', 'TILT_V',
			'AST_O', 'AST_V', 'TRE_V',
		]
		for ind, name in enumerate(parameter_names):
			print(f'x[{ind}] = {name}')
		print('x[11:] = remaining fitted higher-order Zernikes')

		self.number_of_maps = mapcounter
		# x[8:] is mapped to c[3], c[5], c[6], c[9:]
		self.tilt_offset_end_index = 8  # retained for older methods
		self.tilt_offset_start_index = 6
		self.fit_vec_size = beam_class.zernike_polynomials.shape[0] + 2

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

		chi_squared = np.zeros(self.tmpbeamclass.trunc_maps['map0'].shape)

		source_amp = x[0]
		M2z_offset = x[1]

		a_m2x = x[2]
		b_m2x = x[3]

		a_m2y = x[4]
		b_m2y = x[5]

		tilt_counter = self.tilt_offset_start_index
		residual_maps = {}
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

			ctmp[1] = x[6]
			tilt_counter+=1
			ctmp[2] = x[7]
			tilt_counter+=1

			ctmp[3] = x[self.tilt_offset_end_index]
			ctmp[5] = x[self.tilt_offset_end_index+1]
			ctmp[6] = x[self.tilt_offset_end_index+2]
			ctmp[9:] = x[self.tilt_offset_end_index+3:]

			tmpdelx = a_m2x+(b_m2x*self.tmpbeamclass.m2z_vals[i])
			tmpdely = a_m2y+(b_m2y*self.tmpbeamclass.m2z_vals[i])

			modelbeam = source_amp*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+M2z_offset,
							   del_x=tmpdelx,del_y=tmpdely,del_alph_x=0.,del_alph_y=0.,
							   f=17.5,F=525.,D=50.)

			residual = self.tmpbeamclass.trunc_maps[i] - modelbeam
			residual_maps[i] = residual

			chi_squared+= residual**2

		tmpreturn = np.concatenate([residual.ravel() for residual in residual_maps.values()])
		self.fitting_counter+=1
		self.temp_cost = np.sqrt(np.mean(chi_squared))
		
		return tmpreturn

	def make_bounds(self,tilt_plusminus,zern_plusminus):
		tmpbounds = []

		lowerbounds = []
		upperbounds = []

		for i in range(len(self.x0)):
			if i==0:
				# lowerbounds.append(0.7*self.x0[0])
				# upperbounds.append(1.3*self.x0[0])
				lowerbounds.append(0)
				upperbounds.append(np.inf)
			elif i==1:
				# lowerbounds.append(-5E-3)
				# upperbounds.append(5E-3)
				lowerbounds.append(-np.inf)
				upperbounds.append(np.inf)
			elif i in [2,3,4]:
				# lowerbounds.append(-tilt_plusminus)
				# upperbounds.append(tilt_plusminus)
				lowerbounds.append(-np.inf)
				upperbounds.append(np.inf)
			elif i==self.tilt_offset_end_index+8:
				# lowerbounds.append(-tilt_plusminus)
				# upperbounds.append(tilt_plusminus)
				lowerbounds.append(-150)
				upperbounds.append(150)
			else:
				# lowerbounds.append(-zern_plusminus)
				# upperbounds.append(zern_plusminus)
				lowerbounds.append(-np.inf)
				upperbounds.append(np.inf)
		lowerbounds = tuple(lowerbounds)
		upperbounds = tuple(upperbounds)
		return (lowerbounds,upperbounds)



	def run_fitter(self,tilt_plusminus=None,zern_plusminus=None,
					newx0 = None):
		# results = minimize(self.chisquared,x0=self.x0)
		# self.results = results

		if newx0 is not None:
			self.x0 = newx0

		tmpbound = self.make_bounds(1000,350)

		self.temp_cost = 1E6
		
		previous_cost = [None]
		small_change_counter = [0]

		

		self.results = least_squares(
			self.chisquared,
			self.x0,
			x_scale='jac'
		)
		ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
		ctmp[3] = self.results.x[8]
		ctmp[5] = self.results.x[9]
		ctmp[6] = self.results.x[10]
		ctmp[9:] = self.results.x[11:]
		x = self.results.x
		map_names = list(self.tmpbeamclass.trunc_maps)
		middle_map = map_names[len(map_names) // 2]
		z_middle = float(self.tmpbeamclass.m2z_vals[middle_map])

		self.strehl_ratio = float(np.amax(
		    self.tmpbeamclass.make_psf(
		        c=ctmp,
		        secondary_offset=z_middle + x[1],
		        del_x=x[2] + x[3] * z_middle,
		        del_y=x[4] + x[5] * z_middle,
		        del_alph_x=0.,
		        del_alph_y=0.,
		        f=17.5, F=525., D=50.,
		    )
		))
		# self.strehl_ratio = np.amax(self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.results.x[1],
		# 					   del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
		# 					   f=17.5,F=525.,D=50.))


	def plot_fit_results(self,vmax_frac_of_source_flux = 0.2,resids_stretch=5,vmin_val = -50,
					 title=None,savefigname=None,showplot=False):

		#plt.figure(figsize=(15,8))
		fig = plt.figure(figsize=(3.0*self.number_of_maps + 1.2, 8))
		# plt.figure()
		subplotcounter=1
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			corners_tmp = np.rad2deg(enmap.corners(self.tmpbeamclass.trunc_maps[i].shape,self.tmpbeamclass.trunc_maps[i].wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(self.tmpbeamclass.trunc_maps[i],extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			post_stamp_size = 1.5*60.#1.5/60.
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		fit_a_m2x = self.results.x[2]
		fit_b_m2x = self.results.x[3]

		fit_a_m2y = self.results.x[4]
		fit_b_m2y = self.results.x[5]

		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

			ctmp[1] = self.results.x[6]
			tilt_counter+=1
			ctmp[2] = self.results.x[7]
			tilt_counter+=1


			ctmp[3] = self.results.x[8]
			ctmp[5] = self.results.x[9]
			ctmp[6] = self.results.x[10]
			ctmp[9:] = self.results.x[11:]

			tmpdelx = fit_a_m2x+(fit_b_m2x*self.tmpbeamclass.m2z_vals[i])
			tmpdely = fit_a_m2y+(fit_b_m2y*self.tmpbeamclass.m2z_vals[i])

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,
				secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
				del_x=tmpdelx,del_y=tmpdely)
			

			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(tmpmodelbeam,extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			if subplotcounter == self.number_of_maps + 1:
				plt.ylabel('El Offset (arcsec)')
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

			ctmp[1] = self.results.x[6]
			tilt_counter+=1
			ctmp[2] = self.results.x[7]
			tilt_counter+=1
			

			ctmp[3] = self.results.x[8]
			ctmp[5] = self.results.x[9]
			ctmp[6] = self.results.x[10]
			ctmp[9:] = self.results.x[11:]

			tmpdelx = fit_a_m2x+(fit_b_m2x*self.tmpbeamclass.m2z_vals[i])
			tmpdely = fit_a_m2y+(fit_b_m2y*self.tmpbeamclass.m2z_vals[i])

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,
				secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
				del_x=tmpdelx,del_y=tmpdely)
			

			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(100*(self.tmpbeamclass.trunc_maps[i]-tmpmodelbeam)/self.results.x[0],extent=imextent_tmp,origin='lower',vmin=-resids_stretch,vmax=resids_stretch)
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if subplotcounter == 2*self.number_of_maps + 2:
				plt.xlabel('Az Offset (arcsec)')
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])

			subplotcounter+=1

		#fig = plt.gcf()

		nrows = 3
		ncols = self.number_of_maps

		# Optional: leave some room on the right for colorbars
		#fig.subplots_adjust(right=0.88)

		fig.subplots_adjust(
			left=0.10,
			right=0.88,
			bottom=0.06,
			top=0.92,
			wspace=0.03,
			hspace=0.20
		)

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
				pos.x0 - 0.06,     # shift a little left of the first subplot
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
		self.datamodelresid_fname = savefigname


	def _fitted_zernikes(self, include_tilt=True):
		"""Reconstruct exactly the Zernike coefficient vector used in chisquared."""
		x = self.results.x
		c = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
		if include_tilt:
			c[1], c[2] = x[6], x[7]
		c[3] = x[8]
		c[5] = x[9]
		c[6] = x[10]
		c[9:] = x[11:]
		return c

	def save_results(self, savefilename):

		savefilename = os.fspath(savefilename)

		# Zernike labels indexed by their position in ctmp.
		# These follow the naming convention in your existing code.
		zernike_labels = {
			1:  'TILT_H',
			2:  'TILT_V',
			3:  'AST_O',
			5:  'AST_V',
			6:  'TRE_V',
			9:  'TRE_O',
			10: 'QUAD_O',
			11: 'AST2_O',
			12: 'SPH',
			13: 'AST2_V',
			14: 'QUAD_V',
		}

		results_dict = {}

		# --------------------------------------------------
		# General fit parameters
		# --------------------------------------------------

		x = self.results.x

		results_dict['source_amp'] = float(x[0])
		# Model defocus = recorded absolute M2.Z + fitted offset.
		results_dict['M2.Z_offset'] = float(x[1])
		results_dict['M2.Z_target'] = -float(x[1])
		# Evaluate the lateral-offset model at the fitted focus position.
		results_dict['M2.X_target'] = float(x[2] + x[3] * results_dict['M2.Z_target'])
		results_dict['M2.Y_target'] = float(x[4] + x[5] * results_dict['M2.Z_target'])
		# Use the middle map in the same order as the fitted maps.
		# For an even count, select the later of the two middle maps.
		map_names = list(self.tmpbeamclass.trunc_maps)
		middle_map = map_names[len(map_names) // 2]
		results_dict['M2.Z_middle'] = float(self.tmpbeamclass.m2z_vals[middle_map])
		# Signed move from the middle map's absolute position to fitted focus.
		results_dict['M2.Z_move'] = (results_dict['M2.Z_target']
			- results_dict['M2.Z_middle'])
		results_dict['strehl_ratio'] = float(self.strehl_ratio)

		# --------------------------------------------------
		# M2 lateral-offset model
		# del_x = a_m2x + b_m2x * M2.Z
		# del_y = a_m2y + b_m2y * M2.Z
		# --------------------------------------------------

		results_dict['a_m2x'] = float(x[2])
		results_dict['b_m2x'] = float(x[3])
		results_dict['a_m2y'] = float(x[4])
		results_dict['b_m2y'] = float(x[5])

		# --------------------------------------------------
		# Global Zernike tilts
		# --------------------------------------------------

		results_dict['TILT_H'] = float(x[6])
		results_dict['TILT_V'] = float(x[7])

		# --------------------------------------------------
		# Per-map M2 lateral offsets
		# --------------------------------------------------

		for map_name in self.tmpbeamclass.trunc_maps:

			m2z = float(self.tmpbeamclass.m2z_vals[map_name])

			del_x = x[2] + x[3] * m2z
			del_y = x[4] + x[5] * m2z

			results_dict[f'M2.X_{map_name}'] = float(del_x)
			results_dict[f'M2.Y_{map_name}'] = float(del_y)

		# --------------------------------------------------
		# Reconstruct Zernike coefficient vector
		# Identical indexing to chisquared()
		# --------------------------------------------------

		ctmp = np.zeros(
			self.tmpbeamclass.zernike_polynomials.shape[0]
		)

		ctmp[1] = x[6]
		ctmp[2] = x[7]

		ctmp[3] = x[8]
		ctmp[5] = x[9]
		ctmp[6] = x[10]

		ctmp[9:] = x[11:]

		# Only save coefficients actually included in fit.
		# Indices 7 and 8 (coma) are deliberately excluded.

		fitted_indices = [3, 5, 6] + list(range(9, len(ctmp)))

		for idx in fitted_indices:

			label = zernike_labels.get(
				idx,
				f'OSA Index {idx}'
			)

			results_dict[label] = float(ctmp[idx])

		# --------------------------------------------------
		# Print results
		# --------------------------------------------------

		for key, value in results_dict.items():
			print(key, value)

		# --------------------------------------------------
		# Save JSON
		# --------------------------------------------------

		with open(savefilename, "w") as f:
			json.dump(results_dict, f, indent=4)

		self.resultsdict = results_dict

		# --------------------------------------------------
		# Save results as a PNG table
		# --------------------------------------------------

		table_rows = []

		for label, value in results_dict.items():

			# Keep per-map offsets in the JSON, but show only targets in the table.
			if label.startswith(('M2.X_', 'M2.Y_')) and not label.endswith('_target'):
				continue

			if label == 'source_amp':
				unit = 'mJy/beam'
				display_value = value

			elif label in ('M2.Z_offset', 'M2.Z_target', 'M2.Z_middle', 'M2.Z_move'):
				unit = 'mm'
				display_value = value * 1e3

			elif label == 'strehl_ratio':
				unit = ''
				display_value = value

			elif label.startswith(('M2.X_', 'M2.Y_')):
				unit = 'microns'
				display_value = value * 1e6

			elif label in ('a_m2x', 'a_m2y'):
				unit = 'microns'
				display_value = value * 1e6

			elif label in ('b_m2x', 'b_m2y'):
				unit = 'microns/mm'
				display_value = value * 1e3

			else:
				# Zernike coefficients, including global tilts
				unit = 'microns'
				display_value = value

			table_rows.append([
				label,
				f"{display_value:.4f}",
				unit
			])

		# --------------------------------------------------
		# Make table figure
		# --------------------------------------------------

		fig_height = max(
			2.0,
			0.35 * (len(table_rows) + 1)
		)

		fig, ax = plt.subplots(
			figsize=(6, fig_height)
		)

		ax.axis("off")

		table = ax.table(
			cellText=table_rows,
			colLabels=["LABEL", "VALUE", "UNITS"],
			cellLoc="center",
			colLoc="center",
			loc="center",
		)

		table.auto_set_font_size(False)
		table.set_fontsize(11)
		table.scale(1.0, 1.4)

		for (row, col), cell in table.get_celld().items():

			cell.set_edgecolor("black")
			cell.set_linewidth(1.0)

			if row == 0:
				cell.set_facecolor("0.9")
			else:
				cell.set_facecolor("0.96")

		# --------------------------------------------------
		# Save PNG
		# --------------------------------------------------

		pngfilename = (
			os.path.splitext(savefilename)[0]
			+ ".png"
		)

		plt.savefig(
			pngfilename,
			dpi=200,
			bbox_inches="tight"
		)

		plt.close(fig)

		self.resulttable_fname = pngfilename

	def _uncertainty_model(self, x):
		"""Parameter names and result gradients for this fitted model."""
		zernike_labels = {
			3: 'AST_O', 5: 'AST_V', 6: 'TRE_V', 9: 'TRE_O',
			10: 'QUAD_O', 11: 'AST2_O', 12: 'SPH',
			13: 'AST2_V', 14: 'QUAD_V',
		}
		indices = [3, 5, 6] + list(range(9, self.tmpbeamclass.zernike_polynomials.shape[0]))
		names = ['source_amp', 'M2.Z_offset',
			'a_m2x', 'b_m2x', 'a_m2y', 'b_m2y',
			'TILT_H', 'TILT_V']
		names += [zernike_labels.get(idx, f'OSA Index {idx}') for idx in indices]
		basis = np.eye(x.size)
		map_names = list(self.tmpbeamclass.trunc_maps)
		z_middle = float(self.tmpbeamclass.m2z_vals[map_names[len(map_names) // 2]])
		gradients = {
			'source_amp': basis[0],
			'M2.Z_offset': basis[1],
			'M2.Z_target': -basis[1],
			'M2.X_target': basis[2] - x[1] * basis[3] - x[3] * basis[1],
			'M2.Y_target': basis[4] - x[1] * basis[5] - x[5] * basis[1],
			'M2.Z_middle': np.zeros(x.size),
			'M2.Z_move': -basis[1],
		}

		def strehl_at(parameters):
			c = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
			c[indices] = parameters[8:]
			return float(np.amax(self.tmpbeamclass.make_psf(
				c=c, secondary_offset=z_middle + parameters[1],
				del_x=parameters[2] + parameters[3] * z_middle,
				del_y=parameters[4] + parameters[5] * z_middle,
				del_alph_x=0., del_alph_y=0.,
				f=17.5, F=525., D=50.,
			)))

		strehl_gradient = np.zeros(x.size)
		for idx in [1, 2, 3, 4, 5] + list(range(8, x.size)):
			step = np.cbrt(np.finfo(float).eps) * max(abs(x[idx]), 1e-3 if idx == 1 else 1.)
			delta = step * basis[idx]
			strehl_gradient[idx] = (strehl_at(x + delta) - strehl_at(x - delta)) / (2 * step)
		gradients['strehl_ratio'] = strehl_gradient
		for idx in range(2, 8):
			gradients[names[idx]] = basis[idx]
		for map_name in map_names:
			z = float(self.tmpbeamclass.m2z_vals[map_name])
			gradients[f'M2.X_{map_name}'] = basis[2] + z * basis[3]
			gradients[f'M2.Y_{map_name}'] = basis[4] + z * basis[5]
		for idx in range(8, x.size):
			gradients[names[idx]] = basis[idx]
		return names, gradients

	def surface_plot(self, savefilename, vmin=-400, vmax=400):
		"""Plot the common fitted surface terms (excluding global pointing tilt).

		M2 translations produce additional OPD in make_psf and are not
		included in this Zernike-only surface plot.
		"""
		cfit = self._fitted_zernikes(include_tilt=False)
		OPD = np.tensordot(cfit, self.tmpbeamclass.zernike_polynomials,
			axes=([0], [0]))
		plt.figure()
		plt.imshow(OPD, extent=[-self.tmpbeamclass.L/2., self.tmpbeamclass.L/2.,
			-self.tmpbeamclass.L/2., self.tmpbeamclass.L/2.], vmin=vmin, vmax=vmax)
		plt.xlim(-30, 30)
		plt.ylim(-30, 30)
		plt.xlabel('x (meters)')
		plt.ylabel('y (meters)')
		plt.colorbar().set_label('OPD (microns)', rotation=-90, labelpad=20)
		plt.savefig(savefilename, bbox_inches='tight')
		plt.close()
		self.surfaceplot_fname = savefilename

	def combine_result_pngs(self,
							outfile="combined_results.png",
							dpi=200):

		main_png = self.datamodelresid_fname
		table_png = self.resulttable_fname
		opd_png = self.surfaceplot_fname
		"""
		Combine three PNGs into one summary figure.

		Layout:
		- top row: main_png spans full width
		- bottom left: opd_png
		- bottom right: table_png
		"""

		img_main = mpimg.imread(main_png)
		img_table = mpimg.imread(table_png)
		img_opd = mpimg.imread(opd_png)

		fig = plt.figure(figsize=(12, 14), constrained_layout=True)
		gs = GridSpec(
			2, 2,
			figure=fig,
			height_ratios=[1.2, 1.0],
			width_ratios=[1.0, 1.1]
		)

		ax_main = fig.add_subplot(gs[0, :])
		ax_opd = fig.add_subplot(gs[1, 0])
		ax_table = fig.add_subplot(gs[1, 1])

		ax_main.imshow(img_main)
		ax_opd.imshow(img_opd)
		ax_table.imshow(img_table)

		ax_main.axis("off")
		ax_opd.axis("off")
		ax_table.axis("off")

		plt.savefig(outfile, dpi=dpi, bbox_inches="tight")
		plt.close(fig)

		print(f"Saved combined figure to {outfile}")

	def save_zernike_dat(self, filename):
		"""
		Save self.resultsdict in the same format as the LMT OOF
		zernike .dat file.

		Supports both the named low-order Zernike parameters and
		higher-order parameters stored as "OSA Index N".
		"""

		# --------------------------------------------------
		# Full ordered list used by the LMT .dat format
		# --------------------------------------------------

		zernike_rows = [
			(0,  "BIAS"),
			(1,  "TILT_H"),
			(2,  "TILT_V"),
			(3,  "FOCUS"),
			(4,  "AST_V"),
			(5,  "AST_O"),
			(6,  "COMA_H"),
			(7,  "COMA_V"),
			(8,  "TRE_O"),
			(9,  "TRE_V"),
			(10, "SPH"),
			(11, "2AST_V"),
			(12, "2AST_O"),
			(13, "TET_V"),
			(14, "TET_O"),
			(15, "2COMA_H"),
			(16, "2COMA_V"),
			(17, "2TRE_O"),
			(18, "2TRE_V"),
			(19, "PEN_O"),
			(20, "PEN_V"),
			(21, "2SPH"),
			(22, "3AST_V"),
			(23, "3AST_O"),
			(24, "2TET_V"),
			(25, "2TET_O"),
			(26, "HEX_V"),
			(27, "HEX_O"),
			(28, "3COMA_H"),
			(29, "3COMA_V"),
			(30, "3TRE_O"),
			(31, "3TRE_V"),
			(32, "2PEN_O"),
			(33, "2PEN_V"),
			(34, "HEPT_O"),
			(35, "HEPT_V"),
			(36, "3SPH"),
			(37, "4AST_V"),
			(38, "4AST_O"),
			(39, "3TET_V"),
			(40, "3TET_O"),
			(41, "2HEX_V"),
			(42, "2HEX_O"),
			(43, "OCT_V"),
			(44, "OCT_O"),
		]

		# --------------------------------------------------
		# Map .dat names -> names used in self.resultsdict
		#
		# These preserve your existing naming convention.
		# --------------------------------------------------

		result_mapping = {
			"AST_V":   "AST_V",
			"AST_O":   "AST_O",
			"COMA_H":  "COMA_H",
			"COMA_V":  "COMA_V",
			"TRE_O":   "TRE_O",
			"TRE_V":   "TRE_V",
			"SPH":     "SPH",
			"2AST_V":  "AST2_V",
			"2AST_O":  "AST2_O",
			"TET_V":   "QUAD_V",
			"TET_O":   "QUAD_O",
		}

		# --------------------------------------------------
		# Write file
		# --------------------------------------------------

		with open(filename, "w") as f:

			for index, dat_label in zernike_rows:

				value = 0.0
				flag = 0

				# The global Zernike tilts are fitted (unlike per-map tilts).
				if dat_label in ('TILT_H', 'TILT_V') and dat_label in self.resultsdict:
					value = float(self.resultsdict[dat_label])
					flag = 1

				# ------------------------------------------
				# First check for one of your explicitly
				# named Zernike coefficients.
				# ------------------------------------------

				if dat_label in result_mapping:

					result_label = result_mapping[dat_label]

					if result_label in self.resultsdict:
						value = float(self.resultsdict[result_label])
						flag = 1

				# ------------------------------------------
				# Otherwise check whether this mode was
				# stored generically as "OSA Index N".
				# ------------------------------------------

				else:

					osa_label = f"OSA Index {index}"

					if osa_label in self.resultsdict:
						value = float(self.resultsdict[osa_label])
						flag = 1

				f.write(
					f"{index:d} {flag:d} {dat_label} {value:.6f}\n"
				)

	def save_subref_dat(self, filename, reference_m2z=0.0):
		"""Write the absolute axial focus target and lateral offsets in microns.

		M2Z is -x[1], the absolute coordinate where modeled defocus is zero.
		M2X/M2Y remain lateral offsets evaluated at reference_m2z; absolute
		lateral targets cannot be inferred without the telescope X/Y coordinates.
		All arguments and fitted translations are in meters. Lateral slopes
		are saved in JSON, not represented in this constant-value .dat file.
		"""
		x = self.results.x
		m2x = float(x[2] + x[3] * reference_m2z)
		m2y = float(x[4] + x[5] * reference_m2z)
		with open(filename, 'w') as f:
			f.write(f"0 1 M2Z {-float(x[1]) * 1e6:.6f}\n")
			f.write(f"1 1 M2X {m2x * 1e6:.6f}\n")
			f.write(f"2 1 M2Y {m2y * 1e6:.6f}\n")

class fit_beam_with_glob_tilt_M2tilts_fitIllum_leastsquares(_LeastSquaresUncertainties):
	"""Fit M2 tilts linear in the recorded M2.Z position.

	x[2] is the Gaussian half-power radius and illumination cutoff, in meters.
	x[3:7] = (a_alph_x, b_alph_x, a_alph_y, b_alph_y), where
	del_alph = a_alph + b_alph * recorded_M2Z. M2.Z and its fitted
	offset are in meters; modeled defocus is recorded_M2Z + x[1], so
	the absolute axial focus target is -x[1]. a_alph is in degrees and
	b_alph in degrees/m.
	Zernike coefficients (including the separate global pointing tilts)
	are in microns. Lateral M2 translations are fixed at zero.
	"""

	def __init__(self,beam_class):

		print('Initializing the fit beam class ')


		mapcounter = len(beam_class.trunc_maps)
		print(f'There are {mapcounter} maps')
		print('The vector of model parameters will have the form:')
		parameter_names = [
			'source_amplitude', 'M2.Z_offset', 'edge_taper_radius',
			'a_m2alph_x', 'b_m2alph_x', 'a_m2alph_y', 'b_m2alph_y',
			'TILT_H', 'TILT_V',
			'AST_O', 'AST_V', 'TRE_V',
		]
		for ind, name in enumerate(parameter_names):
			print(f'x[{ind}] = {name}')
		print('x[12:] = remaining fitted higher-order Zernikes')

		self.number_of_maps = mapcounter
		# x[9:] is mapped to c[3], c[5], c[6], c[9:]
		self.tilt_offset_end_index = 8+1  # retained for older methods
		self.tilt_offset_start_index = 6+1
		self.fit_vec_size = beam_class.zernike_polynomials.shape[0] + 2 + 1

		self.tmpbeamclass = beam_class

		x0 = np.zeros(self.fit_vec_size)

		map_maxes = np.zeros(self.number_of_maps)

		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			tmpmax = np.amax(self.tmpbeamclass.trunc_maps[i])
			map_maxes[count] = tmpmax
		ampguess = max(map_maxes)

		x0[0] = ampguess
		x0[1] = beam_class.m2z_vals[f'map{int(self.number_of_maps/2.)}']
		x0[2] = 0.5*48.

		self.x0 = x0

		self.fitting_counter = 0
		self.temp_cost = -999

	def _set_taper_radius(self, radius):
		diameter = 2. * radius
		self.tmpbeamclass.set_illumination(aperture_fwhm=diameter, edge_taper_diameter=diameter)
		self.tmpbeamclass.make_normalizing_amplitude()

	def chisquared(self,x):

		if self.fitting_counter%500==0:
			print('On fitting iteration = ',self.fitting_counter, ' with Cost = ',self.temp_cost)

		chi_squared = np.zeros(self.tmpbeamclass.trunc_maps['map0'].shape)

		source_amp = x[0]
		M2z_offset = x[1]

		a_m2alph_x, b_m2alph_x, a_m2alph_y, b_m2alph_y = x[3:7]

		self._set_taper_radius(x[2])

		tilt_counter = self.tilt_offset_start_index
		residual_maps = {}
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

			ctmp[1] = x[7]
			tilt_counter+=1
			ctmp[2] = x[8]
			tilt_counter+=1

			ctmp[3] = x[self.tilt_offset_end_index]
			ctmp[5] = x[self.tilt_offset_end_index+1]
			ctmp[6] = x[self.tilt_offset_end_index+2]
			ctmp[9:] = x[self.tilt_offset_end_index+3:]

			tmpalph_x = a_m2alph_x+(b_m2alph_x*self.tmpbeamclass.m2z_vals[i])
			tmpalph_y = a_m2alph_y+(b_m2alph_y*self.tmpbeamclass.m2z_vals[i])

			modelbeam = source_amp*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+M2z_offset,
							   del_x=0.0,del_y=0.0,del_alph_x=tmpalph_x,del_alph_y=tmpalph_y,
							   f=17.5,F=525.,D=50.)

			residual = self.tmpbeamclass.trunc_maps[i] - modelbeam
			residual_maps[i] = residual

			chi_squared+= residual**2

		tmpreturn = np.concatenate([residual.ravel() for residual in residual_maps.values()])
		self.fitting_counter+=1
		self.temp_cost = np.sqrt(np.mean(chi_squared))
		
		return tmpreturn

	def make_bounds(self,tilt_plusminus=None,zern_plusminus=None):
		"""Bound amplitude, taper radius, and optionally the Zernikes.

		tilt_plusminus limits global Zernike pointing tilts (x[7:9]),
		and zern_plusminus limits the other fitted Zernikes (x[9:]),
		both in microns. None leaves those coefficients unbounded.
		"""
		lower = np.full(len(self.x0), -np.inf)
		upper = np.full(len(self.x0), np.inf)
		lower[0] = 0.0
		lower[2] = 0.5*44.#np.finfo(float).eps

		upper[2] = 0.5*50.


		for limit, indices in ((tilt_plusminus, slice(7, 9)),
							   (zern_plusminus, slice(9, None))):
			if limit is not None:
				if not np.isfinite(limit) or limit <= 0:
					raise ValueError('Zernike bounds must be finite and positive')
				lower[indices] = -limit
				upper[indices] = limit
		return lower, upper

	def run_fitter(self,tilt_plusminus=None,zern_plusminus=None,
					newx0 = None):

		if newx0 is not None:
			self.x0 = newx0

		tmpbound = self.make_bounds(tilt_plusminus,zern_plusminus)

		self.temp_cost = 1E6

		self.results = least_squares(
			self.chisquared,
			self.x0,
			bounds=tmpbound,
			x_scale='jac'
		)
		ctmp = self._fitted_zernikes(include_tilt=False)
		x = self.results.x
		map_names = list(self.tmpbeamclass.trunc_maps)
		middle_map = map_names[len(map_names) // 2]
		z_middle = float(self.tmpbeamclass.m2z_vals[middle_map])

		self._set_taper_radius(x[2])

		# Match the offsets fitter: evaluate the middle map without global
		# pointing tilts or source amplitude, retaining fitted M2 aberrations.
		self.strehl_ratio = float(np.amax(
			self.tmpbeamclass.make_psf(
				c=ctmp,
				secondary_offset=z_middle+ x[1],
				del_x=0., del_y=0.,
				del_alph_x=x[3] + x[4] * z_middle,
				del_alph_y=x[5] + x[6] * z_middle,
				f=17.5, F=525., D=50.,
			)
		))


	def plot_fit_results(self,vmax_frac_of_source_flux = 0.2,resids_stretch=5,vmin_val = -50,
					 title=None,savefigname=None,showplot=False):

		#plt.figure(figsize=(15,8))
		fig = plt.figure(figsize=(3.0*self.number_of_maps + 1.2, 8))
		# plt.figure()
		subplotcounter=1
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			corners_tmp = np.rad2deg(enmap.corners(self.tmpbeamclass.trunc_maps[i].shape,self.tmpbeamclass.trunc_maps[i].wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(self.tmpbeamclass.trunc_maps[i],extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			post_stamp_size = 1.5*60.#1.5/60.
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		self._set_taper_radius(self.results.x[2])
		ctmp = self._fitted_zernikes()
		fit_a_m2alph_x, fit_b_m2alph_x, fit_a_m2alph_y, fit_b_m2alph_y = self.results.x[3:7]

		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			tmpalph_x = fit_a_m2alph_x+(fit_b_m2alph_x*self.tmpbeamclass.m2z_vals[i])
			tmpalph_y = fit_a_m2alph_y+(fit_b_m2alph_y*self.tmpbeamclass.m2z_vals[i])

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,
				secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
				del_x=0.,del_y=0.,
				del_alph_x=tmpalph_x,del_alph_y=tmpalph_y)
			

			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(tmpmodelbeam,extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			if subplotcounter == self.number_of_maps + 1:
				plt.ylabel('El Offset (arcsec)')
			subplotcounter+=1
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			tmpalph_x = fit_a_m2alph_x+(fit_b_m2alph_x*self.tmpbeamclass.m2z_vals[i])
			tmpalph_y = fit_a_m2alph_y+(fit_b_m2alph_y*self.tmpbeamclass.m2z_vals[i])

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,
				secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
				del_x=0.,del_y=0.,
				del_alph_x=tmpalph_x,del_alph_y=tmpalph_y)
			

			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(100*(self.tmpbeamclass.trunc_maps[i]-tmpmodelbeam)/self.results.x[0],extent=imextent_tmp,origin='lower',vmin=-resids_stretch,vmax=resids_stretch)
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if subplotcounter == 2*self.number_of_maps + 2:
				plt.xlabel('Az Offset (arcsec)')
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])

			subplotcounter+=1

		#fig = plt.gcf()

		nrows = 3
		ncols = self.number_of_maps

		# Optional: leave some room on the right for colorbars
		#fig.subplots_adjust(right=0.88)

		fig.subplots_adjust(
			left=0.10,
			right=0.88,
			bottom=0.06,
			top=0.92,
			wspace=0.03,
			hspace=0.20
		)

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
				pos.x0 - 0.06,     # shift a little left of the first subplot
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
		self.datamodelresid_fname = savefigname


	def _fitted_zernikes(self, include_tilt=True):
		"""Reconstruct exactly the Zernike coefficient vector used in chisquared."""
		x = self.results.x
		c = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
		if include_tilt:
			c[1:3] = x[7:9]
		c[3] = x[9]
		c[5] = x[10]
		c[6] = x[11]
		c[9:] = x[12:]
		return c

	def save_results(self, savefilename):

		savefilename = os.fspath(savefilename)

		# Zernike labels indexed by their position in ctmp.
		# These follow the naming convention in your existing code.
		zernike_labels = {
			1:  'TILT_H',
			2:  'TILT_V',
			3:  'AST_O',
			5:  'AST_V',
			6:  'TRE_V',
			9:  'TRE_O',
			10: 'QUAD_O',
			11: 'AST2_O',
			12: 'SPH',
			13: 'AST2_V',
			14: 'QUAD_V',
		}

		results_dict = {}

		# --------------------------------------------------
		# General fit parameters
		# --------------------------------------------------

		x = self.results.x

		results_dict['source_amp'] = float(x[0])
		# Model defocus = recorded absolute M2.Z + fitted offset.
		results_dict['M2.Z_offset'] = float(x[1])
		results_dict['M2.Z_target'] = -float(x[1])
		results_dict['edge_taper_radius'] = float(x[2])
		# Evaluate fitted angles at the axial focus target (degrees).
		results_dict['M2.alphX_target'] = float(x[3] + x[4] * results_dict['M2.Z_target'])
		results_dict['M2.alphY_target'] = float(x[5] + x[6] * results_dict['M2.Z_target'])
		# Select the later middle map for an even count, in fitting order.
		map_names = list(self.tmpbeamclass.trunc_maps)
		middle_map = map_names[len(map_names) // 2]
		results_dict['M2.Z_middle'] = float(self.tmpbeamclass.m2z_vals[middle_map])
		results_dict['M2.Z_move'] = (results_dict['M2.Z_target']
			- results_dict['M2.Z_middle'])
		results_dict['strehl_ratio'] = float(self.strehl_ratio)

		# --------------------------------------------------
		# M2 tilt model (degrees, with slopes in degrees/m)
		# del_alph_x = a_m2alph_x + b_m2alph_x * recorded_M2.Z
		# del_alph_y = a_m2alph_y + b_m2alph_y * recorded_M2.Z
		# --------------------------------------------------

		results_dict['a_m2alph_x'] = float(x[3])
		results_dict['b_m2alph_x'] = float(x[4])
		results_dict['a_m2alph_y'] = float(x[5])
		results_dict['b_m2alph_y'] = float(x[6])

		# --------------------------------------------------
		# Global Zernike tilts
		# --------------------------------------------------

		results_dict['TILT_H'] = float(x[7])
		results_dict['TILT_V'] = float(x[8])

		# --------------------------------------------------
		# Per-map M2 tilts (degrees)
		# --------------------------------------------------

		for map_name in self.tmpbeamclass.trunc_maps:

			m2z = float(self.tmpbeamclass.m2z_vals[map_name])

			del_alph_x = x[3] + x[4] * m2z
			del_alph_y = x[5] + x[6] * m2z

			results_dict[f'M2.alphX_{map_name}'] = float(del_alph_x)
			results_dict[f'M2.alphY_{map_name}'] = float(del_alph_y)

		# --------------------------------------------------
		# Reconstruct Zernike coefficient vector
		# Identical indexing to chisquared()
		# --------------------------------------------------

		ctmp = self._fitted_zernikes()

		# Only save coefficients actually included in fit.
		# Indices 7 and 8 (coma) are deliberately excluded.

		fitted_indices = [3, 5, 6] + list(range(9, len(ctmp)))

		for idx in fitted_indices:

			label = zernike_labels.get(
				idx,
				f'OSA Index {idx}'
			)

			results_dict[label] = float(ctmp[idx])

		# --------------------------------------------------
		# Print results
		# --------------------------------------------------

		for key, value in results_dict.items():
			print(key, value)

		# --------------------------------------------------
		# Save JSON
		# --------------------------------------------------

		with open(savefilename, "w") as f:
			json.dump(results_dict, f, indent=4)

		self.resultsdict = results_dict

		# --------------------------------------------------
		# Save results as a PNG table
		# --------------------------------------------------

		table_rows = []

		for label, value in results_dict.items():

			# Keep per-map angles in JSON; show target angles in the table.
			if label.startswith(('M2.alphX_', 'M2.alphY_')) and not label.endswith('_target'):
				continue

			if label == 'source_amp':
				unit = 'mJy/beam'
				display_value = value

			elif label in ('M2.Z_offset', 'M2.Z_target', 'M2.Z_middle', 'M2.Z_move'):
				unit = 'mm'
				display_value = value * 1e3

			elif label == 'edge_taper_radius':
				unit = 'm'
				display_value = value

			elif label == 'strehl_ratio':
				unit = ''
				display_value = value

			elif label.startswith(('M2.alphX_', 'M2.alphY_')):
				unit = 'degrees'
				display_value = value

			elif label in ('a_m2alph_x', 'a_m2alph_y'):
				unit = 'degrees'
				display_value = value

			elif label in ('b_m2alph_x', 'b_m2alph_y'):
				unit = 'degrees/mm'
				display_value = value * 1e-3

			else:
				# Zernike coefficients, including global tilts
				unit = 'microns'
				display_value = value

			table_rows.append([
				label,
				f"{display_value:.4f}",
				unit
			])

		# --------------------------------------------------
		# Make table figure
		# --------------------------------------------------

		fig_height = max(
			2.0,
			0.35 * (len(table_rows) + 1)
		)

		fig, ax = plt.subplots(
			figsize=(6, fig_height)
		)

		ax.axis("off")

		table = ax.table(
			cellText=table_rows,
			colLabels=["LABEL", "VALUE", "UNITS"],
			cellLoc="center",
			colLoc="center",
			loc="center",
		)

		table.auto_set_font_size(False)
		table.set_fontsize(11)
		table.scale(1.0, 1.4)

		for (row, col), cell in table.get_celld().items():

			cell.set_edgecolor("black")
			cell.set_linewidth(1.0)

			if row == 0:
				cell.set_facecolor("0.9")
			else:
				cell.set_facecolor("0.96")

		# --------------------------------------------------
		# Save PNG
		# --------------------------------------------------

		pngfilename = (
			os.path.splitext(savefilename)[0]
			+ ".png"
		)

		plt.savefig(
			pngfilename,
			dpi=200,
			bbox_inches="tight"
		)

		plt.close(fig)

		self.resulttable_fname = pngfilename

	def _uncertainty_model(self, x):
		"""Parameter names and result gradients for this fitted model."""
		zernike_labels = {
			3: 'AST_O', 5: 'AST_V', 6: 'TRE_V', 9: 'TRE_O',
			10: 'QUAD_O', 11: 'AST2_O', 12: 'SPH',
			13: 'AST2_V', 14: 'QUAD_V',
		}
		indices = [3, 5, 6] + list(range(9, self.tmpbeamclass.zernike_polynomials.shape[0]))
		names = ['source_amp', 'M2.Z_offset', 'edge_taper_radius',
			'a_m2alph_x', 'b_m2alph_x', 'a_m2alph_y', 'b_m2alph_y',
			'TILT_H', 'TILT_V']
		names += [zernike_labels.get(idx, f'OSA Index {idx}') for idx in indices]
		basis = np.eye(x.size)
		map_names = list(self.tmpbeamclass.trunc_maps)
		z_middle = float(self.tmpbeamclass.m2z_vals[map_names[len(map_names) // 2]])
		gradients = {
			'source_amp': basis[0],
			'M2.Z_offset': basis[1],
			'M2.Z_target': -basis[1],
			'M2.alphX_target': basis[3] - x[1] * basis[4] - x[4] * basis[1],
			'M2.alphY_target': basis[5] - x[1] * basis[6] - x[6] * basis[1],
			'M2.Z_middle': np.zeros(x.size),
			'M2.Z_move': -basis[1],
		}

		def strehl_at(parameters):
			c = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
			c[indices] = parameters[9:]
			return float(np.amax(self.tmpbeamclass.make_psf(
				c=c, secondary_offset=z_middle + parameters[1],
				del_x=0., del_y=0.,
				del_alph_x=parameters[3] + parameters[4] * z_middle,
				del_alph_y=parameters[5] + parameters[6] * z_middle,
				f=17.5, F=525., D=50.,
			)))

		strehl_gradient = np.zeros(x.size)
		step_fraction = np.cbrt(np.finfo(float).eps)
		radius_step = step_fraction * x[2]
		try:
			self._set_taper_radius(x[2] + radius_step)
			strehl_high = strehl_at(x)
			self._set_taper_radius(x[2] - radius_step)
			strehl_low = strehl_at(x)
			strehl_gradient[2] = (strehl_high - strehl_low) / (2 * radius_step)
		finally:
			self._set_taper_radius(x[2])
		for idx in [1, 3, 4, 5, 6] + list(range(9, x.size)):
			step = step_fraction * max(abs(x[idx]), 1e-3 if idx == 1 else 1.)
			delta = step * basis[idx]
			strehl_gradient[idx] = (strehl_at(x + delta) - strehl_at(x - delta)) / (2 * step)
		gradients['strehl_ratio'] = strehl_gradient
		for idx in range(2, x.size):
			gradients[names[idx]] = basis[idx]
		for map_name in map_names:
			z = float(self.tmpbeamclass.m2z_vals[map_name])
			gradients[f'M2.alphX_{map_name}'] = basis[3] + z * basis[4]
			gradients[f'M2.alphY_{map_name}'] = basis[5] + z * basis[6]
		return names, gradients

	def surface_plot(self, savefilename, vmin=-400, vmax=400):
		"""Plot the common fitted surface terms (excluding global pointing tilt).

		M2 tilts produce additional OPD in make_psf and are not
		included in this Zernike-only surface plot.
		"""
		cfit = self._fitted_zernikes(include_tilt=False)
		OPD = np.tensordot(cfit, self.tmpbeamclass.zernike_polynomials,
			axes=([0], [0]))
		plt.figure()
		plt.imshow(OPD, extent=[-self.tmpbeamclass.L/2., self.tmpbeamclass.L/2.,
			-self.tmpbeamclass.L/2., self.tmpbeamclass.L/2.], vmin=vmin, vmax=vmax)
		plt.xlim(-30, 30)
		plt.ylim(-30, 30)
		plt.xlabel('x (meters)')
		plt.ylabel('y (meters)')
		plt.colorbar().set_label('OPD (microns)', rotation=-90, labelpad=20)
		plt.savefig(savefilename, bbox_inches='tight')
		plt.close()
		self.surfaceplot_fname = savefilename

	def combine_result_pngs(self,
							outfile="combined_results.png",
							dpi=200):

		main_png = self.datamodelresid_fname
		table_png = self.resulttable_fname
		opd_png = self.surfaceplot_fname
		"""
		Combine three PNGs into one summary figure.

		Layout:
		- top row: main_png spans full width
		- bottom left: opd_png
		- bottom right: table_png
		"""

		img_main = mpimg.imread(main_png)
		img_table = mpimg.imread(table_png)
		img_opd = mpimg.imread(opd_png)

		fig = plt.figure(figsize=(12, 14), constrained_layout=True)
		gs = GridSpec(
			2, 2,
			figure=fig,
			height_ratios=[1.2, 1.0],
			width_ratios=[1.0, 1.1]
		)

		ax_main = fig.add_subplot(gs[0, :])
		ax_opd = fig.add_subplot(gs[1, 0])
		ax_table = fig.add_subplot(gs[1, 1])

		ax_main.imshow(img_main)
		ax_opd.imshow(img_opd)
		ax_table.imshow(img_table)

		ax_main.axis("off")
		ax_opd.axis("off")
		ax_table.axis("off")

		plt.savefig(outfile, dpi=dpi, bbox_inches="tight")
		plt.close(fig)

		print(f"Saved combined figure to {outfile}")

	def save_zernike_dat(self, filename):
		"""
		Save self.resultsdict in the same format as the LMT OOF
		zernike .dat file.

		Supports both the named low-order Zernike parameters and
		higher-order parameters stored as "OSA Index N".
		"""

		# --------------------------------------------------
		# Full ordered list used by the LMT .dat format
		# --------------------------------------------------

		zernike_rows = [
			(0,  "BIAS"),
			(1,  "TILT_H"),
			(2,  "TILT_V"),
			(3,  "FOCUS"),
			(4,  "AST_V"),
			(5,  "AST_O"),
			(6,  "COMA_H"),
			(7,  "COMA_V"),
			(8,  "TRE_O"),
			(9,  "TRE_V"),
			(10, "SPH"),
			(11, "2AST_V"),
			(12, "2AST_O"),
			(13, "TET_V"),
			(14, "TET_O"),
			(15, "2COMA_H"),
			(16, "2COMA_V"),
			(17, "2TRE_O"),
			(18, "2TRE_V"),
			(19, "PEN_O"),
			(20, "PEN_V"),
			(21, "2SPH"),
			(22, "3AST_V"),
			(23, "3AST_O"),
			(24, "2TET_V"),
			(25, "2TET_O"),
			(26, "HEX_V"),
			(27, "HEX_O"),
			(28, "3COMA_H"),
			(29, "3COMA_V"),
			(30, "3TRE_O"),
			(31, "3TRE_V"),
			(32, "2PEN_O"),
			(33, "2PEN_V"),
			(34, "HEPT_O"),
			(35, "HEPT_V"),
			(36, "3SPH"),
			(37, "4AST_V"),
			(38, "4AST_O"),
			(39, "3TET_V"),
			(40, "3TET_O"),
			(41, "2HEX_V"),
			(42, "2HEX_O"),
			(43, "OCT_V"),
			(44, "OCT_O"),
		]

		# --------------------------------------------------
		# Map .dat names -> names used in self.resultsdict
		#
		# These preserve your existing naming convention.
		# --------------------------------------------------

		result_mapping = {
			"AST_V":   "AST_V",
			"AST_O":   "AST_O",
			"COMA_H":  "COMA_H",
			"COMA_V":  "COMA_V",
			"TRE_O":   "TRE_O",
			"TRE_V":   "TRE_V",
			"SPH":     "SPH",
			"2AST_V":  "AST2_V",
			"2AST_O":  "AST2_O",
			"TET_V":   "QUAD_V",
			"TET_O":   "QUAD_O",
		}

		# --------------------------------------------------
		# Write file
		# --------------------------------------------------

		with open(filename, "w") as f:

			for index, dat_label in zernike_rows:

				value = 0.0
				flag = 0

				# The global Zernike tilts are fitted (unlike per-map tilts).
				if dat_label in ('TILT_H', 'TILT_V') and dat_label in self.resultsdict:
					value = float(self.resultsdict[dat_label])
					flag = 1

				# ------------------------------------------
				# First check for one of your explicitly
				# named Zernike coefficients.
				# ------------------------------------------

				if dat_label in result_mapping:

					result_label = result_mapping[dat_label]

					if result_label in self.resultsdict:
						value = float(self.resultsdict[result_label])
						flag = 1

				# ------------------------------------------
				# Otherwise check whether this mode was
				# stored generically as "OSA Index N".
				# ------------------------------------------

				else:

					osa_label = f"OSA Index {index}"

					if osa_label in self.resultsdict:
						value = float(self.resultsdict[osa_label])
						flag = 1

				f.write(
					f"{index:d} {flag:d} {dat_label} {value:.6f}\n"
				)

	def save_subref_dat(self, filename, reference_m2z=0.0):
		"""Write the absolute axial focus target in microns.

		M2Z is -x[1], where modeled defocus is zero, as in the offsets fitter.
		M2X/M2Y are zero and flagged unfitted because this class fits angles,
		not lateral translations. Tilts and their slopes are saved by
		save_results in JSON (degrees and degrees/m). reference_m2z is
		retained for call compatibility but does not affect this export.
		"""
		with open(filename, 'w') as f:
			f.write(f"0 1 M2Z {-float(self.results.x[1]) * 1e6:.6f}\n")
			f.write("1 0 M2X 0.000000\n")
			f.write("2 0 M2Y 0.000000\n")



class fit_beam_with_glob_tilt_M2tilts_leastsquares(_LeastSquaresUncertainties):
	"""Fit M2 tilts linear in the recorded M2.Z position.

	x[2:6] = (a_alph_x, b_alph_x, a_alph_y, b_alph_y), where
	del_alph = a_alph + b_alph * recorded_M2Z. M2.Z and its fitted
	offset are in meters; modeled defocus is recorded_M2Z + x[1], so
	the absolute axial focus target is -x[1]. a_alph is in degrees and
	b_alph in degrees/m.
	Zernike coefficients (including the separate global pointing tilts)
	are in microns. Lateral M2 translations are fixed at zero.
	"""

	def __init__(self,beam_class):

		print('Initializing the fit beam class ')


		mapcounter = len(beam_class.trunc_maps)
		print(f'There are {mapcounter} maps')
		print('The vector of model parameters will have the form:')
		parameter_names = [
			'source_amplitude', 'M2.Z_offset',
			'a_m2alph_x', 'b_m2alph_x', 'a_m2alph_y', 'b_m2alph_y',
			'TILT_H', 'TILT_V',
			'AST_O', 'AST_V', 'TRE_V',
		]
		for ind, name in enumerate(parameter_names):
			print(f'x[{ind}] = {name}')
		print('x[11:] = remaining fitted higher-order Zernikes')

		self.number_of_maps = mapcounter
		# x[8:] is mapped to c[3], c[5], c[6], c[9:]
		self.tilt_offset_end_index = 8  # retained for older methods
		self.tilt_offset_start_index = 6
		self.fit_vec_size = beam_class.zernike_polynomials.shape[0] + 2

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

		chi_squared = np.zeros(self.tmpbeamclass.trunc_maps['map0'].shape)

		source_amp = x[0]
		M2z_offset = x[1]

		a_m2alph_x = x[2]
		b_m2alph_x = x[3]

		a_m2alph_y = x[4]
		b_m2alph_y = x[5]

		tilt_counter = self.tilt_offset_start_index
		residual_maps = {}
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

			ctmp[1] = x[6]
			tilt_counter+=1
			ctmp[2] = x[7]
			tilt_counter+=1

			ctmp[3] = x[self.tilt_offset_end_index]
			ctmp[5] = x[self.tilt_offset_end_index+1]
			ctmp[6] = x[self.tilt_offset_end_index+2]
			ctmp[9:] = x[self.tilt_offset_end_index+3:]

			tmpalph_x = a_m2alph_x+(b_m2alph_x*self.tmpbeamclass.m2z_vals[i])
			tmpalph_y = a_m2alph_y+(b_m2alph_y*self.tmpbeamclass.m2z_vals[i])

			modelbeam = source_amp*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+M2z_offset,
							   del_x=0.0,del_y=0.0,del_alph_x=tmpalph_x,del_alph_y=tmpalph_y,
							   f=17.5,F=525.,D=50.)

			residual = self.tmpbeamclass.trunc_maps[i] - modelbeam
			residual_maps[i] = residual

			chi_squared+= residual**2

		tmpreturn = np.concatenate([residual.ravel() for residual in residual_maps.values()])
		self.fitting_counter+=1
		self.temp_cost = np.sqrt(np.mean(chi_squared))
		
		return tmpreturn

	def make_bounds(self,tilt_plusminus=None,zern_plusminus=None):
		"""Bound amplitude and optionally Zernikes; M2 tilts remain unbounded.

		tilt_plusminus limits global Zernike pointing tilts (x[6:8]),
		and zern_plusminus limits the other fitted Zernikes (x[8:]),
		both in microns. None leaves those coefficients unbounded.
		"""
		lower = np.full(len(self.x0), -np.inf)
		upper = np.full(len(self.x0), np.inf)
		lower[0] = 0.0
		for limit, indices in ((tilt_plusminus, slice(6, 8)),
							   (zern_plusminus, slice(8, None))):
			if limit is not None:
				if not np.isfinite(limit) or limit <= 0:
					raise ValueError('Zernike bounds must be finite and positive')
				lower[indices] = -limit
				upper[indices] = limit
		return lower, upper

	def run_fitter(self,tilt_plusminus=None,zern_plusminus=None,
					newx0 = None):

		if newx0 is not None:
			self.x0 = newx0

		tmpbound = self.make_bounds(tilt_plusminus,zern_plusminus)

		self.temp_cost = 1E6
		

		

		self.results = least_squares(
			self.chisquared,
			self.x0,
			#bounds=tmpbound,
			x_scale='jac'
		)
		ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
		ctmp[3] = self.results.x[8]
		ctmp[5] = self.results.x[9]
		ctmp[6] = self.results.x[10]
		ctmp[9:] = self.results.x[11:]
		x = self.results.x
		map_names = list(self.tmpbeamclass.trunc_maps)
		middle_map = map_names[len(map_names) // 2]
		z_middle = float(self.tmpbeamclass.m2z_vals[middle_map])

		# Match the offsets fitter: evaluate the middle map without global
		# pointing tilts or source amplitude, retaining fitted M2 aberrations.
		self.strehl_ratio = float(np.amax(
			self.tmpbeamclass.make_psf(
				c=ctmp,
				secondary_offset=z_middle+ x[1],
				del_x=0., del_y=0.,
				del_alph_x=x[2] + x[3] * z_middle,
				del_alph_y=x[4] + x[5] * z_middle,
				f=17.5, F=525., D=50.,
			)
		))


	def plot_fit_results(self,vmax_frac_of_source_flux = 0.2,resids_stretch=5,vmin_val = -50,
					 title=None,savefigname=None,showplot=False):

		#plt.figure(figsize=(15,8))
		fig = plt.figure(figsize=(3.0*self.number_of_maps + 1.2, 8))
		# plt.figure()
		subplotcounter=1
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			corners_tmp = np.rad2deg(enmap.corners(self.tmpbeamclass.trunc_maps[i].shape,self.tmpbeamclass.trunc_maps[i].wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(self.tmpbeamclass.trunc_maps[i],extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			post_stamp_size = 1.5*60.#1.5/60.
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		fit_a_m2alph_x = self.results.x[2]
		fit_b_m2alph_x = self.results.x[3]

		fit_a_m2alph_y = self.results.x[4]
		fit_b_m2alph_y = self.results.x[5]

		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

			ctmp[1] = self.results.x[6]
			tilt_counter+=1
			ctmp[2] = self.results.x[7]
			tilt_counter+=1


			ctmp[3] = self.results.x[8]
			ctmp[5] = self.results.x[9]
			ctmp[6] = self.results.x[10]
			ctmp[9:] = self.results.x[11:]

			tmpalph_x = fit_a_m2alph_x+(fit_b_m2alph_x*self.tmpbeamclass.m2z_vals[i])
			tmpalph_y = fit_a_m2alph_y+(fit_b_m2alph_y*self.tmpbeamclass.m2z_vals[i])

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,
				secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
				del_x=0.,del_y=0.,
				del_alph_x=tmpalph_x,del_alph_y=tmpalph_y)
			

			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(tmpmodelbeam,extent=imextent_tmp,origin='lower',vmin=vmin_val,vmax=vmax_frac_of_source_flux*self.results.x[0])
			plt.xticks([])
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])
			if subplotcounter == self.number_of_maps + 1:
				plt.ylabel('El Offset (arcsec)')
			subplotcounter+=1
		tilt_counter = self.tilt_offset_start_index
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

			ctmp[1] = self.results.x[6]
			tilt_counter+=1
			ctmp[2] = self.results.x[7]
			tilt_counter+=1
			

			ctmp[3] = self.results.x[8]
			ctmp[5] = self.results.x[9]
			ctmp[6] = self.results.x[10]
			ctmp[9:] = self.results.x[11:]

			tmpalph_x = fit_a_m2alph_x+(fit_b_m2alph_x*self.tmpbeamclass.m2z_vals[i])
			tmpalph_y = fit_a_m2alph_y+(fit_b_m2alph_y*self.tmpbeamclass.m2z_vals[i])

			tmpmodelbeam = self.results.x[0]*self.tmpbeamclass.make_psf(c=ctmp,
				secondary_offset=self.tmpbeamclass.m2z_vals[i]+self.results.x[1],
				del_x=0.,del_y=0.,
				del_alph_x=tmpalph_x,del_alph_y=tmpalph_y)
			

			corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(100*(self.tmpbeamclass.trunc_maps[i]-tmpmodelbeam)/self.results.x[0],extent=imextent_tmp,origin='lower',vmin=-resids_stretch,vmax=resids_stretch)
			plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
			plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
			if subplotcounter == 2*self.number_of_maps + 2:
				plt.xlabel('Az Offset (arcsec)')
			if (subplotcounter)%self.number_of_maps!=1:
				plt.yticks([])

			subplotcounter+=1

		#fig = plt.gcf()

		nrows = 3
		ncols = self.number_of_maps

		# Optional: leave some room on the right for colorbars
		#fig.subplots_adjust(right=0.88)

		fig.subplots_adjust(
			left=0.10,
			right=0.88,
			bottom=0.06,
			top=0.92,
			wspace=0.03,
			hspace=0.20
		)

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
				pos.x0 - 0.06,     # shift a little left of the first subplot
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
		self.datamodelresid_fname = savefigname


	def _fitted_zernikes(self, include_tilt=True):
		"""Reconstruct exactly the Zernike coefficient vector used in chisquared."""
		x = self.results.x
		c = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
		if include_tilt:
			c[1], c[2] = x[6], x[7]
		c[3] = x[8]
		c[5] = x[9]
		c[6] = x[10]
		c[9:] = x[11:]
		return c

	def save_results(self, savefilename):

		savefilename = os.fspath(savefilename)

		# Zernike labels indexed by their position in ctmp.
		# These follow the naming convention in your existing code.
		zernike_labels = {
			1:  'TILT_H',
			2:  'TILT_V',
			3:  'AST_O',
			5:  'AST_V',
			6:  'TRE_V',
			9:  'TRE_O',
			10: 'QUAD_O',
			11: 'AST2_O',
			12: 'SPH',
			13: 'AST2_V',
			14: 'QUAD_V',
		}

		results_dict = {}

		# --------------------------------------------------
		# General fit parameters
		# --------------------------------------------------

		x = self.results.x

		results_dict['source_amp'] = float(x[0])
		# Model defocus = recorded absolute M2.Z + fitted offset.
		results_dict['M2.Z_offset'] = float(x[1])
		results_dict['M2.Z_target'] = -float(x[1])
		# Evaluate fitted angles at the axial focus target (degrees).
		results_dict['M2.alphX_target'] = float(x[2] + x[3] * results_dict['M2.Z_target'])
		results_dict['M2.alphY_target'] = float(x[4] + x[5] * results_dict['M2.Z_target'])
		# Select the later middle map for an even count, in fitting order.
		map_names = list(self.tmpbeamclass.trunc_maps)
		middle_map = map_names[len(map_names) // 2]
		results_dict['M2.Z_middle'] = float(self.tmpbeamclass.m2z_vals[middle_map])
		results_dict['M2.Z_move'] = (results_dict['M2.Z_target']
			- results_dict['M2.Z_middle'])
		results_dict['strehl_ratio'] = float(self.strehl_ratio)

		# --------------------------------------------------
		# M2 tilt model (degrees, with slopes in degrees/m)
		# del_alph_x = a_m2alph_x + b_m2alph_x * recorded_M2.Z
		# del_alph_y = a_m2alph_y + b_m2alph_y * recorded_M2.Z
		# --------------------------------------------------

		results_dict['a_m2alph_x'] = float(x[2])
		results_dict['b_m2alph_x'] = float(x[3])
		results_dict['a_m2alph_y'] = float(x[4])
		results_dict['b_m2alph_y'] = float(x[5])

		# --------------------------------------------------
		# Global Zernike tilts
		# --------------------------------------------------

		results_dict['TILT_H'] = float(x[6])
		results_dict['TILT_V'] = float(x[7])

		# --------------------------------------------------
		# Per-map M2 tilts (degrees)
		# --------------------------------------------------

		for map_name in self.tmpbeamclass.trunc_maps:

			m2z = float(self.tmpbeamclass.m2z_vals[map_name])

			del_alph_x = x[2] + x[3] * m2z
			del_alph_y = x[4] + x[5] * m2z

			results_dict[f'M2.alphX_{map_name}'] = float(del_alph_x)
			results_dict[f'M2.alphY_{map_name}'] = float(del_alph_y)

		# --------------------------------------------------
		# Reconstruct Zernike coefficient vector
		# Identical indexing to chisquared()
		# --------------------------------------------------

		ctmp = np.zeros(
			self.tmpbeamclass.zernike_polynomials.shape[0]
		)

		ctmp[1] = x[6]
		ctmp[2] = x[7]

		ctmp[3] = x[8]
		ctmp[5] = x[9]
		ctmp[6] = x[10]

		ctmp[9:] = x[11:]

		# Only save coefficients actually included in fit.
		# Indices 7 and 8 (coma) are deliberately excluded.

		fitted_indices = [3, 5, 6] + list(range(9, len(ctmp)))

		for idx in fitted_indices:

			label = zernike_labels.get(
				idx,
				f'OSA Index {idx}'
			)

			results_dict[label] = float(ctmp[idx])

		# --------------------------------------------------
		# Print results
		# --------------------------------------------------

		for key, value in results_dict.items():
			print(key, value)

		# --------------------------------------------------
		# Save JSON
		# --------------------------------------------------

		with open(savefilename, "w") as f:
			json.dump(results_dict, f, indent=4)

		self.resultsdict = results_dict

		# --------------------------------------------------
		# Save results as a PNG table
		# --------------------------------------------------

		table_rows = []

		for label, value in results_dict.items():

			# Keep per-map angles in JSON; show target angles in the table.
			if label.startswith(('M2.alphX_', 'M2.alphY_')) and not label.endswith('_target'):
				continue

			if label == 'source_amp':
				unit = 'mJy/beam'
				display_value = value

			elif label in ('M2.Z_offset', 'M2.Z_target', 'M2.Z_middle', 'M2.Z_move'):
				unit = 'mm'
				display_value = value * 1e3

			elif label == 'strehl_ratio':
				unit = ''
				display_value = value

			elif label.startswith(('M2.alphX_', 'M2.alphY_')):
				unit = 'degrees'
				display_value = value

			elif label in ('a_m2alph_x', 'a_m2alph_y'):
				unit = 'degrees'
				display_value = value

			elif label in ('b_m2alph_x', 'b_m2alph_y'):
				unit = 'degrees/mm'
				display_value = value * 1e-3

			else:
				# Zernike coefficients, including global tilts
				unit = 'microns'
				display_value = value

			table_rows.append([
				label,
				f"{display_value:.4f}",
				unit
			])

		# --------------------------------------------------
		# Make table figure
		# --------------------------------------------------

		fig_height = max(
			2.0,
			0.35 * (len(table_rows) + 1)
		)

		fig, ax = plt.subplots(
			figsize=(6, fig_height)
		)

		ax.axis("off")

		table = ax.table(
			cellText=table_rows,
			colLabels=["LABEL", "VALUE", "UNITS"],
			cellLoc="center",
			colLoc="center",
			loc="center",
		)

		table.auto_set_font_size(False)
		table.set_fontsize(11)
		table.scale(1.0, 1.4)

		for (row, col), cell in table.get_celld().items():

			cell.set_edgecolor("black")
			cell.set_linewidth(1.0)

			if row == 0:
				cell.set_facecolor("0.9")
			else:
				cell.set_facecolor("0.96")

		# --------------------------------------------------
		# Save PNG
		# --------------------------------------------------

		pngfilename = (
			os.path.splitext(savefilename)[0]
			+ ".png"
		)

		plt.savefig(
			pngfilename,
			dpi=200,
			bbox_inches="tight"
		)

		plt.close(fig)

		self.resulttable_fname = pngfilename

	def _uncertainty_model(self, x):
		"""Parameter names and result gradients for this fitted model."""
		zernike_labels = {
			3: 'AST_O', 5: 'AST_V', 6: 'TRE_V', 9: 'TRE_O',
			10: 'QUAD_O', 11: 'AST2_O', 12: 'SPH',
			13: 'AST2_V', 14: 'QUAD_V',
		}
		indices = [3, 5, 6] + list(range(9, self.tmpbeamclass.zernike_polynomials.shape[0]))
		names = ['source_amp', 'M2.Z_offset',
			'a_m2alph_x', 'b_m2alph_x', 'a_m2alph_y', 'b_m2alph_y',
			'TILT_H', 'TILT_V']
		names += [zernike_labels.get(idx, f'OSA Index {idx}') for idx in indices]
		basis = np.eye(x.size)
		map_names = list(self.tmpbeamclass.trunc_maps)
		z_middle = float(self.tmpbeamclass.m2z_vals[map_names[len(map_names) // 2]])
		gradients = {
			'source_amp': basis[0],
			'M2.Z_offset': basis[1],
			'M2.Z_target': -basis[1],
			'M2.alphX_target': basis[2] - x[1] * basis[3] - x[3] * basis[1],
			'M2.alphY_target': basis[4] - x[1] * basis[5] - x[5] * basis[1],
			'M2.Z_middle': np.zeros(x.size),
			'M2.Z_move': -basis[1],
		}

		def strehl_at(parameters):
			c = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])
			c[indices] = parameters[8:]
			return float(np.amax(self.tmpbeamclass.make_psf(
				c=c, secondary_offset=z_middle + parameters[1],
				del_x=0., del_y=0.,
				del_alph_x=parameters[2] + parameters[3] * z_middle,
				del_alph_y=parameters[4] + parameters[5] * z_middle,
				f=17.5, F=525., D=50.,
			)))

		strehl_gradient = np.zeros(x.size)
		for idx in [1, 2, 3, 4, 5] + list(range(8, x.size)):
			step = np.cbrt(np.finfo(float).eps) * max(abs(x[idx]), 1e-3 if idx == 1 else 1.)
			delta = step * basis[idx]
			strehl_gradient[idx] = (strehl_at(x + delta) - strehl_at(x - delta)) / (2 * step)
		gradients['strehl_ratio'] = strehl_gradient
		for idx in range(2, 8):
			gradients[names[idx]] = basis[idx]
		for map_name in map_names:
			z = float(self.tmpbeamclass.m2z_vals[map_name])
			gradients[f'M2.alphX_{map_name}'] = basis[2] + z * basis[3]
			gradients[f'M2.alphY_{map_name}'] = basis[4] + z * basis[5]
		for idx in range(8, x.size):
			gradients[names[idx]] = basis[idx]
		return names, gradients

	def surface_plot(self, savefilename, vmin=-400, vmax=400):
		"""Plot the common fitted surface terms (excluding global pointing tilt).

		M2 tilts produce additional OPD in make_psf and are not
		included in this Zernike-only surface plot.
		"""
		cfit = self._fitted_zernikes(include_tilt=False)
		OPD = np.tensordot(cfit, self.tmpbeamclass.zernike_polynomials,
			axes=([0], [0]))
		plt.figure()
		plt.imshow(OPD, extent=[-self.tmpbeamclass.L/2., self.tmpbeamclass.L/2.,
			-self.tmpbeamclass.L/2., self.tmpbeamclass.L/2.], vmin=vmin, vmax=vmax)
		plt.xlim(-30, 30)
		plt.ylim(-30, 30)
		plt.xlabel('x (meters)')
		plt.ylabel('y (meters)')
		plt.colorbar().set_label('OPD (microns)', rotation=-90, labelpad=20)
		plt.savefig(savefilename, bbox_inches='tight')
		plt.close()
		self.surfaceplot_fname = savefilename

	def combine_result_pngs(self,
							outfile="combined_results.png",
							dpi=200):

		main_png = self.datamodelresid_fname
		table_png = self.resulttable_fname
		opd_png = self.surfaceplot_fname
		"""
		Combine three PNGs into one summary figure.

		Layout:
		- top row: main_png spans full width
		- bottom left: opd_png
		- bottom right: table_png
		"""

		img_main = mpimg.imread(main_png)
		img_table = mpimg.imread(table_png)
		img_opd = mpimg.imread(opd_png)

		fig = plt.figure(figsize=(12, 14), constrained_layout=True)
		gs = GridSpec(
			2, 2,
			figure=fig,
			height_ratios=[1.2, 1.0],
			width_ratios=[1.0, 1.1]
		)

		ax_main = fig.add_subplot(gs[0, :])
		ax_opd = fig.add_subplot(gs[1, 0])
		ax_table = fig.add_subplot(gs[1, 1])

		ax_main.imshow(img_main)
		ax_opd.imshow(img_opd)
		ax_table.imshow(img_table)

		ax_main.axis("off")
		ax_opd.axis("off")
		ax_table.axis("off")

		plt.savefig(outfile, dpi=dpi, bbox_inches="tight")
		plt.close(fig)

		print(f"Saved combined figure to {outfile}")

	def save_zernike_dat(self, filename):
		"""
		Save self.resultsdict in the same format as the LMT OOF
		zernike .dat file.

		Supports both the named low-order Zernike parameters and
		higher-order parameters stored as "OSA Index N".
		"""

		# --------------------------------------------------
		# Full ordered list used by the LMT .dat format
		# --------------------------------------------------

		zernike_rows = [
			(0,  "BIAS"),
			(1,  "TILT_H"),
			(2,  "TILT_V"),
			(3,  "FOCUS"),
			(4,  "AST_V"),
			(5,  "AST_O"),
			(6,  "COMA_H"),
			(7,  "COMA_V"),
			(8,  "TRE_O"),
			(9,  "TRE_V"),
			(10, "SPH"),
			(11, "2AST_V"),
			(12, "2AST_O"),
			(13, "TET_V"),
			(14, "TET_O"),
			(15, "2COMA_H"),
			(16, "2COMA_V"),
			(17, "2TRE_O"),
			(18, "2TRE_V"),
			(19, "PEN_O"),
			(20, "PEN_V"),
			(21, "2SPH"),
			(22, "3AST_V"),
			(23, "3AST_O"),
			(24, "2TET_V"),
			(25, "2TET_O"),
			(26, "HEX_V"),
			(27, "HEX_O"),
			(28, "3COMA_H"),
			(29, "3COMA_V"),
			(30, "3TRE_O"),
			(31, "3TRE_V"),
			(32, "2PEN_O"),
			(33, "2PEN_V"),
			(34, "HEPT_O"),
			(35, "HEPT_V"),
			(36, "3SPH"),
			(37, "4AST_V"),
			(38, "4AST_O"),
			(39, "3TET_V"),
			(40, "3TET_O"),
			(41, "2HEX_V"),
			(42, "2HEX_O"),
			(43, "OCT_V"),
			(44, "OCT_O"),
		]

		# --------------------------------------------------
		# Map .dat names -> names used in self.resultsdict
		#
		# These preserve your existing naming convention.
		# --------------------------------------------------

		result_mapping = {
			"AST_V":   "AST_V",
			"AST_O":   "AST_O",
			"COMA_H":  "COMA_H",
			"COMA_V":  "COMA_V",
			"TRE_O":   "TRE_O",
			"TRE_V":   "TRE_V",
			"SPH":     "SPH",
			"2AST_V":  "AST2_V",
			"2AST_O":  "AST2_O",
			"TET_V":   "QUAD_V",
			"TET_O":   "QUAD_O",
		}

		# --------------------------------------------------
		# Write file
		# --------------------------------------------------

		with open(filename, "w") as f:

			for index, dat_label in zernike_rows:

				value = 0.0
				flag = 0

				# The global Zernike tilts are fitted (unlike per-map tilts).
				if dat_label in ('TILT_H', 'TILT_V') and dat_label in self.resultsdict:
					value = float(self.resultsdict[dat_label])
					flag = 1

				# ------------------------------------------
				# First check for one of your explicitly
				# named Zernike coefficients.
				# ------------------------------------------

				if dat_label in result_mapping:

					result_label = result_mapping[dat_label]

					if result_label in self.resultsdict:
						value = float(self.resultsdict[result_label])
						flag = 1

				# ------------------------------------------
				# Otherwise check whether this mode was
				# stored generically as "OSA Index N".
				# ------------------------------------------

				else:

					osa_label = f"OSA Index {index}"

					if osa_label in self.resultsdict:
						value = float(self.resultsdict[osa_label])
						flag = 1

				f.write(
					f"{index:d} {flag:d} {dat_label} {value:.6f}\n"
				)

	def save_subref_dat(self, filename, reference_m2z=0.0):
		"""Write the absolute axial focus target in microns.

		M2Z is -x[1], where modeled defocus is zero, as in the offsets fitter.
		M2X/M2Y are zero and flagged unfitted because this class fits angles,
		not lateral translations. Tilts and their slopes are saved by
		save_results in JSON (degrees and degrees/m). reference_m2z is
		retained for call compatibility but does not affect this export.
		"""
		with open(filename, 'w') as f:
			f.write(f"0 1 M2Z {-float(self.results.x[1]) * 1e6:.6f}\n")
			f.write("1 0 M2X 0.000000\n")
			f.write("2 0 M2Y 0.000000\n")
