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

class fit_beam_with_pointing_tilt_offsets_leastsquares:

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
		self.datamodelresid_fname = savefigname


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
		self.resultsdict = results_dict

		# --------------------------------------------------
		# Save results as a PNG table
		# --------------------------------------------------

		units = [
			'mJy/beam',
			'mm',
			'',
			'microns',
			'microns',
			'microns',
			'microns',
			'microns',
			'microns',
			'microns',
			'microns',
			'microns',
			'microns',
			'microns',
			'microns',
			'microns',
			'microns',
			'microns',
			'microns',
			'microns'
		]

		table_rows = []

		for (label, value), unit in zip(results_dict.items(), units):

			# if label=='M2.Z':
			# 	value = value*1E3
			# 	value_str = f"{value:.6f}"

			if isinstance(value, (int, float, np.integer, np.floating)) and label!='M2.Z_offset':
				value_str = f"{value:.2f}"
			elif isinstance(value, (int, float, np.integer, np.floating)) and label=='M2.Z_offset':
				value = value*1E3
				value_str = f"{value:.6f}"
			else:
				value_str = str(value)


			table_rows.append([
				label,
				value_str,
				unit
			])

		# Scale figure height with number of rows
		fig_height = max(2.0, 0.35 * (len(table_rows) + 1))

		fig, ax = plt.subplots(figsize=(6, fig_height))
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

		# Use same filename as JSON, but replace extension with .png
		pngfilename = os.path.splitext(savefilename)[0] + ".png"

		plt.savefig(
			pngfilename,
			dpi=200,
			bbox_inches="tight"
		)

		plt.close(fig)
		self.resulttable_fname = pngfilename

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
		"""

		# Full ordered list used by the .dat format
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

		# Map .dat names -> names used in self.resultsdict
		result_mapping = {
			"AST_V":  "AST_V",
			"AST_O":  "AST_O",
			"COMA_H": "COMA_H",
			"COMA_V": "COMA_V",
			"TRE_O":  "TRE_O",
			"TRE_V":  "TRE_V",
			"SPH":     "SPH",
			"2AST_V": "AST2_V",
			"2AST_O": "AST2_O",
			"TET_V":  "QUAD_V",
			"TET_O":  "QUAD_O",
		}

		with open(filename, "w") as f:

			for index, label in zernike_rows:

				if label in result_mapping:
					result_label = result_mapping[label]

					value = float(self.resultsdict[result_label])
					flag = 1

				else:
					value = 0.0
					flag = 0

				f.write(
					f"{index:d} {flag:d} {label} {value:.6f}\n"
				)

	def save_subref_dat(self, filename):
		"""
		Save the M2 position offsets in the same format as the LMT OOF
		subref.dat file.

		Assumes self.resultsdict['M2.Z_offset'] is in meters.
		Writes M2Z in microns.
		"""

		# Convert meters -> microns
		m2z_microns = float(self.resultsdict["M2.Z_offset"]) * 1e6

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

