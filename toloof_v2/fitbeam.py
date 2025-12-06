from scipy.optimize import curve_fit,minimize
import numpy as np
import matplotlib.pyplot as plt
from pixell import enmap, enplot, reproject, utils, curvedsky,wcsutils
import json
from scipy.optimize import minimize



from beamclass import Beam


# class fit_beam_with_pointing_offsets:

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
# 			print(f'x[{vectposoffsetcounter}] = TILT_Y_map{i}')
# 			vectposoffsetcounter+=1
# 			print(f'x[{vectposoffsetcounter}] = TILT_X_map{i}')
# 			vectposoffsetcounter+=1
# 		self.tilt_offset_end_index = vectposoffsetcounter
# 		print(f'x[{vectposoffsetcounter}] = AST_O')
# 		vectposoffsetcounter+=1
# 		print(f'x[{vectposoffsetcounter}] = AST_V')
# 		vectposoffsetcounter+=1
# 		print(f'x[{vectposoffsetcounter}] = TRE_V')
# 		vectposoffsetcounter+=1
# 		print('...')
# 		# print(f'x[{vectposoffsetcounter}] = COMA_V')
# 		# vectposoffsetcounter+=1
# 		# print(f'x[{vectposoffsetcounter}] = COMA_H')
# 		# vectposoffsetcounter+=1
# 		# print(f'x[{vectposoffsetcounter}] = TRE_O')
# 		# vectposoffsetcounter+=1
# 		# print(f'x[{vectposoffsetcounter}] = QUAD_O')
# 		# vectposoffsetcounter+=1
# 		# print(f'x[{vectposoffsetcounter}] = AST2_O')
# 		# vectposoffsetcounter+=1
# 		# print(f'x[{vectposoffsetcounter}] = SPH')
# 		# vectposoffsetcounter+=1
# 		# print(f'x[{vectposoffsetcounter}] = AST2_V')
# 		# vectposoffsetcounter+=1
# 		# print(f'x[{vectposoffsetcounter}] = QUAD_V')

# 		self.number_of_maps = mapcounter
# 		self.fit_vec_size = 2+(2*mapcounter)+beam_class.zernike_polynomials.shape[0]-4 #vectposoffsetcounter+1

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

# 		chi_squared = 0

# 		source_amp = x[0]
# 		M2z_offset = x[1]

# 		tilt_counter = self.tilt_offset_start_index
# 		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
# 			ctmp = np.zeros(self.tmpbeamclass.zernike_polynomials.shape[0])

# 			ctmp[1] = x[tilt_counter]
# 			tilt_counter+=1
# 			ctmp[2] = x[tilt_counter]
# 			tilt_counter+=1

# 			ctmp[3] = x[self.tilt_offset_end_index]
# 			ctmp[5:] = x[self.tilt_offset_end_index+1:]

# 			modelbeam = source_amp*self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=self.tmpbeamclass.m2z_vals[i]+M2z_offset,
# 							   del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
# 							   f=17.5,F=525.,D=50.)


# 			residual = self.tmpbeamclass.trunc_maps[i] - modelbeam

# 			chi_squared+= np.mean(residual**2)
# 		self.fitting_counter+=1
# 		self.temp_cost = chi_squared
# 		self.strehl_ratio = np.amax(self.tmpbeamclass.make_psf(c=ctmp,secondary_offset=0,
# 							   del_x=0.,del_y=0.,del_alph_x=0.,del_alph_y=0.,
# 							   f=17.5,F=525.,D=50.))
# 		return np.sqrt(chi_squared)


# def plot_fit_results(beamclass,fitclass,results,vmax_frac_of_source_flux = 0.2,resids_stretch=5,
# 					 title=None,savefigname=None,showplot=False):

# 	plt.figure(figsize=(15,8))
# 	# plt.figure()
# 	subplotcounter=1
# 	for count,i in enumerate(beamclass.trunc_maps):
# 		plt.subplot(3,fitclass.number_of_maps,subplotcounter)
# 		corners_tmp = np.rad2deg(enmap.corners(beamclass.trunc_maps[i].shape,beamclass.trunc_maps[i].wcs))
# 		imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
# 		plt.imshow(beamclass.trunc_maps[i],extent=imextent_tmp,origin='lower',vmin=-50,vmax=vmax_frac_of_source_flux*results.x[0])
# 		plt.xticks([])
# 		post_stamp_size = 1.5*60.#1.5/60.
# 		plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
# 		plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
# 		if (subplotcounter)%fitclass.number_of_maps!=1:
# 			plt.yticks([])
# 		subplotcounter+=1
# 	tilt_counter = 0
# 	for count,i in enumerate(beamclass.trunc_maps):
# 		plt.subplot(3,fitclass.number_of_maps,subplotcounter)
# 		ctmp = np.zeros(beamclass.zernike_polynomials.shape[0])
# 		ctmp[1] = results.x[fitclass.tilt_offset_start_index+tilt_counter]
# 		tilt_counter+=1
# 		ctmp[2] = results.x[fitclass.tilt_offset_start_index+tilt_counter]
# 		tilt_counter+=1

# 		ctmp[3] = results.x[fitclass.tilt_offset_end_index]
# 		ctmp[5:] = results.x[fitclass.tilt_offset_end_index+1:]

# 		tmpmodelbeam = results.x[0]*beamclass.make_psf(c=ctmp,secondary_offset=beamclass.m2z_vals[i]+results.x[1])
		
# 		corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
# 		imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
# 		plt.imshow(tmpmodelbeam,extent=imextent_tmp,origin='lower',vmin=-50,vmax=vmax_frac_of_source_flux*results.x[0])
# 		plt.xticks([])
# 		plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
# 		plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
# 		if (subplotcounter)%fitclass.number_of_maps!=1:
# 			plt.yticks([])
# 		subplotcounter+=1
# 	tilt_counter = 0
# 	for count,i in enumerate(beamclass.trunc_maps):
# 		plt.subplot(3,fitclass.number_of_maps,subplotcounter)
# 		ctmp = np.zeros(beamclass.zernike_polynomials.shape[0])
# 		ctmp[1] = results.x[fitclass.tilt_offset_start_index+tilt_counter]
# 		tilt_counter+=1
# 		ctmp[2] = results.x[fitclass.tilt_offset_start_index+tilt_counter]
# 		tilt_counter+=1

# 		ctmp[3] = results.x[fitclass.tilt_offset_end_index]
# 		ctmp[5:] = results.x[fitclass.tilt_offset_end_index+1:]

# 		tmpmodelbeam = results.x[0]*beamclass.make_psf(c=ctmp,secondary_offset=beamclass.m2z_vals[i]+results.x[1])
		
# 		corners_tmp = np.rad2deg(enmap.corners(tmpmodelbeam.shape,tmpmodelbeam.wcs))
# 		imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
# 		plt.imshow(100*(beamclass.trunc_maps[i]-tmpmodelbeam)/results.x[0],extent=imextent_tmp,origin='lower',vmin=-resids_stretch,vmax=resids_stretch)
# 		plt.xlim(post_stamp_size/2.,-post_stamp_size/2.)
# 		plt.ylim(-post_stamp_size/2.,post_stamp_size/2.)
# 		if (subplotcounter)%fitclass.number_of_maps!=1:
# 			plt.yticks([])

# 		subplotcounter+=1

# 	fig = plt.gcf()

# 	nrows = 3
# 	ncols = fitclass.number_of_maps

# 	# Optional: leave some room on the right for colorbars
# 	fig.subplots_adjust(right=0.88)

# 	cbar_labels = ['mJy/beam','mJy/beam','% Source Flux']
# 	row_labels = ["data", "model", "residuals"]

# 	for row in range(nrows):



# 		# index of last subplot in this row (1-based for plt.subplot)
# 		last_idx = (row + 1) * ncols
# 		ax = plt.subplot(nrows, ncols, last_idx)

# 		# Get position of this last axes
# 		pos = ax.get_position()

# 		# Make a new axes for the colorbar, just to the right of it
# 		cax = fig.add_axes([
# 			pos.x1 + 0.005,   # a bit to the right of the last axes
# 			pos.y0,           # same bottom
# 			0.01,             # narrow width
# 			pos.height        # same height
# 		])

# 		# Get the image plotted in this axes (the first imshow)
# 		im = ax.images[0]

# 		# Make a colorbar for that row
# 		cb = fig.colorbar(im, cax=cax)
# 		cb.set_label(cbar_labels[row],rotation=-90,labelpad=20)

# 		first_idx = ((row) * ncols)+1
# 		ax = plt.subplot(nrows, ncols, first_idx)
# 		pos = ax.get_position()


# 		# y-coordinate of the center of this row
# 		y = pos.y0 + pos.height/2

# 		# Put text slightly left of this row
# 		fig.text(
# 			pos.x0 - 0.03,     # shift a little left of the first subplot
# 			y,
# 			row_labels[row],
# 			va='center', ha='right', fontsize=14
# 		)
# 	plt.suptitle(title)
# 	if savefigname is not None:
# 		plt.savefig(savefigname,bbox_inches='tight')

# 	if showplot:
# 		plt.show()
# 	else:
# 		plt.close()


# def save_results(results,fitclass,savefilename):
# 	zernike_labels = ['AST_O','AST_V','TRE_V','COMA_V','COMA_H','TRE_O','QUAD_O','AST2_O','SPH','AST2_V','QUAD_V']
# 	results_dict = {}
# 	results_dict['source_amp'] = results.x[0]
# 	results_dict['M2.Z_offset'] = results.x[1]
# 	results_dict['strehl_ratio'] = fitclass.strehl_ratio
# 	map_counter = 0
# 	tmpind = fitclass.tilt_offset_start_index
# 	while tmpind<fitclass.tilt_offset_end_index:
# 		results_dict[f'Tilt_Y_map{map_counter}'] = results.x[tmpind]
# 		tmpind+=1
# 		results_dict[f'Tilt_X_map{map_counter}'] = results.x[tmpind]
# 		tmpind+=1
# 		map_counter+=1
# 	for i,val in enumerate(results.x[fitclass.tilt_offset_end_index:]):
# 		if i < len(zernike_labels):
# 			zernlabel = zernike_labels[i]
# 			# zernlabel = f'Noll {i}'
# 		else:
# 			zernlabel = f'OSA Index {i+4}'
# 			# zernlabel = zernike_labels[i]
# 		results_dict[zernlabel] = val*1E6*(np.mean(fitclass.tmpbeamclass.wavelengths)/(2.*np.pi*np.sqrt(2)))
# 	for i in results_dict:
# 		print(i+' ', results_dict[i])
# 	for key, value in results_dict.items():
# 			if isinstance(value, np.ndarray):
# 				results_dict[key] = value.tolist()
# 	with open(savefilename, "w") as f:
# 		json.dump(results_dict, f, indent=4)




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

	def plot_fit_results(self,vmax_frac_of_source_flux = 0.2,resids_stretch=5,
					 title=None,savefigname=None,showplot=False):

		plt.figure(figsize=(15,8))
		# plt.figure()
		subplotcounter=1
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			corners_tmp = np.rad2deg(enmap.corners(self.tmpbeamclass.trunc_maps[i].shape,self.tmpbeamclass.trunc_maps[i].wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(self.tmpbeamclass.trunc_maps[i],extent=imextent_tmp,origin='lower',vmin=-50,vmax=vmax_frac_of_source_flux*self.results.x[0])
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
			plt.imshow(tmpmodelbeam,extent=imextent_tmp,origin='lower',vmin=-50,vmax=vmax_frac_of_source_flux*self.results.x[0])
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


	def plot_fit_results(self,vmax_frac_of_source_flux = 0.2,resids_stretch=5,
					 title=None,savefigname=None,showplot=False):

		plt.figure(figsize=(15,8))
		# plt.figure()
		subplotcounter=1
		for count,i in enumerate(self.tmpbeamclass.trunc_maps):
			plt.subplot(3,self.number_of_maps,subplotcounter)
			corners_tmp = np.rad2deg(enmap.corners(self.tmpbeamclass.trunc_maps[i].shape,self.tmpbeamclass.trunc_maps[i].wcs))
			imextent_tmp = 3600.*np.array([corners_tmp[0,1],corners_tmp[1,1],corners_tmp[0,0],corners_tmp[1,0]])
			plt.imshow(self.tmpbeamclass.trunc_maps[i],extent=imextent_tmp,origin='lower',vmin=-50,vmax=vmax_frac_of_source_flux*self.results.x[0])
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
			plt.imshow(tmpmodelbeam,extent=imextent_tmp,origin='lower',vmin=-50,vmax=vmax_frac_of_source_flux*self.results.x[0])
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















