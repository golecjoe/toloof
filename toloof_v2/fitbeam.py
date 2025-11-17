from scipy.optimize import curve_fit,minimize
import numpy as np
import matplotlib.pyplot as plt

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
		print(f'x[{vectposoffsetcounter}] = COMA_V')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = COMA_H')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = TRE_O')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = QUAD_O')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = AST2_O')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = SPH')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = AST2_V')
		vectposoffsetcounter+=1
		print(f'x[{vectposoffsetcounter}] = QUAD_V')

		self.number_of_maps = mapcounter
		self.fit_vec_size = vectposoffsetcounter+1

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
		return np.sqrt(chi_squared)


def plot_fit_results(beamclass,fitclass,results):

	plt.figure()
	subplotcounter=1
	for count,i in enumerate(beamclass.trunc_maps):
		plt.subplot(3,fitclass.number_of_maps,subplotcounter)
		plt.imshow(beamclass.trunc_maps[i])
		plt.xticks([])
		if (subplotcounter-1)%fitclass.number_of_maps!=0:
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
		subplotcounter+=1
		plt.imshow(tmpmodelbeam)
		plt.xticks([])
		if (subplotcounter-1)%fitclass.number_of_maps!=0:
			plt.yticks([])
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
		subplotcounter+=1
		plt.imshow(beamclass.trunc_maps[i]-tmpmodelbeam)
		if (subplotcounter-1)%fitclass.number_of_maps!=0:
			plt.yticks([])
	plt.show()



















