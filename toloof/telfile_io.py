import netCDF4 as nc
import numpy as np

def get_M1zernike_from_tel(path2tel):
	tmpnc = nc.Dataset(path2tel)
	zernikes_array = tmpnc['Header.M1.ZernikeC'][:][:]

	zernikes_dict = {}

	zernike_labels = np.array(['AST_V','AST_O','COMA_H','COMA_V','TRE_O','TRE_V','SPH',
							   'QUAD_V','QUAD_O','AST2_O','AST2_V'])
	zernike_values = np.array([zernikes_array[0],zernikes_array[1],zernikes_array[2],zernikes_array[3],zernikes_array[4],zernikes_array[5],zernikes_array[6],
							   zernikes_array[9],zernikes_array[10],zernikes_array[8],zernikes_array[7]])
	zernikes_dict['labels']=zernike_labels
	zernikes_dict['values']=zernike_values
	# zernikes_dict['AST_V'] = zernikes_array[0]
	# zernikes_dict['AST_O'] = zernikes_array[1]
	# zernikes_dict['COMA_H'] = zernikes_array[2]
	# zernikes_dict['COMA_V'] = zernikes_array[3]
	# zernikes_dict['TRE_O'] = zernikes_array[4]
	# zernikes_dict['TRE_V'] = zernikes_array[5]
	# zernikes_dict['SPH'] = zernikes_array[6]
	# zernikes_dict['AST2_V'] = zernikes_array[7]
	# zernikes_dict['AST2_O'] = zernikes_array[8]
	# zernikes_dict['QUAD_V'] = zernikes_array[9]
	# zernikes_dict['QUAD_O'] = zernikes_array[10]
	# zernikes_dict['COMA2_H'] = zernikes_array[11]
	# zernikes_dict['COMA2_V'] = zernikes_array[12]
	# zernikes_dict['TRE2_O'] = zernikes_array[13]
	# zernikes_dict['TRE2_V'] = zernikes_array[14]
	# zernikes_dict['PEN_O'] = zernikes_array[15]
	# zernikes_dict['PEN_V'] = zernikes_array[16]
	# zernikes_dict['SPH2'] = zernikes_array[17]

	tmpnc.close()

	return zernikes_dict

def get_M2z_from_tel(path2tel):
	tmpnc = nc.Dataset(path2tel)
	m2z = float(tmpnc['Header.M2.ZReq'][:].data)
	tmpnc.close()

	return m2z
