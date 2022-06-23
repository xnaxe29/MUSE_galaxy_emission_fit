import scipy
import scipy as sp
from scipy import signal
import numpy
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit
c = 299792.458 # Speed in Light in Km/s
c_kms = 299792.458 # Speed in Light in Km/s
k = 1.38064852e-23 #Boltzmann's constant in m^2 kg s^(-2) K^(-1)


def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")



##################################LOG_TO_REAL_ERROR##################################


def log_to_real_err(log_array, log_error):
	linear_array = 10**(log_array)
	linear_err = np.abs(linear_array * (np.log(10) * log_error))
	return linear_err


##################################LOG_TO_REAL_ERROR##################################

##################################REAL_TO_LOG_ERROR##################################


def real_to_log_err(linear_array, linear_error):
	log_array = np.log10(linear_array)
	log_err = np.abs(linear_error/(linear_array*(np.log(10))))
	return log_err


##################################REAL_TO_LOG_ERROR##################################

###############################################################################

def _wave_convert(lam):
    """
    Convert between vacuum and air wavelengths using
    equation (1) of Ciddor 1996, Applied Optics 35, 1566
        http://doi.org/10.1364/AO.35.001566

    :param lam - Wavelength in Angstroms
    :return: conversion factor

    """
    lam = np.asarray(lam)
    sigma2 = (1e4/lam)**2
    fact = 1 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)

    return fact

###############################################################################

def vac_to_air(lam_vac):
    """
    Convert vacuum to air wavelengths

    :param lam_vac - Wavelength in Angstroms
    :return: lam_air - Wavelength in Angstroms

    """
    return lam_vac/_wave_convert(lam_vac)

###############################################################################

def air_to_vac(lam_air):
    """
    Convert air to vacuum wavelengths

    :param lam_air - Wavelength in Angstroms
    :return: lam_vac - Wavelength in Angstroms

    """
    return lam_air*_wave_convert(lam_air)

###############################################################################


######################FIND_THE_NEAREST_VALUE##################################
def find_nearest(array,value):
	idx = (np.abs(array-value)).argmin()
	return array[idx]

def find_nearest_idx(array,value):
	idx = (np.abs(array-value)).argmin()
	return idx
######################FIND_THE_NEAREST_VALUE##################################

##################################GET_DATA_FROM_FILE##################################

def refine_obs_data_using_scipy(wave_orig, data_real, err_real):
	wave_rebinned = np.logspace(np.log10(wave_orig.min()), np.log10(wave_orig.max()), num=len(wave_orig), endpoint=True, base=10)
	f_flux = interpolate.interp1d(wave_orig, data_real, axis=0, fill_value="extrapolate", kind='cubic')
	f_flux_err = interpolate.interp1d(wave_orig, err_real, axis=0, fill_value="extrapolate", kind='cubic')
	flux_rebinned = f_flux(wave_rebinned)
	flux_err_rebinned = f_flux_err(wave_rebinned)
	frac = wave_rebinned[1]/wave_rebinned[0]    # Constant lambda fraction per pixel
	dlam_gal = (frac - 1)*wave_rebinned            # Size of every pixel in Angstrom
	wdisp = np.ones([len(wave_rebinned)])          # Intrinsic dispersion of every pixel, in pixels units
	fwhm_gal_init = 2.355*wdisp*dlam_gal              # Resolution FWHM of every pixel, in Angstroms.
	fwhm_gal = np.nanmean(fwhm_gal_init)            # Keeping it as mean of fwhm_gal as we need a specific number.
	velscale = c*np.log(wave_rebinned[1]/wave_rebinned[0])
	return (wave_rebinned, flux_rebinned, flux_err_rebinned, fwhm_gal_init, velscale)

def get_data_from_file(file1, file_type, require_air_to_vaccum, extra_redshift):
	if file_type=='SDSS':
		hdu = fits.open(file1)
		t = hdu[1].data
		wave_orig = 10**(t['loglam'])
		flux_orig = t['flux']
		flux_err_orig = np.full_like(flux_orig, 0.01635)
		wave_ref, flux_ref, flux_err_ref, fwhm_gal_init, velscale = refine_obs_data_using_scipy(wave_orig, flux_orig, flux_err_orig)
	elif file_type=='other':
		wave_orig, flux_orig, flux_err_orig = np.loadtxt(file1, unpack=True, comments='#')
		wave_ref, flux_ref, flux_err_ref, fwhm_gal_init, velscale = refine_obs_data_using_scipy(wave_orig, flux_orig, flux_err_orig)
	elif file_type=='direct':
		wave_orig, flux_orig, flux_err_orig = file1
		wave_ref, flux_ref, flux_err_ref, fwhm_gal_init, velscale = refine_obs_data_using_scipy(wave_orig, flux_orig, flux_err_orig)
	if (require_air_to_vaccum):
		#Air to Vacuum conversion
		wave_ref *= np.median(vac_to_air(wave_ref)/wave_ref)
	FWHM_gal = np.nanmean(fwhm_gal_init)
	#print (velscale, FWHM_gal)
	#mask = (wave_ref > 3540.) & (wave_ref < 7409.)
	flux = flux_ref
	galaxy = flux   # Normalize spectrum to avoid numerical issues
	wave = wave_ref
	noise = flux_err_ref
	wave = wave / (1.+float(extra_redshift))
	FWHM_gal = FWHM_gal / (1.+float(extra_redshift))
	return (wave, galaxy, noise, FWHM_gal, velscale)

##################################GET_DATA_FROM_FILE##################################



##################################GET_PARAMS_AND_BOUNDS##################################

def get_params(popt, number_of_narrow_components, number_of_wide_components, center_list, amp_length):
	amplitude_len = amp_length
	center_len = amplitude_len+(number_of_narrow_components+number_of_wide_components)
	sigma_len = center_len+(number_of_narrow_components+number_of_wide_components)
	amp_array = popt[0:amplitude_len]
	center_array = popt[amplitude_len:center_len]
	sigma_array = popt[center_len:sigma_len]
	reddening_val = popt[sigma_len]
	if len(popt)==(sigma_len+1):
		coeff=None
	else:
		coeff = popt[sigma_len+1:]
	return(amp_array, center_array, sigma_array, reddening_val, coeff)

##################################GET_PARAMS_AND_BOUNDS##################################

##################################GET_OPTIMUM_PARAMETERS##################################

def get_opt_param_array_rev2(wave, flux, cont, number_of_narrow_components_init, number_of_wide_components_init, center_list_init, redshift_val, comments_on_Balmer, comments_on_tied, factor_for_tied, e_b_minus_v_init, window_for_fit, stopping_number_for_continuum, center_init_array, sigma_init_array):
	popt_init = []
	amplitude_array, position_init_narrow_comp, position_init_wide_comp, position_final_narrow_comp, position_final_wide_comp, index_for_fixing_bounds_factor_init_narrow_comp, index_for_fixing_bounds_factor_init_wide_comp = get_fitting_amplitude_list_rev2(wave, flux, number_of_narrow_components_init, number_of_wide_components_init, center_list_init, redshift_val, comments_on_Balmer, comments_on_tied, factor_for_tied)
	popt_init.extend(amplitude_array)
	popt_init.extend(center_init_array)
	popt_init.extend(sigma_init_array)
	popt_init.extend([e_b_minus_v_init])
	masked_array = numpy.zeros((len(wave)), dtype=bool)
	for i in range(len(center_list_init)):
		centre = center_list_init[i]*(1.+redshift_val)
		vel_array = vel_prof(wave, centre)
		mask = (vel_array > -2*window_for_fit) & (vel_array < 2*window_for_fit)
		masked_array[mask] = True
	order_test_init, coeff_init_val = chebyshev_order(wave[masked_array], cont[masked_array], stopping_number=stopping_number_for_continuum)
	popt_init.extend(coeff_init_val)
	popt_init = np.nan_to_num(popt_init, copy=True, nan=1e-6, posinf=1e-6, neginf=1e-6)
	return (popt_init, masked_array, amplitude_array, position_init_narrow_comp, position_init_wide_comp, position_final_narrow_comp, position_final_wide_comp, coeff_init_val, index_for_fixing_bounds_factor_init_narrow_comp, index_for_fixing_bounds_factor_init_wide_comp)


def get_fitting_amplitude_list_rev2(wave, flux, number_of_narrow_components_init, number_of_wide_components_init, center_list_init, redshift_val, comments_on_Balmer, comments_on_tied, factor_for_tied):
	amplitude_array = []
	position_init_narrow_comp = []
	position_init_wide_comp = []
	factor_init_narrow_comp = []
	factor_init_wide_comp = []
	index_for_fixing_bounds_factor_init_narrow_comp = []
	index_for_fixing_bounds_factor_init_wide_comp = []
	position_final_narrow_comp = []
	position_final_wide_comp = []
	count_init = 0
	count_paired = 0
	for j in range(len(center_list_init)):
		idx_gen = np.searchsorted(wave, (center_list_init[j]*(1.+redshift_val)))
		if (comments_on_Balmer[j]):
			if (comments_on_tied[j]):
				if ('tied_init' in comments_on_tied[j]):
					amplitude_array.extend([np.log10(flux[idx_gen])]*number_of_narrow_components_init)
					position_init_narrow_comp.extend(np.arange(count_init, count_init+number_of_narrow_components_init))
					count_init+=number_of_narrow_components_init
					count_paired+=number_of_narrow_components_init
					amplitude_array.extend([np.log10(flux[idx_gen])]*number_of_wide_components_init)
					position_init_wide_comp.extend(np.arange(count_init, count_init+number_of_wide_components_init))
					count_init+=number_of_wide_components_init
					count_paired+=number_of_wide_components_init
				elif ('tied_paired' in comments_on_tied[j]):
					position_final_narrow_comp.extend(np.arange(count_paired, count_paired+number_of_narrow_components_init))
					if (number_of_narrow_components_init):
						factor_init_narrow_comp.extend([factor_for_tied[j]])
						index_for_fixing_bounds_factor_init_narrow_comp.extend([j])
					position_final_wide_comp.extend(np.arange(count_paired+number_of_narrow_components_init, count_paired+number_of_narrow_components_init+number_of_wide_components_init))
					if (number_of_wide_components_init):
						factor_init_wide_comp.extend([factor_for_tied[j]])
						index_for_fixing_bounds_factor_init_wide_comp.extend([j])
					count_paired+=(number_of_narrow_components_init+number_of_wide_components_init)
					count_init+=(number_of_narrow_components_init+number_of_wide_components_init)
			else:
					amplitude_array.extend([np.log10(flux[idx_gen])]*number_of_narrow_components_init)
					amplitude_array.extend([np.log10(flux[idx_gen])]*number_of_wide_components_init)
		else:
			if (comments_on_tied[j]):
				if ('tied_init' in comments_on_tied[j]):
					amplitude_array.extend([np.log10(flux[idx_gen])]*number_of_narrow_components_init)
					position_init_narrow_comp.extend(np.arange(count_init, count_init+number_of_narrow_components_init))
					count_init+=number_of_narrow_components_init
					count_paired+=number_of_narrow_components_init
				elif ('tied_paired' in comments_on_tied[j]):
					position_final_narrow_comp.extend(np.arange(count_paired, count_paired+number_of_narrow_components_init))
					if (number_of_narrow_components_init):
						factor_init_narrow_comp.extend([factor_for_tied[j]])
						index_for_fixing_bounds_factor_init_narrow_comp.extend([j])
					count_paired+=number_of_narrow_components_init
					count_init+=number_of_narrow_components_init
			else:
					amplitude_array.extend([np.log10(flux[idx_gen])]*number_of_narrow_components_init)
	amplitude_array.extend(factor_init_narrow_comp)
	amplitude_array.extend(factor_init_wide_comp)
	amplitude_array = np.nan_to_num(amplitude_array, copy=True, nan=1e-6, posinf=1e-6, neginf=1e-6)
	return (amplitude_array, position_init_narrow_comp, position_init_wide_comp, position_final_narrow_comp, position_final_wide_comp, index_for_fixing_bounds_factor_init_narrow_comp, index_for_fixing_bounds_factor_init_wide_comp)

def retrieve_all_amplitude_list_rev2(amplitude_array, number_of_narrow_components_init, number_of_wide_components_init, position_init_narrow_comp, position_init_wide_comp, position_final_narrow_comp, position_final_wide_comp, comments_on_tied, comments_on_Balmer):
	#print (position_init_narrow_comp, position_init_wide_comp)
	#print (position_final_narrow_comp, position_final_wide_comp)
	mask1 = [index for index,value in enumerate(comments_on_tied) if 'tied_init' in value]
	mask2 = np.where(np.array(comments_on_Balmer)==True)[0]
	if (number_of_wide_components_init):
		wide_component_index = len(list(set(mask1) - (set(mask1) - set(mask2))))
	else:
		wide_component_index = 0
	if (number_of_narrow_components_init):
		narrow_component_index = len(mask1)
	else:
		narrow_component_index = 0
	
	total_index = []
	if (wide_component_index):
		#print ('wide')
		index_for_wide_factor = amplitude_array[-wide_component_index:]
		if (narrow_component_index):
			#print ('plus narrow')
			index_for_narrow_factor = amplitude_array[-(wide_component_index+narrow_component_index):-(wide_component_index)]
			if (type(index_for_narrow_factor) is not list):
				index_for_narrow_factor = [index_for_narrow_factor]
			index_for_narrow_factor_rev = np.repeat(index_for_narrow_factor,number_of_narrow_components_init)
			total_index.extend(index_for_narrow_factor_rev)
		if (type(index_for_wide_factor) is not list):
			index_for_wide_factor = [index_for_wide_factor]
		index_for_wide_factor_rev = np.repeat(index_for_wide_factor,number_of_wide_components_init)
		total_index.extend(index_for_wide_factor_rev)
	elif (narrow_component_index) and not (wide_component_index):
		#print ('only narrow')
		index_for_narrow_factor = amplitude_array[-narrow_component_index:]
		if (type(index_for_narrow_factor) is not list):
			index_for_narrow_factor = [index_for_narrow_factor]
		index_for_narrow_factor_rev = np.repeat(index_for_narrow_factor,number_of_narrow_components_init)
		total_index.extend(index_for_narrow_factor_rev)

	rev_amp1 = amplitude_array[:-(wide_component_index+narrow_component_index)]
	rev_amp2 = rev_amp1
	position_init = list(np.append(position_init_narrow_comp, position_init_wide_comp))
	position_final = list(np.append(position_final_narrow_comp, position_final_wide_comp))
	position_final_sorted = [x for _, x in sorted(zip(position_init, position_final))]
	total_index_sorted = [x for _, x in sorted(zip(position_init, total_index))]
	position_init_sorted = np.sort(position_init)
	for i in range(len(position_init_sorted)):
		rev_amp2.insert(int(position_final_sorted[i]), (rev_amp1[int(position_init_sorted[i])]+np.log10(total_index_sorted[i])))

	rev_amp2 = np.nan_to_num(rev_amp2, copy=True, nan=1e-6, posinf=1e-6, neginf=1e-6)
	return(rev_amp2)

##################################GET_OPTIMUM_PARAMETERS##################################

########################################FLIP########################################

def flip(m, axis):
    if not hasattr(m, 'ndim'):
        m = asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]

########################################FLIP########################################

####################ARRAY_SMOOTHING_FUNCTION####################
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth
####################ARRAY_SMOOTHING_FUNCTION####################








##################################EXTINCTION_FITTING##################################

##################################REDENNING_FUNCTION##################################
#Never use SMC Wing for anything - Tip from JK, hence 'SMC Wing' removed
#raj = np.array(['Galactic', 'SMC_Bar', 'LMC_Supershell', 'LMC_Average'])

def func_6(xdata, ydata, *t):
	E_bv_dla, flux_redu, E_bv_qso, del_beta=t
	extinction_class_string = 'Galactic'
	x0_dla, gamma_dla, c1_dla, c2_dla, c3_dla, c4_dla, c5_dla, O1_dla, O2_dla, O3_dla, R_v_dla, k_IR_dla = extinction(extinction_class_string)
	x0_qso, gamma_qso, c1_qso, c2_qso, c3_qso, c4_qso, c5_qso, O1_qso, O2_qso, O3_qso, R_v_qso, k_IR_qso = extinction('SMC_Bar')
	#model = ((reddening_func2(ydata, wave, E_bv_dla, R_v_dla, spline_part_new_fit(xdata, extinction_class_string, c3_dla), 0., R_v_qso, spline_part_new_fit_qso(xdata), 0.))/flux_redu)
	model = ((reddening_func2(ydata, xdata, E_bv_dla, R_v_dla, spline_part_new_fit(xdata, extinction_class_string, c3_dla), E_bv_qso, R_v_qso, spline_part_new_fit_qso(xdata), del_beta))/flux_redu)
	return (model)

##################################REDENNING_FUNCTION##################################

#######################################DEFINING_FIXED_PARAMETERS#######################################

def extinction(extinction_class):
	if (extinction_class=='Galactic'):
		x0 = 4.592; gamma = 0.922; c1 = -0.175;	c2 = 0.807; c3 = 2.991; c4 = 0.319; c5 = 6.097; O1 = 2.055; O2 = 1.322; O3 = 0.000
		R_v = 3.001; k_IR = 1.057; x0_err = 0.00; gamma_err = 0.00; c1_err = 0.00; c2_err = 0.00; c3_err = 0.00; c4_err = 0.00
		c5_err = 0.00; O1_err = 0.00; O2_err = 0.00; O3_err = 0.00; R_v_err = 0.00; k_IR_err = 0.00

	elif (extinction_class=='SMC_Bar'):
		x0 = 4.600; gamma = 1.000; c1 = -4.959; c2 = 2.264; c3 = 0.389; c4 = 0.461; c5 = 6.097; O1 = 2.055; O2 = 1.322; O3 = 0.000
		R_v = 2.74; k_IR = 1.057; x0_err = 0.00; gamma_err = 0.00; c1_err = 0.197; c2_err = 0.040; c3_err = 0.110; c4_err = 0.079
		c5_err = 0.00; O1_err = 0.00; O2_err = 0.00; O3_err = 0.00; R_v_err = 0.00; k_IR_err = 0.00

	elif (extinction_class=='SMC_Wing'):
		x0 = 4.703; gamma = 1.212; c1 = -0.856; c2 = 1.038; c3 = 3.215;	c4 = 0.107; c5 = 6.097; O1 = 2.055; O2 = 1.322; O3 = 0.000
		R_v = 2.05; k_IR = 1.057; x0_err = 0.018; gamma_err = 0.019; c1_err = 0.246; c2_err = 0.074; c3_err = 0.439; c4_err = 0.038
		c5_err = 0.00; O1_err = 0.00; O2_err = 0.00; O3_err = 0.00; R_v_err = 0.00; k_IR_err = 0.00

	elif (extinction_class=='LMC_Supershell'):
		x0 = 4.558; gamma = 0.945; c1 = -1.475; c2 = 1.132; c3 = 1.463; c4 = 0.294; c5 = 6.097; O1 = 2.055; O2 = 1.322; O3 = 0.000
		R_v = 2.76; k_IR = 1.057; x0_err = 0.021; gamma_err = 0.026; c1_err = 0.152; c2_err = 0.029; c3_err = 0.121; c4_err = 0.057
		c5_err = 0.00; O1_err = 0.00; O2_err = 0.00; O3_err = 0.00; R_v_err = 0.00; k_IR_err = 0.00

	elif (extinction_class=='LMC_Average'):
		x0 = 4.579; gamma = 0.934; c1 = -0.890; c2 = 0.998; c3 = 2.719; c4 = 0.400; c5 = 6.097; O1 = 2.055; O2 = 1.322; O3 = 0.000
		R_v = 3.41; k_IR = 1.057; x0_err = 0.007; gamma_err = 0.016; c1_err = 0.142; c2_err = 0.027; c3_err = 0.137; c4_err = 0.036
		c5_err = 0.00; O1_err = 0.00; O2_err = 0.00; O3_err = 0.00; R_v_err = 0.00; k_IR_err = 0.00


	return (x0, gamma, c1, c2, c3, c4, c5, O1, O2, O3, R_v, k_IR)

#######################################DEFINING_FIXED_PARAMETERS#######################################

####################################REDDENING_FUNCTION####################################
def reddening_func2(F_rest_lambda, wave, E_bv_dla, R_v_dla, initial_result_dla, E_bv_qso, R_v_qso, initial_result_qso, del_beta):
	constant = np.zeros([len(F_rest_lambda)])
	F_lambda = np.zeros([len(F_rest_lambda)])
	for i in range(len(F_rest_lambda)):
		constant[i] = -0.4 * ((E_bv_dla * (initial_result_dla[i] + R_v_dla)) + (E_bv_qso * (initial_result_qso[i] + R_v_qso)))
		F_lambda[i] = ((F_rest_lambda[i]*((wave[i]/5510.)**(del_beta))) * (10.**(constant[i])))
	return (F_lambda)
	
def k_lambda_V(x, x0, gamma, c1, c2, c3, c4, c5):
	D_func = np.zeros([len(x)])
	result = np.zeros([len(x)])
	for i in range(len(x)):
		D_func[i] = x[i]**2. / ( ( (x[i]**2.) - (x0**2.) )**2. + (x[i] * gamma)**2. )
		if (x[i] <= c5):
			result[i] = c1 + (c2*x[i]) + c3*D_func[i]
		else:
			result[i] = c1 + (c2*x[i]) + c3*D_func[i] + c4*((x[i]-c5)**2)
	return (result)
	
#GLOBAL PARAMETER DEFINITIONS
U1_pos = 0.27; U2_pos = 0.26; O1_pos = 0.33; O2_pos = 0.4; O3_pos = 0.553; O4_pos = 0.7; I1_pos = 0; I2_pos = 1/0.25; I3_pos = 1/0.50; I4_pos = 1/0.75; I5_pos = 1/1.00
####################################REDDENING_FUNCTION####################################

#########################################REDDENING_BACKBONE_DLA#########################################
def spline_part_new_fit(wave, extinction_class_string, c3_fit_new):
	z = 0.0
	x0, gamma, c1, c2, c3, c4, c5, O1, O2, O3, R_v, k_IR = extinction(extinction_class_string)
	c3=c3_fit_new
	I1_pos_new = 0.00000000001
	I2_pos_new = 1/I2_pos
	I3_pos_new = 1/I3_pos
	I4_pos_new = 1/I4_pos
	I5_pos_new = 1/I5_pos
	O1_val = O1
	O2_val = O2
	O3_val = O3
	def I_n(k_IR, pos, R_v):
		return (((k_IR * (pos**(-1.84))) - R_v))
	I1_val = I_n(k_IR, I1_pos_new, R_v)
	I2_val = I_n(k_IR, I2_pos_new, R_v)
	I3_val = I_n(k_IR, I3_pos_new, R_v)
	I4_val = I_n(k_IR, I4_pos_new, R_v)
	I5_val = I_n(k_IR, I5_pos_new, R_v)
	x_new = np.array([1/O1_pos, 1/O2_pos, 1/O3_pos, 1/I4_pos, 1/I3_pos, 1/I2_pos, 0])
	y_new = np.array([O1_val, O2_val, O3_val, I4_val, I3_val, I2_val, I1_val])
	func = interpolate.interp1d(x_new, y_new)
	wave_new = (wave*1e-4)/(1+z)
	x = np.sort(1/(wave_new))
	x1 = np.array([])
	x2 = np.array([])
	for i in range(len(x)):
		if (x[i] < (1/O1_pos)):
			x1 = np.append(x1, x[i])
		else:
			x2 = np.append(x2, x[i])
	y1 = func(x1)
	position = np.searchsorted(x, (1/O1_pos))
	y = np.zeros([len(x)])
	y2 = k_lambda_V(x2, x0, gamma, c1, c2, c3, c4, c5)
	for i in range(len(x)):
		if (x[i] < (1/O1_pos)):
			y[i] = y1[i]
		else:
			y[i] = y2[i-position]
	y_k_lambda_V = y
	y_k_lambda_V = flip(y_k_lambda_V, 0)
	return (y_k_lambda_V)
#########################################REDDENING_BACKBONE_DLA#########################################

#########################################REDDENING_BACKBONE_QSO#########################################

def spline_part_new_fit_qso(wave):
	z = 0.0
	extinction_class_string = 'SMC_Bar'
	x0, gamma, c1, c2, c3, c4, c5, O1, O2, O3, R_v, k_IR = extinction(extinction_class_string)
	#c3=c3_fit_new
	I1_pos_new = 0.00000000001
	I2_pos_new = 1/I2_pos
	I3_pos_new = 1/I3_pos
	I4_pos_new = 1/I4_pos
	I5_pos_new = 1/I5_pos
	O1_val = O1
	O2_val = O2
	O3_val = O3
	def I_n(k_IR, pos, R_v):
		return (((k_IR * (pos**(-1.84))) - R_v))
	I1_val = I_n(k_IR, I1_pos_new, R_v)
	I2_val = I_n(k_IR, I2_pos_new, R_v)
	I3_val = I_n(k_IR, I3_pos_new, R_v)
	I4_val = I_n(k_IR, I4_pos_new, R_v)
	I5_val = I_n(k_IR, I5_pos_new, R_v)
	x_new = np.array([1/O1_pos, 1/O2_pos, 1/O3_pos, 1/I4_pos, 1/I3_pos, 1/I2_pos, 0])
	y_new = np.array([O1_val, O2_val, O3_val, I4_val, I3_val, I2_val, I1_val])
	func = interpolate.interp1d(x_new, y_new)
	wave_new = (wave*1e-4)/(1+z)
	x = np.sort(1/(wave_new))
	x1 = np.array([])
	x2 = np.array([])
	for i in range(len(x)):
		if (x[i] < (1/O1_pos)):
			x1 = np.append(x1, x[i])
		else:
			x2 = np.append(x2, x[i])
	y1 = func(x1)
	position = np.searchsorted(x, (1/O1_pos))
	y = np.zeros([len(x)])
	y2 = k_lambda_V(x2, x0, gamma, c1, c2, c3, c4, c5)
	for i in range(len(x)):
		if (x[i] < (1/O1_pos)):
			y[i] = y1[i]
		else:
			y[i] = y2[i-position]
	y_k_lambda_V = y
	y_k_lambda_V = flip(y_k_lambda_V, 0)
	return (y_k_lambda_V)

#########################################REDDENING_BACKBONE_QSO#########################################
##################################EXTINCTION_FITTING##################################



##################################CHEBYSHEV_FUNCTIONS##################################

def chebyshev_order(wave, cont, stopping_number):
    wave_new = np.linspace(-1, 1, len(wave))
    i=1
    while True:
        roots = numpy.polynomial.chebyshev.chebfit(wave_new, cont, i, rcond=None, full=False, w=None)
        poly = numpy.polynomial.chebyshev.chebval(wave_new, roots, tensor=True)
        chi_sq = ((poly - cont) ** 2)
        chi_sq_sum = (np.sum(chi_sq))/len(cont)
        i+=1
        #print (chi_sq_sum)
        if chi_sq_sum<(float(stopping_number)):
            break
    return (i, roots)

def chebyshev_fit(wave, cont, order_test):
    wave_new = np.linspace(-1, 1, len(wave))
    roots = numpy.polynomial.chebyshev.chebfit(wave_new, cont, int(order_test), rcond=None, full=False, w=None)
    poly_new = numpy.polynomial.chebyshev.chebval(wave_new, roots, tensor=True)
    return(roots, poly_new)

def chebyshev_disp(wave, coeff):
    wave_new = np.linspace(-1, 1, len(wave))
    poly_new = numpy.polynomial.chebyshev.chebval(wave_new, coeff, tensor=True)
    return(poly_new)

##################################CHEBYSHEV_FUNCTIONS##################################

##################################GAUSS_BASED_FUNCTIONS##################################

def vel_prof(x, centre):
    xnew = c_kms * ((x-centre)/x)
    return (xnew)

def wave_prof(vel_center, centre):
    xnew = (centre*c_kms) / (c_kms-vel_center)
    return (xnew)

def gaus_prof_vel(vel_array, amp, center, sigma):
	prof = (10**amp)*np.exp(-(vel_array-center)**2./(2.*sigma**2.))
	return (prof)

##################################GAUSS_BASED_FUNCTIONS##################################


##################################GET_CONTINUUM##################################

def get_initial_continuum(wave, flux):
	x = wave
	coordsx = [4750, 4800, 4840, 4870, 4900, 4940, 4960, 4974, 4985, 5000, 5050, 5155, 5223, 5274, 5296, 5300, 5331, 5365, 5400, 5440, 5485, 5531, 5541, 5568, 5603, 5640, 5734, 5770, 5808, 5844, 5916, 5970, 5992, 6031, 6056, 6121, 6130, 6241, 6339, 6429, 6545, 6649, 6711, 6722, 6799, 6990, 7106, 7112, 7200, 7324, 7385, 7448, 7530, 7583, 7659, 7725, 7775, 7890, 8024, 8100, 8200, 8300, 8400, 8500, 8812, 8945, 9039, 9164, 9332]

	coordsy = np.zeros_like(coordsx)
	for i in range(len(coordsx)):
		idx_new = np.searchsorted(wave, coordsx[i])
		coordsy[i] = flux[idx_new]

	points = zip(coordsx, coordsy)
	points = sorted(points, key=lambda point: point[0])
	x1, y1 = zip(*points)
	new_length = len(x)
	l1 = np.searchsorted(x, min(x1))
	l2 = np.searchsorted(x, max(x1))
	new_x = []
	new_y = []
	new_x = np.linspace(min(x1), max(x1), (l2-l1))
	new_y = sp.interpolate.splrep(x1, y1)
	cont_init = sp.interpolate.splev(x, new_y, der=0)
	return (cont_init)


from scipy.signal import find_peaks
from scipy import signal
from scipy.signal import savgol_filter


#This function has been created from literature (Martin+2020)
def get_initial_continuum_revised(wave, flux, smoothing_par=20, find_peaks_prominence=0.6, find_peaks_width=10, allowed_percentile=25, savgol_filter_window_length=501, savgol_filter_polyorder=5, plot=False):
	x = wave
	y = smooth(flux, smoothing_par)
	residual = np.abs(flux - y)
	peaks, dict_cust = find_peaks(y, prominence=(None, find_peaks_prominence), width=find_peaks_width)
	left_edge = dict_cust['left_bases']
	right_edge = dict_cust['right_bases']
	left_width = dict_cust['left_ips']
	right_width = dict_cust['right_ips']
	mask = np.ones_like(wave, dtype=np.bool8)
	for i in range(len(left_width)):
		dw = (int(right_width[i]) - int(left_width[i]))
		mask[int(left_width[i])-dw:int(right_width[i])+dw] = False
	ynew = np.abs(np.diff(flux[mask], prepend=1e-10))
	ynew2 = np.percentile(ynew, allowed_percentile)
	x_rev = x[mask][ynew < ynew2]
	y_rev = flux[mask][ynew < ynew2]
	f_flux = interpolate.interp1d(x_rev, y_rev, axis=0, fill_value="extrapolate", kind='linear')
	y_rev2 = f_flux(x)
	y_rev3 = savgol_filter(y_rev2, window_length = savgol_filter_window_length, polyorder=savgol_filter_polyorder)
	'''
	y = smooth(flux, smoothing_par)
	ydata_normalized = flux / y
	mask = (ydata_normalized > (1.-allowed_deviation)) & (ydata_normalized < (1.+allowed_deviation))
	xnew_sel = x[mask]
	ynew_sel = y[mask]
	f_flux = interpolate.interp1d(xnew_sel, ynew_sel, axis=0, fill_value="extrapolate", kind='linear')
	flux_rebinned = f_flux(x)
	flux_rebinned_rev = smooth(flux_rebinned, smoothing_par)
	plt.plot(wave, flux, zorder=1)
	plt.plot(x, y, 'r--', zorder=2)
	plt.plot(xnew_sel, ynew_sel, 'm.', zorder=3)
	plt.plot(x,flux_rebinned_rev, 'g--', zorder=4)
	plt.show()
	quit()
	'''
	if (plot):
		plt.plot(wave, flux, zorder=1)
		plt.plot(x, y, 'r--', zorder=2)
		plt.plot(x, y_rev3, 'g--', zorder=4)
		plt.show()

	#quit()

	return (y_rev3)



#This function has been created from literature (Martin+2020)
def get_initial_continuum_rev_2(wave, flux, smoothing_par=5, allowed_percentile=5, savgol_filter_window_length=500, savgol_filter_polyorder=5, fwhm_galaxy=3, line_type='emission', plot=False):
	if (savgol_filter_window_length % 2) == 0:
		savgol_filter_window_length+=1
	local_std = np.median([ np.std(s) for s in np.array_split(flux, smoothing_par) ])
	fwhm_index = int(np.ceil((3.*fwhm_galaxy / np.mean(np.diff(wave)))))
	if line_type=='absorption':
		print ('Estimating continuum for absorption lines.')
		peaks, dict_cust = find_peaks(flux, width = [fwhm_index, None], prominence=(None, local_std), height=[None,0])
	else:
		print ('Estimating continuum for emission lines.')
		peaks, dict_cust = find_peaks(flux, width = [fwhm_index, None], prominence=(None, local_std), height=[0,None])

	left_edge = dict_cust['left_bases']
	right_edge = dict_cust['right_bases']
	left_width = dict_cust['left_ips']
	right_width = dict_cust['right_ips']
	mask = np.ones_like(wave, dtype=np.bool8)
	for i in range(len(left_width)):
		dw = (int(right_width[i]) - int(left_width[i]))
		mask[int(left_width[i])-dw:int(right_width[i])+dw] = False
	ynew = np.abs(np.diff(flux[mask], prepend=1e-10))
	ynew2 = np.percentile(ynew, allowed_percentile)
	x_rev = wave[mask][ynew < ynew2]
	y_rev = flux[mask][ynew < ynew2]
	f_flux = interpolate.interp1d(x_rev, y_rev, axis=0, fill_value="extrapolate", kind='linear')
	y_rev2 = f_flux(wave)
	y_rev3 = savgol_filter(y_rev2, window_length = savgol_filter_window_length, polyorder=savgol_filter_polyorder)
	if (plot):
		plt.plot(wave, flux, zorder=1)
		plt.plot(x_rev, y_rev, 'g--', zorder=4)
		plt.plot(wave, y_rev3, 'r--', zorder=5)
		plt.show()
	return (y_rev3)


import scipy as scp
def fit_cont(tsteps, a, plot=False, legend=False, ax=None, n_smooth=5, order=8, label=None, color='darkblue'):
    pick = np.isfinite(a)
    a = a[pick]
    local_std = np.median([ np.std(s) for s in np.array_split(a, n_smooth) ])
    peaks = find_peaks(scp.ndimage.convolve1d(a,np.asarray([1.]*n_smooth)/n_smooth), width = [3,100], prominence=local_std*1., height=[0,None])
    edges = np.int32([np.round(peaks[1]['left_ips']), np.round(peaks[1]['right_ips'])])
    d = (np.diff(a, n=1))
    w = 1./np.concatenate((np.asarray([np.median(d)]*1),d))
    w[0] = np.max(w)
    w[-1] = np.max(w)
    for edge in edges.T:
        w[edge[0]:edge[1]] = 1./10000.
    w = np.abs(w)

    pick_2 = np.where(w > np.percentile(w, 75 * (float(len(a)) / float(len(tsteps)))))[0]
    #fit = np.poly1d(np.polyfit(tsteps[pick][pick_2], a[pick_2], order))
    
    xx = np.linspace(np.min(tsteps[pick][pick_2]), np.max(tsteps[pick][pick_2]), 1000)
    itp = interpolate.interp1d(tsteps[pick][pick_2], a[pick_2], kind='linear')
    window_size = int(((1.0 / (step_to_t[pick][pick_2][-1] - step_to_t[pick][pick_2][0])) * 1000.))
    if window_size % 2 == 0:
        window_size = window_size + 1
    poly_order = 3
    fit_savgol = savgol_filter(itp(xx), window_size, poly_order)
    
    fit = interpolate.interp1d(xx, fit_savgol, kind='cubic', fill_value="extrapolate")
    
    
    #std_cont = np.std((a[pick_2] - fit(tsteps[pick][pick_2])) / fit(tsteps[pick][pick_2]))
    std_cont = np.std(a[pick_2] - fit(tsteps[pick][pick_2]))
    #print std_cont
    #r = (a[pick_2] - fit(tsteps[pick][pick_2])) / fit(tsteps[pick][pick_2])
    r = a[pick_2] - fit(tsteps[pick][pick_2])
    #print ([ np.std(s) for s in np.array_split(r, n_smooth) ])
    #print std_cont
    #print '-----'
    
    if plot:
        tt = step_to_t[tsteps[pick] - np.min(tsteps)]
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,4))
            
        if label:
            ax.plot(tt, a, color=color, label=label)
        else:
            ax.plot(tt, a, color=color, zorder=10)
        if legend:
            ax.plot(tt, fit(tsteps[pick]), color='k', ls='--', label='Fit', zorder=12, lw=1.5)
            ax.plot(tt[pick_2], a[pick_2], 'o ', zorder=10, markerfacecolor='none',
                    markeredgecolor='r', markersize=5, label='Used in fit', alpha=0.8)
            ax.plot(tt[peaks[0]], peaks[1]['peak_heights'], 'x ', color='orange', zorder=11, label='Peaks (masked)')
        else:
            ax.plot(tt, fit(tsteps[pick]), color='k', ls='--', zorder=12, lw=1.5)
            ax.plot(tt[pick_2], a[pick_2], 'o ', zorder=10, markerfacecolor='none',
                    markeredgecolor='r', markersize=5, alpha=0.8)
            ax.plot(tt[peaks[0]], peaks[1]['peak_heights'], 'x ', color='orange', zorder=11)
            ax.plot(tt, a, color=color, zorder=10)
        ax.fill_between(tt, fit(tsteps[pick])+std_cont*1.5, fit(tsteps[pick])-std_cont*1.5, color='darkred', alpha=0.15)
        
    return fit, std_cont


##################################GET_CONTINUUM##################################


############################CONVOLUTION#####################################

def convolved_prof5(wave, profile, res):
	#res = 3026.62381895424
	wave_short = np.logspace(np.log10(wave.min()), np.log10(wave.max()), len(wave)*10)
	center = np.searchsorted(wave_short, np.log10(np.median(wave_short)))
	deltalam = wave_short[center + 1] - wave_short[center]
	sigma = (wave_short[center]) / (res*2. * (2 * np.sqrt(2 * np.log(2))) * deltalam)
	gauss = scipy.signal.gaussian(len(profile), sigma, sym=True)
	gauss = gauss / np.sum(gauss)
	prof_new = signal.fftconvolve(profile, gauss, mode='same')
	return (prof_new)

############################CONVOLUTION#####################################
