import scipy
import scipy as sp
from scipy import signal
import numpy
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit
from tabulate import tabulate
import ast
import sys
from custom_functions import *
global c, c_kms, window_for_fit, redshift_val, center_list_init, poly_bound_val, amp_init_array, index_for_tied_init, index_for_tied_paired, factor_for_tied_paired, coeff_init_val, order_test_init, continuum, sigma_narrow_val_init, sigma_wide_val_init, fit_continuum_val, minimum_amplitude_val, maximum_amplitude_val, minimum_reddening_val, maximum_reddening_val, sigma_min_bound_val, index_for_tied_init_cross_component_tying, index_for_tied_paired_cross_component_tying, e_b_minus_v_init, plot_fit_refined, multiplicative_factor_for_plot, factor_fixed_for_tying, index_for_fixing_bounds_factor_init_narrow_comp, index_for_fixing_bounds_factor_init_wide_comp, flux_sky_orig

c = 299792.458 # Speed in Light in Km/s
c_kms = 299792.458 # Speed in Light in Km/s
parameter_file_string_base = 'basic_dictionary.dat'
if (len(sys.argv)==2):
	parameter_file_string = str(sys.argv[1])
else:
	parameter_file_string = parameter_file_string_base

initial_guesses = {}
with open(str(parameter_file_string)) as f:
	for line in f:
		if '#' not in line:
			#print (len(line.split()))
			if (len(line.split())>2):
				(key, val) = line.split(':')
				key = key.replace(':', '').replace('-', '').lower()
				initial_guesses[str(key)] = ast.literal_eval(val.replace(' ', ''))
			else:
				(key, val) = line.split()
				key = key.replace(':', '').replace('-', '').lower()
				initial_guesses[str(key)] = val

sky_spectra_file = str(initial_guesses['sky_spectra_file'])
window_for_fit = float(initial_guesses['window_for_fit'])
redshift_val = float(initial_guesses['redshift_val'])
size_of_font = int(initial_guesses['size_of_font'])
poly_bound_val = float(initial_guesses['poly_bound_val'])
e_b_minus_v_init = float(initial_guesses['e_b_minus_v_init'])
stopping_number_for_continuum = float(initial_guesses['stopping_number_for_continuum'])
sigma_narrow_val_init = float(initial_guesses['sigma_narrow_val_init'])
sigma_wide_val_init = float(initial_guesses['sigma_wide_val_init'])
fit_continuum_val = str2bool(initial_guesses['fit_continuum_val'])
fit_reddening_val = str2bool(initial_guesses['fit_reddening_val'])
minimum_amplitude_val = float(initial_guesses['minimum_amplitude_val'])
maximum_amplitude_val = float(initial_guesses['maximum_amplitude_val'])
minimum_reddening_val = float(initial_guesses['minimum_reddening_val'])
maximum_reddening_val = float(initial_guesses['maximum_reddening_val'])
sigma_min_bound_val = float(initial_guesses['sigma_min_bound_val'])
plot_fit_refined = str2bool(initial_guesses['plot_fit_refined'])
multiplicative_factor_for_plot = int(initial_guesses['multiplicative_factor_for_plot'])
muse_data_filename = str(initial_guesses['muse_data_filename'])
vel_window = float(initial_guesses['vel_window'])
center_list_names_plot = list(initial_guesses['center_list_names_plot'])
center_list_plot = list(initial_guesses['center_list_plot'])
x_pos = int(initial_guesses['target_data_region_x_axis_pos'])
y_pos = int(initial_guesses['target_data_region_y_axis_pos'])
file_type = str(initial_guesses['file_type_for_data_rebinning'])
require_air_to_vaccum = str2bool(initial_guesses['require_air_to_vaccum'])
extra_redshift = float(initial_guesses['extra_redshift'])
number_of_narrow_components_init = int(initial_guesses['number_of_narrow_components_init'])
number_of_wide_components_init = int(initial_guesses['number_of_wide_components_init'])
center_list_init_names = list(initial_guesses['center_list_init_names'])
center_list_init = list(initial_guesses['center_list_init'])
comments_on_Balmer = list(initial_guesses['comments_on_balmer'])
comments_on_tied = list(initial_guesses['comments_on_tied'])
factor_for_tied = list(initial_guesses['factor_for_tied'])
factor_fixed_for_tying = list(initial_guesses['factor_fixed_for_tying'])
x_slot = int(initial_guesses['x_slot'])
y_slot = int(initial_guesses['y_slot'])
fig_x_size = int(initial_guesses['fig_x_size'])
fig_y_size = int(initial_guesses['fig_y_size'])
dpi_val = int(initial_guesses['dpi_val'])
qso_name = str(initial_guesses['qso_name'])
file_name = str(initial_guesses['file_name'])
bound_lower_factor_for_tied = np.array(factor_for_tied)-1e-3
bound_upper_factor_for_tied = np.array(factor_for_tied)+1e-3


def get_table(new_fitted_popt_param_array_rev, new_fitted_perr_param_array, center_list_init_names, number_of_narrow_components_init, number_of_wide_components_init):
	final_array_to_save = np.chararray([(len(center_list_init_names)+2), (number_of_narrow_components_init+number_of_wide_components_init)], itemsize=100)
	final_array_to_save[:] = 'a'
	for i in range(final_array_to_save.shape[0]):
		for j in range(final_array_to_save.shape[1]):
			final_array_to_save[i,j] = str(np.round(new_fitted_popt_param_array_rev[i,j],3)) + str("+/-") + str(np.round(new_fitted_perr_param_array[i,j],3))
	column_text = np.append(np.array(['Comp']), center_list_init_names)
	column_text = np.append(column_text, np.array(['V', 'sigma']))
	row_text_1 = np.arange(1, number_of_narrow_components_init+1)
	row_text_1 = np.append(row_text_1, np.arange(number_of_narrow_components_init+1, number_of_narrow_components_init+1+number_of_wide_components_init))
	row_text_1 = list(row_text_1.astype(np.str))
	string = 'Comp'
	row_text = [string + x for x in row_text_1]
	final_array_to_save = numpy.vstack([row_text, final_array_to_save])
	final_array_to_save = np.append(np.array([column_text]).transpose(), final_array_to_save, axis=1)
	return (final_array_to_save)



##################################GET_PARAMS_AND_BOUNDS##################################
def get_bounds(amp_length, number_of_narrow_components, number_of_wide_components, coeff_len, center_list, **kwargs):
	max_amp = kwargs.get('max_amp', maximum_amplitude_val)  # Maximum Amplitude
	center_min_val = kwargs.get('center_min_val', -window_for_fit)  # Minimum center val
	center_max_val = kwargs.get('center_max_val', window_for_fit)  # Maximum center val
	sigma_narrow_val = kwargs.get('sigma_narrow_val', sigma_narrow_val_init)  # Minimum center val
	sigma_wide_val = kwargs.get('sigma_wide_val', sigma_wide_val_init)  # Maximum center val
	fit_continuum = kwargs.get('fit_continuum', fit_continuum_val)  # Fit continuum?
	fit_dust = kwargs.get('fit_dust', True)  # Fit dust?
	amp_ratio_fixed = kwargs.get('factor_fixed_for_tying', factor_fixed_for_tying)  # Fit dust?
	idx_for_fixing_bounds_factor_init_narrow_comp = kwargs.get('index_for_fixing_bounds_factor_init_narrow_comp', index_for_fixing_bounds_factor_init_narrow_comp)  # Fit dust?
	idx_for_fixing_bounds_factor_init_wide_comp = kwargs.get('index_for_fixing_bounds_factor_init_wide_comp', index_for_fixing_bounds_factor_init_wide_comp)  # Fit dust?
	bound_lower_factor_for_tied_amp = kwargs.get('bound_lower_factor_for_tied', bound_lower_factor_for_tied)  # bound_lower_factor_for_tied
	bound_upper_factor_for_tied_amp = kwargs.get('bound_upper_factor_for_tied', bound_upper_factor_for_tied)  # bound_upper_factor_for_tied
	amplitude_len = amp_length
	amp_bound_min = np.full([(amplitude_len)], minimum_amplitude_val)
	amp_bound_max = np.full([(amplitude_len)], max_amp)
	idx_for_fixing_bounds_factor_init_wide_comp = idx_for_fixing_bounds_factor_init_wide_comp[::-1]
	for i in range(len(idx_for_fixing_bounds_factor_init_wide_comp)):
		if (amp_ratio_fixed[int(idx_for_fixing_bounds_factor_init_wide_comp[i])]==True):
			amp_bound_min[-(i+1)] = bound_lower_factor_for_tied_amp[int(idx_for_fixing_bounds_factor_init_wide_comp[i])]
			amp_bound_max[-(i+1)] = bound_upper_factor_for_tied_amp[int(idx_for_fixing_bounds_factor_init_wide_comp[i])]
		else:
			amp_bound_min[-(i+1)] = 1e-5
			amp_bound_max[-(i+1)] = 1e5

	idx_for_fixing_bounds_factor_init_narrow_comp = idx_for_fixing_bounds_factor_init_narrow_comp[::-1]
	for i in range(len(idx_for_fixing_bounds_factor_init_narrow_comp)):
		if (amp_ratio_fixed[int(idx_for_fixing_bounds_factor_init_narrow_comp[i])]==True):
			amp_bound_min[-(i+len(idx_for_fixing_bounds_factor_init_wide_comp)+1)] = bound_lower_factor_for_tied_amp[int(idx_for_fixing_bounds_factor_init_narrow_comp[i])]
			amp_bound_max[-(i+len(idx_for_fixing_bounds_factor_init_wide_comp)+1)] = bound_upper_factor_for_tied_amp[int(idx_for_fixing_bounds_factor_init_narrow_comp[i])]
		else:
			amp_bound_min[-(i+len(idx_for_fixing_bounds_factor_init_wide_comp)+1)] = 1e-5
			amp_bound_max[-(i+len(idx_for_fixing_bounds_factor_init_wide_comp)+1)] = 1e5


	center_bound_min = np.full([(number_of_narrow_components+number_of_wide_components)], center_min_val)
	center_bound_max = np.full([(number_of_narrow_components+number_of_wide_components)], center_max_val)
	sigma_bound_min = np.full([(number_of_narrow_components)], sigma_min_bound_val)
	sigma_bound_min = np.append(sigma_bound_min, np.full([(number_of_wide_components)], sigma_narrow_val))
	sigma_bound_max = np.full([(number_of_narrow_components)], sigma_narrow_val)
	sigma_bound_max = np.append(sigma_bound_max, np.full([(number_of_wide_components)], sigma_wide_val))
	if (fit_dust):
		reddening_bound_min = np.full([(1)], minimum_reddening_val)
		reddening_bound_max = np.full([(1)], maximum_reddening_val)
	else:
		reddening_bound_min = np.full([(1)], (e_b_minus_v_init-(e_b_minus_v_init/10.)))
		reddening_bound_max = np.full([(1)], (e_b_minus_v_init+(e_b_minus_v_init/10.)))
	coeff_lower_bound = np.full([coeff_len], -poly_bound_val)
	coeff_upper_bound = np.full([coeff_len], poly_bound_val)
	bounds_lower = []
	bounds_lower.extend(amp_bound_min)
	bounds_lower.extend(center_bound_min)
	bounds_lower.extend(sigma_bound_min)
	bounds_lower.extend(reddening_bound_min)
	if (fit_continuum):
		bounds_lower.extend(coeff_lower_bound)
	bounds_upper = []
	bounds_upper.extend(amp_bound_max)
	bounds_upper.extend(center_bound_max)
	bounds_upper.extend(sigma_bound_max)
	bounds_upper.extend(reddening_bound_max)
	if (fit_continuum):
		bounds_upper.extend(coeff_upper_bound)
	return(bounds_lower, bounds_upper)
##################################GET_PARAMS_AND_BOUNDS##################################

##################################GAUSS_BASED_FUNCTIONS##################################

def make_table_with_result(wave_array, new_fitted_popt_param_array, *popt, **kwargs):
	center_list = kwargs.get('center_list', center_list_init)  # Centre List
	number_of_narrow_components = kwargs.get('number_of_narrow_components', number_of_narrow_components_init)  # Number of narrow components
	number_of_wide_components = kwargs.get('number_of_wide_components', number_of_wide_components_init)  # Number of wide components
	amp_length = kwargs.get('amp_length', len(amp_init_array))  # length of amplitude array to be fitted
	coeff_init = kwargs.get('coeff_init', coeff_init_val)  # length of amplitude array to be fitted
	check_data_type = kwargs.get('check_data_type', 'data')  # length of amplitude array to be fitted

	amp_array, center_array, sigma_array, reddening_val, coeff = get_params(popt, number_of_narrow_components, number_of_wide_components, center_list, amp_length)
	if (coeff==None):
		coeff_val = coeff_init
	else:
		coeff_val = coeff
	amp_array_rev = retrieve_all_amplitude_list_rev2(list(amp_array), number_of_narrow_components, number_of_wide_components, position_init_narrow_comp, position_init_wide_comp, position_final_narrow_comp, position_final_wide_comp, comments_on_tied, comments_on_Balmer)

	if (check_data_type=='data'):
		amp_array_rev = np.power(10, amp_array_rev)
		
	count_amp = 0
	new_fitted_popt_param_array_rev = new_fitted_popt_param_array
	for j in range(len(center_list)):
		centre = center_list[j]*(1.+redshift_val)
		vel_array = vel_prof(wave_array, centre)
		count = 0
		if (comments_on_Balmer[j]):
			for k in range(number_of_narrow_components):
				new_fitted_popt_param_array[j, count] = amp_array_rev[count_amp]
				count+=1
				count_amp+=1
			for l in range(number_of_wide_components):
				new_fitted_popt_param_array[j, count] = amp_array_rev[count_amp]
				count+=1
				count_amp+=1
		else:
			for m in range(number_of_narrow_components):
				new_fitted_popt_param_array[j, count] = amp_array_rev[count_amp]
				count+=1
				count_amp+=1

	return (new_fitted_popt_param_array_rev)


def plot_gaus_group_with_cont_rev(wave_array_rev, *popt, **kwargs):
	if (plot_fit_refined):
		wave_array = np.logspace(np.log10(wave_array_rev.min()), np.log10(wave_array_rev.max()), num=len(wave_array_rev)*multiplicative_factor_for_plot, endpoint=True, base=10)
	else:
		wave_array = wave_array_rev
	group_prof = np.zeros([len(wave_array)])
	center_list = kwargs.get('center_list', center_list_init)  # Centre List
	number_of_narrow_components = kwargs.get('number_of_narrow_components', number_of_narrow_components_init)  # Number of narrow components
	number_of_wide_components = kwargs.get('number_of_wide_components', number_of_wide_components_init)  # Number of wide components
	amp_length = kwargs.get('amp_length', len(amp_init_array))  # length of amplitude array to be fitted
	coeff_init = kwargs.get('coeff_init', coeff_init_val)  # length of amplitude array to be fitted
	flux_sky = kwargs.get('sky_flux', flux_sky_orig)  # Sky Flux
	amp_array, center_array, sigma_array, reddening_val, coeff = get_params(popt, number_of_narrow_components, number_of_wide_components, center_list, amp_length)
	if (coeff==None):
		coeff_val = coeff_init
	else:
		coeff_val = coeff
	amp_array_rev = retrieve_all_amplitude_list_rev2(list(amp_array), number_of_narrow_components, number_of_wide_components, position_init_narrow_comp, position_init_wide_comp, position_final_narrow_comp, position_final_wide_comp, comments_on_tied, comments_on_Balmer)
	cont_array = chebyshev_disp(wave_array, coeff_val)
	count_amp = 0
	for j in range(len(center_list)):
		centre = center_list[j]*(1.+redshift_val)
		vel_array = vel_prof(wave_array, centre)
		count = 0
		if (comments_on_Balmer[j]):
			for k in range(number_of_narrow_components):
				group_prof += gaus_prof_vel(vel_array, amp_array_rev[count_amp], center_array[count], sigma_array[count])
				count+=1
				count_amp+=1
			for l in range(number_of_wide_components):
				group_prof += gaus_prof_vel(vel_array, amp_array_rev[count_amp], center_array[count], sigma_array[count])
				count+=1
				count_amp+=1
		else:
			for m in range(number_of_narrow_components):
				group_prof += gaus_prof_vel(vel_array, amp_array_rev[count_amp], center_array[count], sigma_array[count])
				count+=1
				count_amp+=1
	group_prof+=cont_array
	reddening_array = [reddening_val, 1.0, 0.0, 0.0]
	group_prof_adv = func_6(wave_array/(1.+redshift_val), group_prof, *reddening_array)
	group_prof_adv_rev = group_prof_adv + flux_sky
	group_prof_convolved = convolved_prof5(wave_array, group_prof_adv_rev, res)
	return (wave_array, group_prof_convolved)

def gaus_group_with_cont_rev(wave_array, *popt, **kwargs):
	group_prof = np.zeros([len(wave_array)])
	center_list = kwargs.get('center_list', center_list_init)  # Centre List
	number_of_narrow_components = kwargs.get('number_of_narrow_components', number_of_narrow_components_init)  # Number of narrow components
	number_of_wide_components = kwargs.get('number_of_wide_components', number_of_wide_components_init)  # Number of wide components
	amp_length = kwargs.get('amp_length', len(amp_init_array))  # length of amplitude array to be fitted
	coeff_init = kwargs.get('coeff_init', coeff_init_val)  # length of amplitude array to be fitted
	flux_sky = kwargs.get('sky_flux', flux_sky_orig)  # Sky Flux
	amp_array, center_array, sigma_array, reddening_val, coeff = get_params(popt, number_of_narrow_components, number_of_wide_components, center_list, amp_length)
	if (coeff==None):
		coeff_val = coeff_init
	else:
		coeff_val = coeff
	#print (amp_array)
	amp_array_rev = retrieve_all_amplitude_list_rev2(list(amp_array), number_of_narrow_components, number_of_wide_components, position_init_narrow_comp, position_init_wide_comp, position_final_narrow_comp, position_final_wide_comp, comments_on_tied, comments_on_Balmer)
	#print (amp_array_rev)
	#print (len(amp_array_rev))
	cont_array = chebyshev_disp(wave_array, coeff_val)
	count_amp = 0
	for j in range(len(center_list)):
		centre = center_list[j]*(1.+redshift_val)
		vel_array = vel_prof(wave_array, centre)
		count = 0
		if (comments_on_Balmer[j]):
			for k in range(number_of_narrow_components):
				group_prof += gaus_prof_vel(vel_array, amp_array_rev[count_amp], center_array[count], sigma_array[count])
				count+=1
				count_amp+=1
			for l in range(number_of_wide_components):
				group_prof += gaus_prof_vel(vel_array, amp_array_rev[count_amp], center_array[count], sigma_array[count])
				count+=1
				count_amp+=1
		else:
			for m in range(number_of_narrow_components):
				#print (count, count_amp)
				group_prof += gaus_prof_vel(vel_array, amp_array_rev[count_amp], center_array[count], sigma_array[count])
				count+=1
				count_amp+=1
	group_prof+=cont_array
	reddening_array = [reddening_val, 1.0, 0.0, 0.0]
	group_prof_adv = func_6(wave_array/(1.+redshift_val), group_prof, *reddening_array)
	group_prof_adv_rev = group_prof_adv + flux_sky
	group_prof_convolved = convolved_prof5(wave_array, group_prof_adv_rev, res)
	return (group_prof_convolved)
##################################GAUSS_BASED_FUNCTIONS##################################


##################################FITTING_FUNCTIONS##################################
def fitting_function_rev(popt_init, datax, datay, yerror, **kwargs):
	center_list = kwargs.get('center_list', center_list_init)  # Centre List
	number_of_narrow_components = kwargs.get('number_of_narrow_components', number_of_narrow_components_init)  # Number of narrow components
	number_of_wide_components = kwargs.get('number_of_wide_components', number_of_wide_components_init)  # Number of wide components
	amp_length = kwargs.get('amp_length', len(amp_init_array))  # length of amplitude array to be fitted
	method_str = kwargs.get('method_str', 'trf')  # Method of fit
	maxfev_val = kwargs.get('maxfev_val', 2000000)  # Maxfev
	fit_continuum = kwargs.get('fit_continuum', True)  # Fit continuum?
	fit_dust = kwargs.get('fit_dust', True)  # Fit dust?
	non_coeff_elements = amp_length + ((number_of_narrow_components+number_of_wide_components)*2)+1
	coeff_len = len(popt_init[non_coeff_elements:])
	if (fit_continuum):
		print ('Fitting with continuum')
		params_new = popt_init
		#print (params_new)
		bounds_lower, bounds_upper = get_bounds(amp_length, number_of_narrow_components, number_of_wide_components, coeff_len, center_list, fit_continuum=True, fit_dust=fit_dust)
		#for i in range(len(params_new)):
			#if not (bounds_lower[i]<=params_new[i]<=bounds_upper[i]):
				#print (i, bounds_lower[i], params_new[i], bounds_upper[i])
		#quit()
		if not (bounds_lower < params_new < bounds_upper):
			params_new = popt_init[0:non_coeff_elements]
			bounds_lower, bounds_upper = get_bounds(amp_length, number_of_narrow_components, number_of_wide_components, coeff_len, center_list, fit_continuum=False, fit_dust=fit_dust)
			print ('Continuum out of bounds')
	else:
		params_new = popt_init[0:non_coeff_elements]
		bounds_lower, bounds_upper = get_bounds(amp_length, number_of_narrow_components, number_of_wide_components, coeff_len, center_list, fit_continuum=False, fit_dust=fit_dust)
		print ('Fitting without continuum')
		
	pfit, pcov = curve_fit(gaus_group_with_cont_rev, datax, datay, p0=params_new, bounds=((bounds_lower), (bounds_upper)), sigma=yerror, maxfev=maxfev_val, method=method_str)
	error = []
	for i in range(len(pfit)):
		try:
			error.append(np.absolute(pcov[i][i])**0.5)
		except:
			error.append(0.00)
	pfit_curvefit = pfit
	perr_curvefit = np.array(error)
	return pfit_curvefit, perr_curvefit
##################################FITTING_FUNCTIONS##################################

#wave_orig, flux_orig, err_orig = np.loadtxt('emission_1d.dat', unpack=True)
print ("Loading data...")
with fits.open(str(muse_data_filename)) as hdul:
#with fits.open('/Users/adarshranjan/Desktop/current_work/sravani_work/new_codes/MRK_463_2019.fits') as hdul:
	header_original = hdul[1].header
	header_original_err = hdul[2].header
	data_original = hdul[1].data
	err_original = np.sqrt(hdul[2].data)

data_original = np.nan_to_num(data_original, nan=1e-6, posinf=1e-6, neginf=-1e-6)
err_original = np.nan_to_num(err_original, nan=1e-6, posinf=1e-6, neginf=1e-6)
mask = np.where(data_original[:,:,:]=='')
data_original[mask] = 1e-6
err_original[mask] = 1e-6
err_original[err_original<=0.]==1e-6

#Flux unit - '10**(-20)*erg/s/cm**2/Angstrom'
#print (data1.shape[0], data1.shape[1], data1.shape[2])
#MAKE 2-D INTEGRATED DATA FROM 3D IFU
data1_new = np.nansum(data_original, axis=0)
#CREATE A WAVELENGTH ARRAY
x_init = header_original['CRVAL1'] + (header_original['CRPIX1']-1 * (header_original['CD1_1']))
x_full = x_init - (np.arange(0, header_original['NAXIS1'], 1)*header_original['CD1_1'])
x_full = np.flip(x_full, axis=0)
y_init = header_original['CRVAL2'] - (header_original['CRPIX2']-1 * (header_original['CD2_2']))
y_full = y_init + (np.arange(0, header_original['NAXIS2'], 1)*header_original['CD2_2'])
muse_wcs = WCS(header_original).celestial
wave_init = header_original['CRVAL3'] - (header_original['CRPIX3']-1 * (header_original['CD3_3']))
wave_full = wave_init + (np.arange(0, header_original['NAXIS3'], 1)*header_original['CD3_3'])
#wave1 = wave_full/(1.+redshift)
wave1 = wave_full
physical_x_axis = x_full
physical_y_axis = y_full
z = 0.0
wave_from_header = wave1
wave_orig = wave1
name_list_sorted = [x for _, x in sorted(zip(center_list_plot, center_list_names_plot))]
center_list_sorted = np.sort(center_list_plot)
flux_orig = data_original[:, x_pos, y_pos]
err_orig = err_original[:, x_pos, y_pos]
if (sky_spectra_file):
	wave_sky_full, flux_sky_full, flux_err_sky_full = np.loadtxt(sky_spectra_file, unpack=True)
else:
	wave_sky_full = wave_orig
	flux_sky_orig = np.zeros([len(wave_orig)])
	flux_err_sky = np.ones([len(wave_orig)])

file1 = [wave_orig, flux_orig, err_orig]
wave, flux, err, FWHM_gal, velscale = get_data_from_file(file1, file_type, require_air_to_vaccum, extra_redshift)
file_sky = [wave_sky_full, flux_sky_full, flux_err_sky_full]
wave_sky, flux_sky, flux_err_sky, FWHM_gal_sky, velscale_sky = get_data_from_file(file_sky, file_type, require_air_to_vaccum, extra_redshift)

global res
res = (c_kms/1.)/FWHM_gal
#cont_init = get_initial_continuum(wave, flux)
#cont_init = get_initial_continuum_revised(wave, flux)
cont_init = get_initial_continuum_rev_2(wave, flux, fwhm_galaxy=FWHM_gal)


center_init_array = np.linspace(-200., 200., number_of_narrow_components_init)
center_init_array = np.append(center_init_array, np.linspace(-200., 200., number_of_wide_components_init))
sigma_init_array = np.full([number_of_narrow_components_init], 100.)
sigma_init_array = np.append(sigma_init_array, np.full([number_of_wide_components_init], 300.))
popt_init, masked_array, amp_init_array, position_init_narrow_comp, position_init_wide_comp, position_final_narrow_comp, position_final_wide_comp, coeff_init_val, index_for_fixing_bounds_factor_init_narrow_comp, index_for_fixing_bounds_factor_init_wide_comp = get_opt_param_array_rev2(wave, flux, cont_init, number_of_narrow_components_init, number_of_wide_components_init, center_list_init, redshift_val, comments_on_Balmer, comments_on_tied, factor_for_tied, e_b_minus_v_init, window_for_fit, stopping_number_for_continuum, center_init_array, sigma_init_array)
continuum = chebyshev_disp(wave[masked_array], coeff_init_val)
flux_sky_orig = flux_sky[masked_array]
#plt.errorbar(wave[masked_array], flux[masked_array], yerr=err[masked_array], color='tab:blue', label='Fitted Region')
'''
pfit_curvefit, perr_curvefit = fitting_function_rev(popt_init, wave[masked_array], flux[masked_array], err[masked_array], number_of_narrow_components=number_of_narrow_components_init, number_of_wide_components=number_of_wide_components_init, fit_continuum=fit_continuum_val, fit_dust=fit_reddening_val)
amp_array, center_array, sigma_array, reddening_val_fit, coeff_fit = get_params(pfit_curvefit, number_of_narrow_components_init, number_of_wide_components_init, center_list_init, len(amp_init_array))
if (reddening_val_fit>0.3):
	print ('fitting suspicious... reverting to fixed continuum and reddening fit')
	pfit_curvefit, perr_curvefit = fitting_function_rev(popt_init, wave[masked_array], flux[masked_array], err[masked_array], number_of_narrow_components=number_of_narrow_components_init, number_of_wide_components=number_of_wide_components_init, fit_continuum=False, fit_dust=False)
'''

pfit_curvefit, perr_curvefit = np.loadtxt('fit_params.dat', unpack=True)
amp_array, center_array, sigma_array, reddening_val_fit, coeff_fit = get_params(pfit_curvefit, number_of_narrow_components_init, number_of_wide_components_init, center_list_init, len(amp_init_array))
#print (amp_array)
reddening_array_fit = [reddening_val_fit, 1.0, 0.0, 0.0]
amp_array_rev = retrieve_all_amplitude_list_rev2(list(amp_array), number_of_narrow_components_init, number_of_wide_components_init, position_init_narrow_comp, position_init_wide_comp, position_final_narrow_comp, position_final_wide_comp, comments_on_tied, comments_on_Balmer)
if (coeff_fit is None):
	coeff_val_fit = coeff_init_val
else:
	coeff_val_fit = coeff_fit

wave_fitted, result_fitted = plot_gaus_group_with_cont_rev(wave[masked_array], *pfit_curvefit)
cont_array_fitted = chebyshev_disp(wave[masked_array], coeff_val_fit)
wave_fit = wave[masked_array]
flux_fit = flux[masked_array]
err_fit = err[masked_array]

f, axarr = plt.subplots(x_slot, y_slot, sharex=False, sharey=False, figsize=((fig_x_size), (fig_y_size)), dpi=dpi_val)
f.subplots_adjust(wspace=0.4, hspace=0.4)
for u in range(x_slot):
	for v in range(y_slot):
		if (len(center_list_sorted) > ((x_slot*v)+u)):
			test_x3 = vel_prof(wave_fit/(1.+redshift_val), center_list_sorted[(x_slot*v)+u])
			idx1 = find_nearest_idx(test_x3,-vel_window-100)
			idx2 = find_nearest_idx(test_x3,vel_window+100)
			if (u>1):
				axarr[u,v].errorbar(wave_fit[idx1:idx2], flux_fit[idx1:idx2], yerr=err_fit[idx1:idx2], color='black', drawstyle='steps-mid', zorder=1)
				axarr[u,v].plot(wave_fit[idx1:idx2], result_fitted[idx1:idx2], color='red', ls='--', zorder=2)
			else:
				axarr[v].errorbar(wave_fit[idx1:idx2], flux_fit[idx1:idx2], yerr=err_fit[idx1:idx2], color='black', drawstyle='steps-mid', zorder=1)
				axarr[v].plot(wave_fit[idx1:idx2], result_fitted[idx1:idx2], color='red', ls='--', zorder=2)

			count_amp = 0
			for z in range(len(center_list_init)):
				centre = center_list_init[z]
				vel_array = vel_prof(wave_fit/(1.+redshift_val), centre)
				count = 0
				if (comments_on_Balmer[z]):
					for k in range(number_of_narrow_components_init):
						#group_prof += gaus_prof_vel(vel_array, amp_array_rev[count_amp], center_array[count], sigma_array[count])
						prof_fit = gaus_prof_vel(vel_array, amp_array_rev[count_amp], center_array[count], sigma_array[count]) + cont_array_fitted
						prof_fit_adv = func_6(wave_fit/(1.+redshift_val), prof_fit, *reddening_array_fit)
						if (u>1):
							axarr[u,v].plot(wave_fit[idx1:idx2]*(1.+redshift_val), prof_fit_adv[idx1:idx2], 'g--')
							axarr[u,v].axvline(wave_prof(center_array[count], centre)*(1.+redshift_val), ls='dashed', color='green', alpha=0.5)
						else:
							axarr[v].plot(wave_fit[idx1:idx2]*(1.+redshift_val), prof_fit_adv[idx1:idx2], 'g--')
							axarr[v].axvline(wave_prof(center_array[count], centre)*(1.+redshift_val), ls='dashed', color='green', alpha=0.5)

						prof_fit = []
						prof_fit_adv = []
						count+=1
						count_amp+=1
					for l in range(number_of_wide_components_init):
						#group_prof += gaus_prof_vel(vel_array, amp_array_rev[count_amp], center_array[count], sigma_array[count])
						prof_fit = gaus_prof_vel(vel_array, amp_array_rev[count_amp], center_array[count], sigma_array[count]) + cont_array_fitted
						prof_fit_adv = func_6(wave_fit/(1.+redshift_val), prof_fit, *reddening_array_fit)
						if (u>1):
							axarr[u,v].plot(wave_fit[idx1:idx2]*(1.+redshift_val), prof_fit_adv[idx1:idx2], 'm--')
							axarr[u,v].axvline(wave_prof(center_array[count], centre)*(1.+redshift_val), ls='dashed', color='magenta', alpha=0.5)
						else:
							axarr[v].plot(wave_fit[idx1:idx2]*(1.+redshift_val), prof_fit_adv[idx1:idx2], 'm--')
							axarr[v].axvline(wave_prof(center_array[count], centre)*(1.+redshift_val), ls='dashed', color='magenta', alpha=0.5)

						prof_fit = []
						prof_fit_adv = []
						count+=1
						count_amp+=1
				else:
					for m in range(number_of_narrow_components_init):
						#group_prof += gaus_prof_vel(vel_array, amp_array_rev[count_amp], center_array[count], sigma_array[count])
						prof_fit = gaus_prof_vel(vel_array, amp_array_rev[count_amp], center_array[count], sigma_array[count]) + cont_array_fitted
						prof_fit_adv = func_6(wave_fit/(1.+redshift_val), prof_fit, *reddening_array_fit)
						if (u>1):
							axarr[u,v].plot(wave_fit[idx1:idx2]*(1.+redshift_val), prof_fit_adv[idx1:idx2], 'g--')
							axarr[u,v].axvline(wave_prof(center_array[count], centre)*(1.+redshift_val), ls='dashed', color='green', alpha=0.5)
						else:
							axarr[v].plot(wave_fit[idx1:idx2]*(1.+redshift_val), prof_fit_adv[idx1:idx2], 'g--')
							axarr[v].axvline(wave_prof(center_array[count], centre)*(1.+redshift_val), ls='dashed', color='green', alpha=0.5)

						prof_fit = []
						prof_fit_adv = []
						count+=1
						count_amp+=1

			if (u>1):
				axarr[u,v].set_title(name_list_sorted[(x_slot*v)+u], fontsize=size_of_font)
				#print (wave_fit[idx1], wave_fit[idx2])
				axarr[u,v].set_xlim(wave_fit[idx1], wave_fit[idx2])
				axarr[u,v].tick_params(axis = 'both', which = 'major', direction='in', length=size_of_font/2, width=2, colors='k')
				axarr[u,v].tick_params(axis = 'both', which = 'minor', direction='in', length=size_of_font/4, width=1, colors='k')
				axarr[u,v].tick_params(axis='both', labelsize=size_of_font)
			else:
				axarr[v].set_title(name_list_sorted[(x_slot*v)+u], fontsize=size_of_font)
				#print (wave_fit[idx1], wave_fit[idx2])
				axarr[v].set_xlim(wave_fit[idx1], wave_fit[idx2])
				axarr[v].tick_params(axis = 'both', which = 'major', direction='in', length=size_of_font/2, width=2, colors='k')
				axarr[v].tick_params(axis = 'both', which = 'minor', direction='in', length=size_of_font/4, width=1, colors='k')
				axarr[v].tick_params(axis='both', labelsize=size_of_font)

		else:
			if (u>1):
				axarr[u,v].set_visible(False)
			else:
				axarr[v].set_visible(False)

f.text(0.48, 0.92, qso_name, fontsize=1.2*size_of_font)
f.text(0.52, 0.06, r'Wavelength ($\rm \AA$)', ha='center', va='center', fontsize=1.2*size_of_font)
f.text(0.06, 0.5, r'Flux (10$^{-20}$ erg s$^{-1}$ cm$^{-2}$ AA$^{-1}$)', ha='center', va='center', rotation='vertical', fontsize=1.2*size_of_font)
f.patch.set_linewidth(2)
f.patch.set_edgecolor('black')
plt.savefig('saved_fig.png', edgecolor=f.get_edgecolor())
print ('saved_fig.png', ' saved...')



quit()



