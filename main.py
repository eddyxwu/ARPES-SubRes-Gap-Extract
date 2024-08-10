##################################################
# IMPORT LIBRARIES
##################################################
import scipy.optimize
import scipy.integrate

from control_panel import dk
from kf_from_norm_state import *
##################################################
# Z HEAT MAP INFO
##################################################

print("kf:", kf)
print("energy_conv_sigma:", energy_conv_sigma)
print("dk:", dk)
print("T:", T)
print(k_as_index(kf))

##################################################
# EXTRACT GAP - TRAJECTORY V2
##################################################

# NORMAN TECHNIQUE

# requires knowledge of temperature and energy resolution --> should test resilience of these
# light dependence on kf? but depends on accuracy of knowledge of k (no k convolution)

def energy_conv_integrand(integration_w, fixed_w, T, dk, a, c, fixed_k):
    return A_BCS(fixed_k, integration_w, a, c, dk, T) * R(math.fabs(integration_w - fixed_w), energy_conv_sigma) * n(integration_w)

def spectrum_slice_array(w_array, scale, T, dk, a, c, fixed_k):
    return_array = np.zeros(w_array.size)
    for i in range(w_array.size):
        # return_array[i] = scale*A_BCS(slice_k, w_array[i], a, c, dk, T) * n(w_array[i])
        return_array[i] = scale*scipy.integrate.quad(energy_conv_integrand, w_array[i] - 250, w_array[i] + 250, args=(w_array[i], T, dk, a, c, fixed_k))[0]
    return return_array

def spectrum_slice_array_alt(w_array, scale, T, a, c, dk, fixed_k):
    return spectrum_slice_array(w_array, scale, T, dk, a, c, fixed_k)

# short_k = np.arange(math.floor(k_as_index(fit_start_k)),k_as_index(kf)+1, 1) # all indexes
short_k = np.linspace(max(0,k_as_index(fit_start_k)), k_as_index(kf), 6)
short_k = short_k.astype(int)
short_k = np.unique(short_k)
short_k = np.flip(short_k, 0) # try to fit starting from fermi slice
print(short_k)
curr_index=0

last_dk = 1
last_scale = scaleup_factor/10
last_T = 1

norm_state_last_scale = scaleup_factor/10
norm_state_last_T = 1
norm_state_last_a = 1
norm_state_last_c = -1

easy_print_array = []

# set up multiple subplots
num_plots = short_k.size + 1
plt.figure(figsize=(6 * num_plots, 6), dpi=120)
first_plot = plt.subplot(1, num_plots, 1)
first_plot.set_title("Spectrum (dk=" + str(dk)+")")
im = first_plot.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)], origin='lower')  # drawing the function
plt.colorbar(im)
plt.xlabel('k ($A^{-1}$)')
plt.ylabel('w (mev)')
# /Users/ianhu/PycharmProjects/ARPES-SubRes-Gap-Extract
# np.savetxt("/Users/ianhu/Documents/ARPES CNN/Dataset 1 - c=-1000, sigma=15/"+str(round(dk,6))+".csv", Z, delimiter=",")

plt_index=2 # first plot is for spectrum

for slice_k_index in short_k:
    Norman_subplot = plt.subplot(1, num_plots, plt_index)
    plt_index+=1
    print('==============================================')
    print('slice_k_index: ' + str(slice_k_index) + ' (' + str(k[slice_k_index]) + ')')
    print('progress: ' + str(curr_index / short_k.size))
    EDC = np.zeros(z_height)
    curr_k = k[slice_k_index]

    # IGNORE NOISY DATA
    fit_start_index=-1
    fit_end_index=-1

    for i in range(z_height):
        EDC[i] = Z[i][slice_k_index]
        if fit_start_index == -1:
            if EDC[i] >= min_fit_count:
                fit_start_index=i
        if EDC[i] >= min_fit_count:
            fit_end_index=i

    # SUFFICIENT ROOM FOR ENERGY CONV
    min_indexes_from_edge = 3*energy_conv_sigma/w_step
    fit_start_index = int(max(fit_start_index, round(min_indexes_from_edge)))
    fit_end_index = int(min(fit_end_index, round(z_height-1-min_indexes_from_edge)))
    points_in_fit = fit_end_index-fit_start_index+1 # include end point

    # LOW NOISE SLICE CREATION
    low_noise_slice = np.zeros(points_in_fit)
    low_noise_w = np.zeros(points_in_fit)
    for i in range(points_in_fit):
        low_noise_slice[i]=Z[i+fit_start_index][slice_k_index]
        low_noise_w[i] = w[i + fit_start_index]

    # FUNCTION TO FIT
    fit_func_w_scale_T_dk = partial(spectrum_slice_array, a=a, c=c, fixed_k=curr_k)
    fit_func_w_scale_T = partial(spectrum_slice_array, dk = 0, a=a, c=c, fixed_k=curr_k)

    nofe_fit_func_w_scale_T_dk = partial(spectrum_slice_array, a=extracted_a, c=extracted_c, fixed_k=curr_k)

    # ===== ===== ===== SCIPY ===== ===== =====
    scipy_full_params, scipy_full_pcov = scipy.optimize.curve_fit(fit_func_w_scale_T_dk, low_noise_w, low_noise_slice, p0=[last_scale, last_T, last_dk], maxfev=2000, bounds=([scaleup_factor / 10, 0., 0.], [scaleup_factor * 10, 50., 50.]), sigma=np.sqrt(low_noise_slice))
    scipy_red_params, scipy_red_pcov = scipy.optimize.curve_fit(fit_func_w_scale_T, low_noise_w, low_noise_slice, p0=[last_scale, last_T], maxfev=2000, bounds=([scaleup_factor / 10, 0.], [scaleup_factor * 10, 50.]), sigma=np.sqrt(low_noise_slice))

    nofe_scipy_full_params, nofe_scipy_full_pcov = scipy.optimize.curve_fit(nofe_fit_func_w_scale_T_dk, low_noise_w, low_noise_slice, p0=[last_scale, last_T, last_dk], maxfev=2000, bounds=([scaleup_factor / 10, 0., 0.], [scaleup_factor * 10, 50., 50.]), sigma=np.sqrt(low_noise_slice))

    last_scale = scipy_full_params[0]
    last_T = scipy_full_params[1]
    last_dk = scipy_full_params[2]

    print("scipy full params: ", scipy_full_params)
    scipy_std_err = math.sqrt(scipy_full_pcov[2][2] / math.sqrt(points_in_fit))
    print("dk stderr +/- : " + str(scipy_std_err) + " (" + str(100 * scipy_std_err / last_dk) + "%)")
    print("scipy full pcov: \n", scipy_full_pcov)

    print("nofe scipy full params: ", nofe_scipy_full_params)

    # SCIPY STATISTICAL RESULTS
    print("DOF: ", points_in_fit-3)
    print("scipy redchi:", manualRedChi(\
        low_noise_slice, fit_func_w_scale_T_dk(low_noise_w, *scipy_full_params), \
        low_noise_slice, points_in_fit - 3))
    print("perfect redchi:", manualRedChi(\
        low_noise_slice, fit_func_w_scale_T_dk(low_noise_w, scaleup_factor / w_step, T, dk), \
        low_noise_slice, points_in_fit - 3))

    scipy_f_stat = manualFTest(\
        low_noise_slice, fit_func_w_scale_T(low_noise_w, *scipy_red_params), 2, \
        fit_func_w_scale_T_dk(low_noise_w, *scipy_full_params), 3,
        low_noise_slice, points_in_fit
        )
    print("f test value:", scipy_f_stat)
    print("p-value (of gap): ", scipy.stats.f.cdf(scipy_f_stat, 3-2, points_in_fit-3))

    # SCIPY PLOT
    plt.plot(w, fit_func_w_scale_T_dk(w, *scipy_full_params), label='Fitted curve')
    plt.plot(w, fit_func_w_scale_T_dk(w, scaleup_factor / w_step, T, dk), label='Perfect fit')

    # DATA/REFERENCE PLOTS
    plt.plot(w, EDC, label='Data')

    # ===== ===== ===== LMFIT ===== ===== =====
    '''
    pars = lmfit.Parameters()
    pars.add('scale', value=last_scale, min=scaleup_factor / 10, max=scaleup_factor * 10)
    pars.add('T', value=last_T, min=0, max=50)
    pars.add('dk', value=last_dk, min=0, max=50)
    # pars.add('scale', value=last_scale)
    # pars.add('T', value=last_T)
    # pars.add('dk', value=last_dk)
    def residual(p):
        return fit_func_w_dk_scale_T(low_noise_selected_w, p['scale'], p['T'], p['dk']) - low_noise_slice
    mini = lmfit.Minimizer(residual, pars, nan_policy='propagate', calc_covar=True)
    out1 = mini.minimize(method='nelder')
    kwargs = {"sigma": np.sqrt(low_noise_slice)}
    result = mini.minimize(method='leastsq', params=out1.params, args=kwargs)
    lmfit_scale = result.params.get('scale').value
    lmfit_T = result.params.get('T').value
    lmfit_dk = result.params.get('dk').value
    print("\n", lmfit.fit_report(result.params))
    try:
        print(result.covar)
    except:
        print('no covariance matrix')
    # lmfit statistical results
    print("manual redchi:", \
        manualRedChi(low_noise_slice, \
            fit_func_w_dk_scale_T(low_noise_selected_w, lmfit_scale, lmfit_T, lmfit_dk), \
            low_noise_slice,\
            points_in_fit - 3))
    # abbreviated --> don't redo lmfit for reduced model
    lmfit_f_stat = manualFTest( \
        low_noise_slice, fit_func_w_scale_T(low_noise_selected_w, *scipy_red_params), 2, \
        fit_func_w_dk_scale_T(low_noise_selected_w, lmfit_scale, lmfit_T, lmfit_dk), 3,
        low_noise_slice, points_in_fit
    )
    print("f test value:", lmfit_f_stat)
    print("p-value (of gap): ", scipy.stats.f.cdf(lmfit_f_stat, 3 - 2, points_in_fit - 3))
    # lmfit plot
    plt.plot(w, fit_func_w_dk_scale_T(w, lmfit_scale, lmfit_T, lmfit_dk), label='lmfit')
    '''
    # plot

    Norman_subplot.set_title("k ($A^{-1}$): " + str(round(k[slice_k_index], 3)) + " | dk estimate:" + str(round(last_dk, 2)))
    plt.vlines(w[fit_start_index], 0, min_fit_count, color='black')
    plt.vlines(w[fit_end_index], 0, min_fit_count, color='black')
    plt.xlabel('w (mev)')
    plt.ylabel('counts')
    plt.legend()
    curr_index += 1

    if slice_k_index == 16:
        Norman_subplot.set_title("k ($A^{-1}$): " + str(round(k[slice_k_index],3)) + "(kf) | dk estimate:" + str(round(last_dk, 2)), color='olive')

    # EASY_PRINT_ARRAY
    easy_print_array.append(last_dk)
    easy_print_array.append(scipy_std_err)
    easy_print_array.append(manualRedChi( \
        low_noise_slice, fit_func_w_scale_T_dk(low_noise_w, *scipy_full_params), \
        low_noise_slice, points_in_fit - 3))
    easy_print_array.append(manualRedChi( \
        low_noise_slice, fit_func_w_scale_T_dk(low_noise_w, scaleup_factor / w_step, T, dk), \
        low_noise_slice, points_in_fit - 3))
    easy_print_array.append(scipy.stats.f.cdf(scipy_f_stat, 3-2, points_in_fit-3))

##################################################
# SHOW PLOT
##################################################
plt.tight_layout()
# plt.savefig('(11) Fitting EDCs.svg', format='svg')
plt.show()
print('\n\n')
for element in easy_print_array:
    print(element)