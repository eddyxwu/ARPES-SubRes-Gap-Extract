from gold_reference import *

norm_state_Z = norm_state_I(X,Y)
add_noise(norm_state_Z)

im = plt.imshow(norm_state_Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)], origin='lower')  # drawing the function
plt.colorbar(im)
plt.show()

# find a, c from trajectory
trajectory = np.zeros(z_width)

if(trajectory.size<=10):
    raise RuntimeError('k width is too small')

lorentz_scale_array=np.zeros(3)
fermi_energy_extraction_stop=0

temp = 0

for slice_k_index in range(z_width):
    norm_state_EDC = np.zeros(z_height)

    # IGNORE NOISY DATA
    norm_state_fit_start_index=-1
    norm_state_fit_end_index=-1

    for i in range(z_height):
        norm_state_EDC[i] = norm_state_Z[i][slice_k_index]
        if norm_state_fit_start_index == -1:
            if norm_state_EDC[i] >= min_fit_count:
                norm_state_fit_start_index = i
        if norm_state_EDC[i] >= min_fit_count:
            norm_state_fit_end_index = i

    # SUFFICIENT ROOM FOR ENERGY CONV
    min_indexes_from_edge = 3 * energy_conv_sigma / w_step
    norm_state_fit_start_index = int(max(norm_state_fit_start_index, round(min_indexes_from_edge)))
    norm_state_fit_end_index = int(min(norm_state_fit_end_index, round(z_height - 1 - min_indexes_from_edge)))
    norm_state_points_in_fit = norm_state_fit_end_index - norm_state_fit_start_index + 1  # include end point

    # LOW NOISE SLICE CREATION
    norm_state_low_noise_slice = np.zeros(norm_state_points_in_fit)
    norm_state_low_noise_w = np.zeros(norm_state_points_in_fit)
    for i in range(norm_state_points_in_fit):
        norm_state_low_noise_slice[i] = norm_state_Z[i + norm_state_fit_start_index][slice_k_index]
        norm_state_low_noise_w[i] = w[i + norm_state_fit_start_index]

    scipy_nofe_params, scipy_nofe_pcov = scipy.optimize.curve_fit(lorentz_form, norm_state_low_noise_w, norm_state_low_noise_slice, maxfev=3000)

    # a is amplitude scale, b is position, c is width
    if slice_k_index<3:
        lorentz_scale_array[slice_k_index] = scipy_nofe_params[0]
    elif (scipy_nofe_params[0] < 0.95 * np.average(lorentz_scale_array)):
        fermi_energy_extraction_stop = slice_k_index
        print("ending at:",slice_k_index)
        break

    # add to trajectory
    trajectory[slice_k_index] = scipy_nofe_params[1]

    temp+=1

reduced_k = np.zeros(fermi_energy_extraction_stop)
reduced_trajectory = np.zeros(fermi_energy_extraction_stop)

for i in range(fermi_energy_extraction_stop):
    reduced_k[i] = k[i]
    reduced_trajectory[i] = trajectory[i]

scipy_trajectory_params, scipy_trajectory_pcov = scipy.optimize.curve_fit(e, reduced_k, reduced_trajectory)

print(scipy_trajectory_params)

extracted_a = scipy_trajectory_params[0]
extracted_c = scipy_trajectory_params[1]