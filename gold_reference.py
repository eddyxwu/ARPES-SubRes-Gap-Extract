import matplotlib.pyplot as plt
import lmfit
import scipy.optimize
import scipy.integrate

from heat_map_setup import *

# 7.91621939e+01 1.34835884e+02 9.99062365e+02 6.46765425e+00
#  2.27918349e-02 1.47047030e+01

a0 = 1786.54730
a1 = 0
b0 = -2773.49092
b1 = 0.19939887
gold_sigma = 0.28934574
f_e = 14125.0469

def DOS_gold(k,w):
    return n_vectorized(w) * (a0 + a1 * w)

def I_gold(k,w):
    return energy_convolution_map(k, w, DOS_gold, R_vectorized)

def background_gold(k,w):
    return b0 + b1 * w

def gold_func1(w, v0, v1):
    return n_vectorized(w) * (v0 + v1 * w)

def for_energy_res(w, a0, a1, b0, b1, energy_res):
    R_temp = partial(R_vectorized, sigma=energy_res)
    func2 = partial(gold_func1, v0=a0, v1=a1)
    return energy_convolution(w, func2, R_temp) + b0 + b1 * w
'''
w_gold = np.arange(-15,15,0.5)
X_gold, Y_gold = np.meshgrid(k, w_gold)
Z_gold = I_gold(X_gold, Y_gold) + background_gold(X_gold, Y_gold)
print(Z_gold.shape)
# add_noise(Z_gold, scaleup = 10)
im = plt.imshow(Z_gold, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w_gold), max(w_gold)])  # drawing the function
plt.colorbar(im)
plt.show()
inv_Z_gold = np.array([list(i) for i in zip(*Z_gold)])
gold_slice = inv_Z_gold[0]
print(gold_slice)
fit_gold_for_energy_res, pcov_gold_for_energy_res = scipy.optimize.curve_fit(for_energy_res, w_gold, gold_slice, bounds=([-np.inf, -np.inf, -np.inf, -np.inf, 0],[np.inf, np.inf, np.inf, np.inf, 50]))
print("energy_res_fit", fit_gold_for_energy_res)
print(np.sqrt(np.diag(pcov_gold_for_energy_res)))
plt.plot(w_gold, gold_slice, label='actual')
plt.plot(w_gold, for_energy_res(w_gold, *fit_gold_for_energy_res), label='fit')
plt.plot(w_gold, for_energy_res(w_gold, a0, a1, b0, b1, energy_conv_sigma), label='force')
plt.legend()
plt.show()
'''
# ===== ===== ===== ===== =====
'''
# FIND GOOD a0, a1, b0, b1 VALUES BASED ON DATA
gold_file = open(r"/Users/ianhu/Downloads/OD50#9_0334.txt", "r")
gold_data = np.zeros((101,700))
k_gold = np.linspace(-16.48000, 15.47429, num=700) # in degrees
w_gold = np.linspace(14072.0, 14172.0, num=101) # in mev
# get to [Data 1] line
while gold_file.readline() != '[Data 1]\n':
    pass
for i in range(101):
    temp_k_slice = gold_file.readline()
    temp_split_array = temp_k_slice.split()
    for j in range(700):
        gold_data[101-i-1][j] = temp_split_array[j+1] # ignore first
# plot gold reference
# plt.set_title("Raw gold data")
im = plt.imshow(gold_data, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k_gold), max(k_gold), min(w_gold), max(w_gold)])  # drawing the function
plt.colorbar(im)
plt.show()
# get compressed averages
avg_gold_slice = np.zeros(101)
for i in range(101):
    for j in range(350,355):
        avg_gold_slice[i]+=gold_data[101-i-1][j]/5
gold_file.close()
def gold_func(w, v0, v1, v2, v3, sigma, fermi_energy):
    R_partial = partial(R_vectorized, sigma=sigma)
    gold_func2 = partial(gold_func1, v0=v0, v1=v1)
    # return func1(w, v0, v1) + background_gold(w, v2, v3)
    return energy_convolution(w-fermi_energy, gold_func2, R_partial) + v2 + v3 * w
# n_vectorized(w) * (a0 + a1 * w)
# b0 + b1 * w
gold_sigma = np.sqrt(avg_gold_slice)
for i in range(10):
    gold_sigma[i] = 999999999
    gold_sigma[101-i-1] = 999999999
# print(w_gold)
# print(avg_gold_slice)
#fit to selected points
fit_gold, pcov_gold = scipy.optimize.curve_fit(gold_func, w_gold, avg_gold_slice,\
    maxfev=3000, bounds=([1700, -25, -4000, -1, 0, 14110], [2100, 0, 0, 1, 6, 14140]),\
    sigma = gold_sigma)
print("FIT FOR GOLD BASED ON DATA")
print('a0, a1, b0, b1, sigma, f_e')
print(fit_gold)
print(np.sqrt(np.diag(pcov_gold)))
pars = lmfit.Parameters()
pars.add_many(('a0', fit_gold[0], True, 1700, 2100), ('a1', fit_gold[1], True, -25, 0),\
    ('b0', fit_gold[2], True, -3000, 1500), ('b1', fit_gold[3], True, -1, 1),\
    ('sigma', fit_gold[4], True, 0, 6), ('f_e', fit_gold[5], True, 14110, 14140))
def gold_residual(p):
    return gold_func(w_gold, p['a0'], p['a1'], p['b0'], p['b1'], p['sigma'], p['f_e']) - avg_gold_slice
gold_mini = lmfit.Minimizer(gold_residual, pars, nan_policy='propagate', calc_covar=True)
gold_out1 = gold_mini.minimize(method='nelder')
gold_kwargs = {"sigma": gold_sigma}
gold_out2 = gold_mini.minimize(method='leastsq', params=gold_out1.params, args=gold_kwargs)
print(lmfit.fit_report(gold_out2.params))
plt.plot(w_gold, gold_func(w_gold, gold_out2.params.get('a0').value, \
    gold_out2.params.get('a1').value, gold_out2.params.get('b0').value, \
    gold_out2.params.get('b1').value, gold_out2.params.get('sigma').value, \
    gold_out2.params.get('f_e').value), label='fitted lmfit')
plt.plot(w_gold, avg_gold_slice)
plt.plot(w_gold, gold_func(w_gold, *fit_gold), label='fitted scipy')
plt.legend()
plt.show()
'''


'''
# TO CREATE OVERLAPPING NORMALIZED GAUSSIAN PEAKS
plt.figure(figsize=(8, 4.8), dpi=320) # 6.4, 4.8
temp_w = np.arange(-10, 7, 0.1)
temp_std = 1.5
plt.plot(temp_w, gaussian_form_normalized(temp_w, temp_std, -1), label='Gaussian (sigma=1.5, mu = -1)')
plt.plot(temp_w, gaussian_form_normalized(temp_w, temp_std, 1), label='Gaussian (sigma=1.5, mu = 1)')
plt.plot(temp_w, gaussian_form_normalized(temp_w, temp_std, -1)+gaussian_form_normalized(temp_w, temp_std, 1), label='Sum of blue and orange')
temp_params, temp_pcov = scipy.optimize.curve_fit(gaussian_form, temp_w, gaussian_form_normalized(temp_w, temp_std, -1)+gaussian_form_normalized(temp_w, temp_std, 1))
plt.plot(temp_w, gaussian_form(temp_w, *temp_params), linestyle='dashdot', label='Single Gaussian fitted')
print(temp_params)
plt.legend()
plt.title('Overlapping Normalized Gaussian Peaks')
# plt.tight_layout()
plt.savefig('test_export.svg', format='svg')
'''
'''
# TO CREATE SPECTRUM
plt.figure(figsize=(6.4, 4.8), dpi=320) # 6.4, 4.8
X, Y = np.meshgrid(k, w)
# Z = A_BCS(X,Y)
# plt.title('Spectral Function')
# Z = Io_n_A_BCS(X, Y)
# plt.title('Add Fermi-Dirac Distribution')
Z = I(X,Y)
# plt.title('Apply an Energy Convolution')
add_noise(Z)
# plt.title('Add Poisson Noise')
im = plt.imshow(Z, cmap=plt.cm.RdBu, aspect='auto', extent=[min(k), max(k), min(w), max(w)], origin='lower')  # drawing the function
# plt.colorbar(im)
plt.xlabel('k ($A^{-1}$)')
plt.ylabel('w (mev)')
# plt.savefig('Poisson Noise.svg', format='svg')
short_k = np.linspace(max(0,k_as_index(fit_start_k)), k_as_index(kf), 6)
short_k = short_k.astype(int)
short_k = np.unique(short_k)
short_k = np.flip(short_k, 0) # try to fit starting from fermi slice
print(short_k)
plt.title('EDC Selection')
for index in short_k:
    if index==16:
        plt.vlines(k[index], -75,75, color='gold')
        continue
    plt.vlines(k[index], -75, 75, color='black')
plt.savefig('(10) EDC Selection.svg', format='svg')
# plt.show()
quit()
'''


import mpmath as mp

def forward_transformation(z):
    return 2*np.sqrt(z+0.375)

def asympt_inverse_transformation(D):
    return D*D/4-0.125

def inverse_transformation(y):
    return 2*mp.nsum(lambda z: mp.sqrt(z+0.375)*mp.power(y,z)*mp.exp(-y)/mp.fac(z),[0,mp.inf])
'''
print(forward_transformation(600))
print(asympt_inverse_transformation(forward_transformation(600)))
s = np.random.poisson(25, 10000)
s = forward_transformation(s)
count, bins, ignored = plt.hist(s, np.arange(0, 15, 0.05), density=True)
# count, bins, ignored = plt.hist(s, np.arange(0, 50, 1), density=True)
mu, sigma = scipy.stats.norm.fit(s)
print(mu)
print(sigma)
print(asympt_inverse_transformation(mu))
best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
plt.plot(bins,best_fit_line*4)
print(best_fit_line)
'''

'''
def test_parabola(x,a,c):
    return a*x*x+c
def median_noise_removal(array, radius=1):
    ret = np.zeros(array.size)
    for i in range(radius):
        ret[i] = array[i]
        ret[array.size-1-i] = array[array.size-1-i]
    for i in range(radius,array.size-radius):
        nearby = np.array([array[i]])
        for j in range(1, radius+1):
            nearby = np.append(nearby, [[array[i-j]], [array[i+j]]])
        nearby.sort()
        ret[i] = nearby[radius]
    return ret
def mean_noise_removal(array, radius=1):
    ret = np.zeros(array.size)
    for i in range(radius):
        ret[i] = array[i]
        ret[array.size-1-i] = array[array.size-1-i]
    for i in range(radius,array.size-radius):
        nearby_avg = array[i]
        for j in range(1, radius+1):
            nearby_avg+=array[i-j]+array[i+j]
        nearby_avg = nearby_avg / (2*radius+1)
        ret[i] = nearby_avg
    return ret
num_trials = 1000
norm_avg_a = 0
norm_avg_b = 0
norm_avg_c = 0
norm_avg_d = 0
norm_werr_avg_a = 0
norm_werr_avg_b = 0
norm_werr_avg_c = 0
norm_werr_avg_d = 0
alt_avg_a = 0
alt_avg_b = 0
alt_avg_c = 0
alt_avg_d = 0
act_a = 30
act_b = 0
act_c = 1
act_d = 64
def flat_line(x, d):
    return d + x - x
for k in range(num_trials):
    test_x = np.arange(-3.5, 3.5, 0.1) # a = 30, (-3.5, 3.5, 0.3)
    test_y = lorentz_form(test_x,act_a,act_b,act_c)
    # test_y = flat_line(test_x, act_d)
    test_y = np.random.poisson(test_y)
    plt.plot(test_x, test_y)
    norm_params, norm_covar = scipy.optimize.curve_fit(lorentz_form, test_x, test_y)
    no_zero_test_y = test_y
    for i in range(no_zero_test_y.size):
        if no_zero_test_y[i] == 0:
            no_zero_test_y[i] = 1
    norm_werr_params, norm_werr_covar = scipy.optimize.curve_fit(lorentz_form, test_x, test_y, sigma=np.sqrt(no_zero_test_y))
    # print(test_params)
    norm_avg_a += (norm_params[0]-act_a) ** 2
    norm_avg_b += (norm_params[1]-act_b) ** 2
    norm_avg_c += (norm_params[2]-act_c) ** 2
    # norm_avg_d += (norm_params[0] - act_d) ** 2
    norm_werr_avg_a += (norm_werr_params[0]-act_a) ** 2
    norm_werr_avg_b += (norm_werr_params[1]-act_b) ** 2
    norm_werr_avg_c += (norm_werr_params[2]-act_c) ** 2
    # norm_werr_avg_d += (norm_werr_params[0] - act_d) ** 2
    # FORWARD TRANSFORMED
    alt_test_y = forward_transformation(test_y)
    # plt.plot(test_x, alt_test_y, label='forward transformed')
    # NOISE REMOVAL
    alt_test_y = median_noise_removal(alt_test_y,3)
    alt_test_y = mean_noise_removal(alt_test_y,3)
    # plt.plot(test_x, alt_test_y, label='noise removed')
    # RETURN BACK
    alt_test_y = asympt_inverse_transformation(alt_test_y)
    # plt.plot(test_x, alt_test_y, label='return')
    alt_params, alt_covar = scipy.optimize.curve_fit(lorentz_form, test_x, alt_test_y, maxfev = 1500)
    alt_avg_a += (alt_params[0]-act_a) ** 2
    alt_avg_b += (alt_params[1]-act_b) ** 2
    alt_avg_c += (alt_params[2]-act_c) ** 2
    # alt_avg_d += (alt_params[0] - act_d) ** 2
norm_avg_a/=num_trials
norm_avg_b/=num_trials
norm_avg_c/=num_trials
norm_avg_d/=num_trials
norm_werr_avg_a/=num_trials
norm_werr_avg_b/=num_trials
norm_werr_avg_c/=num_trials
norm_werr_avg_d/=num_trials
alt_avg_a/=num_trials
alt_avg_b/=num_trials
alt_avg_c/=num_trials
alt_avg_d/=num_trials
print("norm: ", norm_avg_a, norm_avg_b, norm_avg_c)
# print("norm: ", norm_avg_d)
print("norm_werr: ", norm_werr_avg_a, norm_werr_avg_b, norm_werr_avg_c)
# print("norm_werr: ", norm_werr_avg_d)
print("alt: ", alt_avg_a, alt_avg_b, alt_avg_c)
# print("alt: ", alt_avg_d)
# some sort of gaussian smoothing --> return --> then fit
# fit --> map return
plt.legend()
plt.show()
quit()
'''