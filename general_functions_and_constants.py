##################################################
# GENERAL FUNCTIONS
##################################################

import numpy as np
import math

from control_panel import *

# constants
hbar = 6.582 * 10 ** (-13) # meV*s
mass_electron = 5.1100 * 10 ** 8 # mev/c^2
speed_light = 2.998 * 10 ** 8 # m/s

# gaussian function
def R(dw, sigma):
    # if (0.5 * (dw) ** 2 / sigma ** 2 > 100): return 0
    return (1 / sigma / math.sqrt(2 * math.pi)) * math.exp(-0.5 * (dw) ** 2 / sigma ** 2)
R_vectorized = np.vectorize(R, excluded=['sigma'])

# energy convolution
def energy_convolution_map(k, w, func_k_w, func2_w, scaleup = scaleup_factor):
    height = math.floor(k.size / k[0].size)
    width = k[0].size

    results = np.zeros((height, width))

    # === to test w/out convolution === #
    '''
    for i in range(height):
        for j in range(width):
            results[i][j] = scaleup * func_k_w(k[i][j], w[i][j])
    return results
    '''
    # === to test w/out convolution === #

    inv_k = np.array([list(i) for i in zip(*k)])  # extract vertical arrays from 2d array --> convolute over w
    inv_w = np.array([list(i) for i in zip(*w)])
    rev_inv_w = np.flip(inv_w)  # flip vertically

    for i in range(height):
        for j in range(width):
            curr_w = np.full(inv_w[j].size, w[i][j])  # w at point (to find dw)
            res = np.convolve(scaleup * func_k_w(inv_k[j], inv_w[j]), func2_w(rev_inv_w[j] - curr_w, energy_conv_sigma), mode='valid')
            results[i][j] = res

    return results

def energy_convolution(w, func1, func2):

    results = np.zeros(w.size)

    rev_w = np.flip(w)  # flip vertically

    for i in range(w.size):
        curr_w = np.full(w.size, w[i])
        res = np.convolve(func1(w), func2(rev_w-curr_w), mode='valid')
        results[i] = res

    return results
'''
# scaleup intensity
def scale_up(map, scaleup = scaleup_factor):
    height = math.floor(map.size / map[0].size)
    width = map[0].size
    for i in range(height):
        for j in range(width):
            map[i][j] = map[i][j] * scaleup
'''
# add noise
def add_noise(map, scaleup=scaleup_factor): # counts typically from 200 to 1000 to 10000
    height = math.floor(map.size/map[0].size)
    width = map[0].size
    for i in range(height):
        for j in range(width):

            if map[i][j] < 1:
                map[i][j] = 1
            else:
                map[i][j] = np.random.poisson(map[i][j])
                if map[i][j] == 0:
                    map[i][j] = 1

            # ensure no 0's for error sigma (for fitting)
            if map[i][j] < 1:
                map[i][j] = 1


            # Gaussian noise: map[i][j] += noise_percentage * random.gauss(0, math.sqrt(map[i][j]))
    return

# fermi-dirac function
def n(w, temp=60):
    kB = 8.617 * 10 ** (-2)  # Boltzmann's constant (mev/K)
    # h-bar: 6.582 * 10 ** (-13) (mev*s) # implicit bc already expressing energy
    uP = 0
    # if w > 150: return 0
    # if w < -150: return 1
    return 1 / (math.exp((w - uP)/kB/temp) + 1)
n_vectorized = np.vectorize(n)

# remove fermi effects
def remove_n(energy_array, intensity_array):
    result = np.zeros(intensity_array.size)
    for i in range(intensity_array.size):
        result[i] = intensity_array[i] / n(energy_array[i])
    return result

def manualRedChi(data, fit, absSigmaSquared, DOF=1):
    res = 0
    for i in range(data.size):
        res += (data[i]-fit[i]) ** 2 / absSigmaSquared[i]
    return res/DOF

def manualFTest(data, fit1, para1, fit2, para2, absSigmaSquared, n):
    # fit1 should be 'nested' within fit2
    if(para2<=para1):
        return ValueError
    chi1 = manualRedChi(data, fit1, absSigmaSquared)
    chi2 = manualRedChi(data, fit2, absSigmaSquared)
    return ((chi1-chi2)/(para2-para1))/(chi2/(n-para2))

def gaussian_form_normalized(x, sigma, mu):
    return 1 / (sigma * (2 * math.pi) ** 0.5) * math.e ** (-0.5 * ((x-mu)/sigma) ** 2)

def gaussian_form(x, a, b, c):
    return a * math.e ** ((- (x-b) ** 2 ) / (2 * c ** 2))

def lorentz_form(x, a, b, c):
    return a * c / ((x-b) ** 2 + c ** 2)

def no_shift_parabola_form(x, a, c):
    return a * x ** 2 + c