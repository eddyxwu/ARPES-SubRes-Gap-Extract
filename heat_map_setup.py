from general_functions_and_constants import *
from functools import *

##################################################
# DISPERSION FUNCTIONS
##################################################

# normal-state dispersion
lattice_a = 4 # 2-5 angstrom (mainly 3.6-4.2)
brillouin_k = math.pi / lattice_a # angstrom || # hbar * math.pi / lattice_a # g*m/s
fermi_k = 0.5092958179 * brillouin_k # 0.5092958179 for 1000
# overwrite fermi_k
fermi_k = 0.4

c = -1000 # min(-50*dk, -50) # -1000 to -10,000 mev
a = -c / (fermi_k ** 2) # controlled by lattice_a and c

print("c, a:",c, a)

# fermi momentum
kf = fermi_k # math.fabs(c/a) ** 0.5

# ------------------------------------------------
w_step = 1.5
w = np.arange(-115, 50, w_step)

# w: (-45, 20), k: (-0.0125, 0.125)
# w: (-100, 40), k: (-0.03, 0.025)
# w: (-400, 0), k: (-0.05, 0.025)

def w_as_index(input_w):
    return int(round((input_w-min(w)) / (max(w)-min(w)) * (w.size-1)))

d_theta = 0.045 * math.pi / 180 # 0.1-0.2 degrees  # from 0.045
k_step = (1 / hbar) * math.sqrt(2 * mass_electron / speed_light / speed_light * (6176.5840329647)) * d_theta / (10 ** 10)
k = np.arange(fermi_k - 0.04 * fermi_k, fermi_k + 0.04 * fermi_k, k_step)
print("k_step: " + str(k_step) + " | mink: " + str(min(k)) + " | maxk: "  + str(max(k)) + " | #steps: " + str(k.size))

def k_as_index(input_k):
    return int(round((input_k-min(k)) / (max(k)-min(k)) * (k.size-1)))
# ------------------------------------------------

def e(k, a, c):
    return a * k ** 2 + c

# band dispersion of BQPs
def E(k, a, c, dk):
    return (e(k, a, c) ** 2 + dk ** 2) ** 0.5

# coherence factors (relative intensity of BQP bands above and below EF)
def u(k, a, c, dk):
    if (dk==0):
        if (a * k ** 2 + c > 0): return 1
        elif (a * k ** 2 + c < 0): return 0
        else: return 0.5
    return 0.5 * (1 + e(k, a, c) / E(k, a, c, dk))
u_vectorized = np.vectorize(u)

def v(k, a, c, dk):
    return 1 - u_vectorized(k, a, c, dk)

# linewidth broadening due to the finite lifetime of photoholes
T = 5 # 22mev? (5-10mev)

# BCS spectral function
def A_BCS(k, w, a, c, dk, T): # from (https://arxiv.org/pdf/cond-mat/0304505.pdf) (non-constant gap)
    return (1 / math.pi) * (u_vectorized(k, a, c, dk) * T / ((w - E(k, a, c, dk)) ** 2 + T ** 2) + v(k, a, c, dk) * T / ((w + E(k, a, c, dk)) ** 2 + T ** 2))
    # return (1 / math.pi) * (u(k) * T / ((w - E(k, dk)) ** 2 + T ** 2) + v(k) * T / ((w + E(k, dk)) ** 2 + T ** 2))
def A_BCS_2(k,w, dk=dk, T=T): # from (http://ex7.iphy.ac.cn/downfile/32_PRB_57_R11093.pdf)
    return T / math.pi / ((w - e(k) - (dk ** 2)/(w + e(k))) ** 2 + T ** 2)

# intensity pre-factor
def Io(k):
    return 1;

# composition function, using directly
def Io_n_A_BCS(k,w):
    return Io(k)*n_vectorized(w)*A_BCS(k,w,a,c,dk,T)

def Io_n_A_BCS_2(k, w):
    return Io(k)*n_vectorized(w)*A_BCS_2(k, w)

# intensity
def I(k,w):
    return energy_convolution_map(k, w, Io_n_A_BCS, R_vectorized)

def norm_state_Io_n_A_BCS(k,w):
    return Io(k)*n_vectorized(w)*A_BCS(k,w,a,c,0,T)
def norm_state_I(k,w):
    return energy_convolution_map(k,w, norm_state_Io_n_A_BCS, R_vectorized)

# how far left to shift fit
left_shift_mult = 2
# fit_start_k = math.sqrt((-math.sqrt(left_shift_mult ** 2 - 1) * dk - c)/a)
# print(fit_start_k)
fit_start_k = 0
if a!=0:
    fit_start_k = math.sqrt((-left_shift_mult*dk-c)/a)
print('fit_start_k (not indexed): ', fit_start_k)

##################################################
# HEAT MAP
##################################################

X, Y = np.meshgrid(k, w)

Z = I(X,Y)
print("Z.size:", Z.shape, "\n")
add_noise(Z)

z_width = Z[0].size
z_height = int(Z.size / z_width)
kf_index = k_as_index(kf)