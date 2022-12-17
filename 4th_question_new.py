import numpy as np
import scipy as sc
from scipy import signal
import math
import cmath
import matplotlib.pyplot as plt

# plt.rcParams['text.usetex'] = True

w_s = 800
w_p = [100, 200]
w_stop = [110, 190]

# calculate the analog frequencies
w_stop = np.multiply(w_stop, 2 * math.pi / w_s)
w_p = np.multiply(w_p, 2 * math.pi / w_s)
w_stop_analog = np.multiply(np.tan(np.multiply(w_stop, 0.5)), 2)
w_p_analog = np.multiply(np.tan(np.multiply(w_p, 0.5)), 2)
w_0_analog = math.sqrt(w_p_analog[1] * w_p_analog[0])

B = w_p_analog[1] - w_p_analog[0]

# Calculating the ripples in the linear scale
# ripple values in dB
stop_ripple_db = 40
pass_ripple_db = 30
# converting the ripple values in linear scale
stop_ripple = 10 ** (-stop_ripple_db / 20)
pass_ripple = 10 ** (-pass_ripple_db / 20)

# analog low pass filter specifications

w_p_lpf = 1
num1 = ((w_p_analog[1] - w_p_analog[0]) * w_stop_analog[0])
dem1 = ( (w_p_analog[1] * w_p_analog[0]) - w_stop_analog[0]**2)
w_stop_lpf_1 =  num1/dem1

num2 = ((w_p_analog[1] - w_p_analog[0]) * w_stop_analog[1])
dem2 = (w_stop_analog[1] ** 2 - (w_p_analog[1] * w_p_analog[0]))
w_stop_lpf_2 = num2/dem2

w_stop_lpf = min(w_stop_lpf_2, w_stop_lpf_1)

# calculating the order of the filter


epsilon = math.sqrt(1 / (1 - pass_ripple) ** 2 - 1)

N = math.ceil(math.log((math.sqrt(1 - stop_ripple ** 2) + math.sqrt(
    1 - stop_ripple ** 2 * (1 + epsilon ** 2))) / stop_ripple / epsilon) / math.log(
    w_stop_lpf / w_p_lpf + math.sqrt(w_stop_lpf ** 2 / w_p_lpf ** 2 - 1)))

rp = -20 * math.log10(1 - pass_ripple)
b_lpf, a_lpf = signal.cheby1(N, rp, w_p_lpf, analog=True)

# Magnitude response (linear scale) of the initial low-pass filter
w_lpf, h_lpf = signal.freqs(b_lpf, a_lpf, worN=1000)
plt.plot(w_lpf, abs(h_lpf))
plt.axis([0, 10, 0, 1.1])
plt.xlabel('w')
plt.ylabel('H(w)')
plt.xticks(np.arange(0, 11, 1))
plt.yticks(np.arange(0, 1.25, 0.25))
plt.title('Magnitude response of the initial low-pass filter in linear scale')
plt.grid(linestyle='dashed')
plt.show()


# analog BPF specifications
b_bpf, a_bpf = signal.lp2bs(b_lpf, a_lpf, w_0_analog, B)
w_bpf, h_bpf = signal.freqs(b_bpf, a_bpf, worN=1000)

# Plotting the magnitude response of the analog Band pass filter
plt.plot(w_bpf, abs(h_bpf))
plt.axis([0, w_stop_analog[1] + 1, 0, 1.1])
plt.xlabel("w")
plt.ylabel("H(w)")
plt.yticks(np.arange(0, 1.25, 0.25))
plt.title('Magnitude response of the initial bandpass filter in lineaer scale')
plt.grid(linestyle='dashed')
plt.show()

# Applying Bilinear transform

l,m = signal.bilinear(b_bpf, a_bpf)

w, h = signal.freqz(l,m, worN=10000)
w *= w_s / 2 / math.pi

plt.plot(w, abs(h))
plt.axis([0, w_s / 2, 0, 1.1])
plt.xlabel("w")
plt.ylabel("H(w)")
plt.xticks(np.arange(0, 410, 30))
plt.yticks(np.arange(0, 1.25, 0.25))
plt.title('Bilenear Magnitude response in linear scale of BPF')
plt.grid(linestyle='dashed')
plt.show()

# magnitude_response = 20 * np.log10(abs(h))
plt.plot(w, 20 * np.log10(abs(h)))
plt.axis([0, w_s / 2, -100, 0])
plt.xlabel("w")
plt.ylabel("H(w)")
plt.xticks(np.arange(0,400, 20))
plt.yticks(np.arange(-100, 10, 10))
plt.title('Bilenear Magnitude response in dB of BPF')
plt.grid(linestyle='dashed')
plt.show()
