import numpy as np
import scipy as sc
from scipy import signal
import math
import cmath
import matplotlib.pyplot as plt
#plt.rcParams['text.usetex'] = True

w_s= 240
w_width = 10
w_cutoff = np.array([40, 85])
ripple = 30
# width = 2*w_width/w_s
width = 20/240
# F_cutoff_norm = np.multiply(w_cutoff,2/w_s)
F_cutoff_norm = np.multiply(np.array([40, 85]),2/240)

M, B = signal.kaiserord(ripple, width)

#incerement M by 1 to make it odd as told in the question
M += 1
print(M)

window_coeff = signal.windows.kaiser(M, B)
print(window_coeff)
# plotting the window coefficients
plt.stem(window_coeff)
plt.title('Kaiser window coefficients')
plt.xlabel('n')
plt.grid(linestyle='dashed')
plt.show()

filter_coeff = signal.firwin(M, F_cutoff_norm, window = ('kaiser', B))
# plotting the filter coefficients
plt.stem(filter_coeff)
plt.title('Filter coefficients')
plt.ylabel('h[n]')
plt.xlabel('n')
plt.grid(linestyle='dashed')
plt.show()

w, h = signal.freqz(filter_coeff, worN=10000)

w *= w_s/2/math.pi

# plotting the magnitude response in linear scale
plt.plot(w,abs(h))
plt.xlabel('w')
plt.ylabel('|H(w)|')
plt.title("linear scale magnitude response")
plt.grid(linestyle='dashed')
plt.show()

mag_response = 20*np.log10(abs(h))
# plotting the magnitude response in dB scale
plt.plot(w,mag_response)
plt.xlabel('w')
plt.ylabel('20log|H(w)|')
plt.title('magnitude response in decibels')
plt.grid(linestyle='dashed')
plt.show()