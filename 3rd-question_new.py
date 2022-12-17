import numpy as np
import scipy as sc
from scipy import signal
import math
import cmath
import matplotlib.pyplot as plt

# sampling frequency
w_s= 400


#Calculating ripples for the weight
# calculating the ripple values in decibeles and converting them to linear values
s_ripple_db = 30
p_ripple_db = 20
# these are the linear values of the ripples
s_ripple = 10**(-s_ripple_db/20)
p_ripple = 10**(-p_ripple_db/20)
# this is the maximum ripple
ripple = max(s_ripple,p_ripple)
# these are the linear values of the ripples divided by the maximum ripple
s_ripple = ripple/s_ripple
p_ripple = ripple/p_ripple

weight = np.array([p_ripple, s_ripple, p_ripple, s_ripple])

#Calculating M
# the del_f was calculated in the notes
del_f= 1/20

M = math.ceil((0.5*(s_ripple_db+p_ripple_db)-13)/14.6/del_f)+1
print(M)
M=21


#Calculating filter specifications for remez
bands = np.multiply([0,20,60, 100 , 140, 160, 180, 200], 1/w_s)
desired = np.array([1,0, 1, 0])

#Obtaining the filter
filter_coeff = signal.remez(M, bands, desired, weight=weight)
w, h = signal.freqz(filter_coeff, worN=10000)

w *=w_s/2/math.pi

magnitude_response = 20*np.log10(abs(h))


plt.plot(w,abs(h))
plt.axis([0, 100, 0, 1.25])
plt.xlabel("w")
plt.ylabel("H(w)")
plt.xticks(np.arange(-10,190,30))
plt.yticks(np.arange(0,1.25,0.25))
plt.title("linear scale magnitude response")
plt.grid(linestyle='dashed')
plt.show()

plt.plot(w,magnitude_response)
plt.axis([0, 100, -100, 0])
plt.xlabel("w")
plt.ylabel("H(w)")
plt.xticks(np.arange(0,190,10))
plt.yticks(np.arange(-100,10,10))
plt.title('magnitude response in decibels')
plt.grid(linestyle='dashed')
plt.show()
