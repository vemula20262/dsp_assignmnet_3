import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from math import comb
import scipy

h = [-1,0,1]

x = np.zeros(50)
for i in range(50):
  x[i] = comb(2*i,i)*(0.25**i)

a,b,c,d = np.zeros(14),np.zeros(14),np.zeros(14),np.zeros(14)


for i in range(50):
  if(i<14):
    a[i] = x[i]
  elif(i<28):
    b[i-14] = x[i]
  elif(i<42):
    c[i-28] = x[i]
  else:
    d[i-42] = x[i]


y1 = scipy.signal.fftconvolve(a,h)

y2 = scipy.signal.fftconvolve(b,h)
y3 = scipy.signal.fftconvolve(c,h)
y4 = scipy.signal.fftconvolve(d,h)

y = np.zeros(52)

print(y1.shape)



for i in range(52):
    if(i<16):
        y[i] = y[i] + y1[i]
    if(14<=i<30):
        y[i] = y[i] + y2[i-14]
    if(28<=i<44):
        y[i] = y[i] + y3[i-28]
    if(42<=i<52):
        y[i] = y[i] + y4[i-42]

plt.stem(y)
plt.show()
