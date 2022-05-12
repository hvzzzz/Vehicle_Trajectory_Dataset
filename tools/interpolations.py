import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import ndimage

def f(x,x_points,y_points):  
  tck = interpolate.splrep(x_points, y_points)
  return interpolate.splev(x, tck)

y=[ 191.78 ,   191.59,    191.59,    191.41,    191.47,    191.33,    191.25  \
  ,191.33 ,   191.48 ,   191.48,    191.51,    191.43,    191.42,    191.54    \
  ,191.5975,  191.555,   191.52 ,   191.25 ,   191.15  ,  191.01  ]
x = np.linspace(1 ,20,len(y))

# convert both to arrays
x_sm = np.array(x)
y_sm = np.array(y)

# resample to lots more points - needed for the smoothed curves
x_smooth = np.linspace(x_sm.min(), x_sm.max(), 200)

# spline - always goes through all the data points x/y
#y_spline = interpolate.spline(x, y, x_smooth)
y_spline = f(x_smooth,x_sm,y_sm) 
spl = interpolate.UnivariateSpline(x, y,k=4)

sigma = 2
x_g1d = ndimage.gaussian_filter1d(x_sm, sigma)
y_g1d = ndimage.gaussian_filter1d(y_sm, sigma)

fig, ax = plt.subplots(figsize=(10, 10))
ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), frameon=False)

plt.plot(x_sm, y_sm, 'green', linewidth=1,label='Original Data')
plt.plot(x_smooth, y_spline, 'red', linewidth=1,label='Cubic Spline')
#plt.plot(x_sm, spl(x_sm), linewidth=1,label='Univariante Spline')
#plt.plot(x_sm,y_g1d, 'magenta', linewidth=1,label='Gaussian Kernel Smoothing')
plt.legend()

#print(len(y_sm),len(y_spline),len(y_g1d))
#print(y_spline)
plt.show()
