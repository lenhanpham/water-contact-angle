#!/usr/bin/env python
# coding: utf-8

# In[432]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import measure

densmap = np.loadtxt('E:\Projects\MoS2\MD\Droplet\output\\densmap.dat')

to_kg_m3=((18.01528)*(10**-3))/((6.022*10**23)*(10**-27)) 
densmass =  densmap*to_kg_m3 

x = densmap[1:-1,0]
z = densmap[0,1:-1]
dens = densmass[1:-1, 1:-1]

trans_dens =  np.transpose(dens)

plt.figure(1, figsize = (20,10))
axes = plt.gca()
axes.set_xlim([10,19])
axes.set_ylim([1.5,4])
axes.set_aspect(1)

masscontour = plt.contour(x, z , trans_dens, levels = [300])

p = masscontour.collections[0].get_paths()[0]

v = p.vertices
xc = v[:,0]
zc = v[:,1]



maxzc = np.amax(v[:,1])

maxzc_index = np.where(v == maxzc)

highest_point = v[maxzc_index[0],:] 

center_line = trans_dens[:,maxzc_index[0]]


# In[433]:


### Plot mass density along the totally symmetric axis of the droplet ######

plt.figure(1, figsize = (24,12))
axes = plt.gca()
axes.set_xlim([1,5])


center_libe_plot = plt.scatter(z, center_line, marker = "o", s=200, facecolors='none', edgecolors='b')


# In[434]:


####Fitting#####
from scipy import optimize

def density_func(x, a, b, c):
    return a*(1 - np.tanh((x - b)/c)) 

density_line = np.transpose(center_line).flatten() 

zfit_index1 = np.where(z == 2.3)
zfit_index2 = np.where(z == 6.0)

params, params_covariance = optimize.curve_fit(f = density_func, xdata = z[zfit_index1[0][0]:zfit_index2[0][0]], ydata = density_line[zfit_index1[0][0]:300], p0=[400, 8, 5])

yfitteddata = density_func(z, params[0], params[1], params[2])

plt.figure(1, figsize = (24,12))
fitted_curve = plt.plot(z[0:zfit_index2[0][0]], yfitteddata[0:zfit_index2[0][0]], linewidth = 4)
center_line_plot = plt.scatter(z[0:zfit_index2[0][0]], center_line[0:zfit_index2[0][0]], marker = "o", s=150, facecolors='none', edgecolors='b')

print(params)
print(np.sqrt(np.diag(params_covariance)))

#zfit_index1[0]


# In[386]:


np.where(np.isclose(z, 2.3))
#np.where(np.isclose(z, 6.0))


# In[436]:


plt.figure(1, figsize = (20,10))
axes = plt.gca()
axes.set_xlim([10,19])
axes.set_ylim([1.5,4])
axes.set_aspect(1)
masscontour = plt.contour(x, z , trans_dens, levels = [537])

cir = masscontour.collections[0].get_paths()[0]

circle = cir.vertices
circle[:,:]


# In[437]:


#### select contour which is at least 2.3 nm above the surface to remove noise from layers which are too close to the surface

zfit_index = np.where(circle == 2.30)
start_index = zfit_index[0][0]
stop_index = zfit_index[0][-1]

print(zfit_index)


# In[438]:


####Fitting Circle#####
from scipy import optimize 

def circle_func(x, a, b, c):
    return b + np.sqrt(c**2 - (x - a)**2) 



pars, pars_covariance = optimize.curve_fit(f = circle_func, xdata = circle[start_index:stop_index,0], ydata = circle[start_index:stop_index,1], p0=[14, 1, 5])

ycirfitteddata = circle_func(circle[start_index:stop_index,0], params[0], params[1], params[2])

#plt.figure(2, figsize = (15,5))
#axes = plt.gca()
#axes.set_xlim([10,18])
#axes.set_ylim([1.5,5])

#fitted_cir = plt.plot(circle[start_index:stop_index,0], ycirfitteddata)

print(pars)
print(np.sqrt(np.diag(pars_covariance)))

#print(ycirfitteddata)


# In[417]:


#### Plot the fitted circle #####

circle = plt.figure(1, figsize = (20,20))
draw_circle = plt.Circle((14.33974065, -0.79343233), 4.08972749, color='b', fill=False)
axes = plt.gca()
axes.set_xlim([10,19])
axes.set_ylim([1.5,4])
axes.set_aspect(1)
axes.add_artist(draw_circle)
axes.set_title('Fitted Circle', fontsize = 25)


# In[450]:


#### Plot the fitted circle and the droplet contour to check the fitted result

plt.figure(1, figsize = (20,20))
draw_circle = plt.Circle((14.33974065, -0.79343233), 4.08972749, color='b', fill=False)
ax = plt.gca()
ax.set_xlim([10,19])
ax.set_ylim([1.5,4])
ax.set_aspect(1)
ax.add_artist(draw_circle)
crl = plt.plot(circle[start_index-150:stop_index+150,0], circle[start_index-150:stop_index+150,1])
max_height = np.amax(circle[start_index:stop_index,1])

max_height_index = np.where(circle == max_height)

center_dense = trans_dens[:, max_height_index[0][0]]

#### uncomment this line to get index of z density along the totally symmetric axis of the droplet 
#np.where(center_dense == np.amin(center_dense, 0))

### uncomment this lien to get the 1-dimension density matrix along the totally symmetric axis of the droplet
#center_dense


# In[453]:


### Calculate the contact angle 


import sympy as sp

x = sp.Symbol('x')

a = pars[0]    
b = pars[1]   
c = pars[2]   


#### 1.7 nm is the thickess of the MoS2 surface 
xvalues = sp.solve(sp.sqrt(c**2 - (x - a)**2) + b -1.7)

base = (sp.Abs(xvalues[1] - xvalues[0]))/2

print(xvalues)


height = b + c - 1.7 

print(height)

theta = sp.asin((2*height*base)/(base**2 + height**2))

theta_degree = sp.N(sp.deg(theta), 4)


print("The base distance: ")
print(base, "\n")

print("Contact angle: ")
print(theta_degree)





