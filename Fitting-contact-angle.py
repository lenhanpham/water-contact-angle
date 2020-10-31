#!/usr/bin/env python
# coding: utf-8


import sympy as sp
from scipy import optimize
import matplotlib.pyplot as plt
from matplotlib import cm, rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# import pandas as pd

### Uncommet to use latex font #####
# plt.rc('text', usetex=True)

#### Uncomment to use other font family #####
# font = {'family' : 'serif',
#        'weight' : 'bold',
#        'size'   : 18}
# rc('font', **font)

droplet_size = "6000-0.505"
cell = [26.26790, 22.74867, 22.00000]
shift = 3

densmap = np.loadtxt(
    f"E:\Projects\MoS2\MD\Droplet\{droplet_size}\densmap-{droplet_size}.dat"
)
# densmap = np.loadtxt('E:\Projects\MoS2\MD\Droplet\output\\densmap.dat')


to_kg_m3 = ((18.01528) * (10 ** -3)) / ((6.022 * 10 ** 23) * (10 ** -27))
densmass = densmap * to_kg_m3


###height of simulation space ###
z = densmap[0, 1:-1]

#### Extract the mass density values by removing the width and the height in the number density matrix ####
dens = densmass[1:-1, 1:-1]

##### Rotate the droplet mass density to plot it ######
trans_dens = np.transpose(dens)

### width of simulation space###
x = densmap[1:-1, 0] + shift


### Determine the center point of droplet with densest mass density ####
maxdens_idx = np.argmax(trans_dens, axis=None)

maxdens_idx = np.unravel_index(maxdens_idx, trans_dens.shape)

print(maxdens_idx)


center_line = trans_dens[:, maxdens_idx[1]]


### Shift the part of droplets to make a whole droplet in the box #####
xshift = np.empty(x.size)
for i in range(x.size):
    if x[i] > cell[0]:
        xshift[i] = x[i] - cell[0] - x[maxdens_idx[1]]
    else:
        xshift[i] = x[i] - x[maxdens_idx[1]]


##### Plot droplet #####

figuresize = [20, 10]
x_minmax = [-8, 8]
y_minmax = [1.5, 6.0]


def plot_droplet():
    plt.figure(1, figsize=(figuresize[0], figuresize[1]))
    axes = plt.gca()
    axes.set_xlim([x_minmax[0], x_minmax[1]])
    axes.set_ylim([y_minmax[0], y_minmax[1]])
    axes.set_aspect(1)
    masscontour = plt.contour(xshift, z, trans_dens, 50, cmap="jet", origin="lower")
    # masscontour = plt.contourf(xshift, z , trans_dens, )
    droplet = plt.imshow(
        trans_dens,
        extent=[x_minmax[0] * 20, x_minmax[1] * 20, y_minmax[0], y_minmax[1]],
        origin="lower",
        aspect="equal",
        cmap="jet",
        interpolation="bilinear",
    )
    # create an axis on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(masscontour, cax=cax)


plot_droplet()

### Plot mass density along the totally symmetric axis of the droplet ######

plt.figure(1, figsize=(figuresize[0], figuresize[1]))
axes = plt.gca()
axes.set_xlim([0, 7])


center_line_plot = plt.scatter(
    z, center_line, marker="o", s=200, facecolors="none", edgecolors="b"
)


####Fitting#####


def density_func(x, a, b, c):
    return a * (1 - np.tanh((x - b) / c))


density_line = np.transpose(center_line).flatten()

#### lower and upper bounds of the mass density of the droplet along the totally symmetric axis ####
zfit_index1 = np.where(z == 2.2)
zfit_index2 = np.where(z == 8)

#### Fitting process ######
params, params_covariance = optimize.curve_fit(
    f=density_func,
    xdata=z[zfit_index1[0][0] : zfit_index2[0][0]],
    ydata=density_line[zfit_index1[0][0] : zfit_index2[0][0]],
    p0=[613, 4, 1],
)

yfitteddata = density_func(z, params[0], params[1], params[2])

plt.figure(1, figsize=(24, 12))
fitted_curve = plt.plot(
    z[0 : zfit_index2[0][0]], yfitteddata[0 : zfit_index2[0][0]], linewidth=4
)
center_line_plot = plt.scatter(
    z[0 : zfit_index2[0][0]],
    center_line[0 : zfit_index2[0][0]],
    marker="o",
    s=150,
    facecolors="none",
    edgecolors="b",
)

print(params)
print(np.sqrt(np.diag(params_covariance)))

# zfit_index1[0]


plt.figure(2, figsize=(figuresize[0], figuresize[1]))
axes = plt.gca()
axes.set_xlim([x_minmax[0], x_minmax[1]])
axes.set_ylim([y_minmax[0], y_minmax[1]])
axes.set_aspect(1)
masscontour = plt.contour(xshift, z, trans_dens, levels=[params[0]])

cir = masscontour.collections[0].get_paths()[0]

circle = cir.vertices
# circle


# select contour which is at least 2.3 nm above the surface to remove noise from layers which are too close to the surface

zfit_index = np.where(circle == 2.4)
start_index = zfit_index[0][0]
stop_index = zfit_index[0][-1]

print(start_index, stop_index)

####Fitting Circle#####


def circle_func(x, a, b, c):
    return b + np.sqrt(c ** 2 - (x - a) ** 2)


pars, pars_covariance = optimize.curve_fit(
    f=circle_func,
    xdata=circle[start_index:stop_index, 0],
    ydata=circle[start_index:stop_index, 1],
    p0=[0, -2, 5],
    maxfev=5000,
)

ycirfitteddata = circle_func(
    circle[start_index:stop_index, 0], params[0], params[1], params[2]
)

# plt.figure(2, figsize = (15,5))
# axes = plt.gca()
# axes.set_xlim([10,18])
# axes.set_ylim([1.5,5])

# fitted_cir = plt.plot(circle[start_index:stop_index,0], ycirfitteddata)

print(pars)
print(np.sqrt(np.diag(pars_covariance)))

# print(ycirfitteddata)

#### Plot the fitted circle #####

fittedcircle = plt.figure(3, figsize=(figuresize[0], figuresize[1]))
axes = plt.gca()
axes.set_xlim([x_minmax[0], x_minmax[1]])
axes.set_ylim([y_minmax[0], y_minmax[1]])
axes.set_aspect(1)
draw_circle = plt.Circle((pars[0], pars[1]), pars[2], color="b", fill=False)
axes.add_artist(draw_circle)
axes.set_title("Fitted Circle", fontsize=25)


# Plot the fitted circle and the droplet contour to check the fitted result

plt.figure(5, figsize=(figuresize[0], figuresize[1]))
axes = plt.gca()
axes.set_xlim([x_minmax[0], x_minmax[1]])
axes.set_ylim([y_minmax[0], y_minmax[1]])
axes.set_aspect(1)
draw_circle = plt.Circle((pars[0], pars[1]), pars[2], color="b", fill=False)
axes.add_artist(draw_circle)
crl = plt.plot(
    circle[start_index - 150 : stop_index + 150, 0],
    circle[start_index - 150 : stop_index + 150, 1],
    color="red",
)


# Calculate the contact angle

x = sp.Symbol("x")

a = pars[0]
b = pars[1]
c = pars[2]

# height of the substrate
h = 1.7

# 1.7 nm is the thickess of the MoS2 surface
xvalues = sp.solve(sp.sqrt(c ** 2 - (x - a) ** 2) + b - h)


#### Radius of the fitted droplet #####
base = (sp.Abs(xvalues[1] - xvalues[0])) / 2

# print(xvalues)


height = b + c - h


theta = sp.asin((2 * height * base) / (base ** 2 + height ** 2))

theta_degree = sp.N(sp.deg(theta), 4)


print("The height of the droplet: ")

print(height, "\n")

print("The radius of the fitted droplet: ")
print(base, "\n")

print("Contact angle: ")
print(theta_degree)

with open(
    f"E:\Projects\MoS2\MD\Droplet\{droplet_size}\Contac-angle-{droplet_size}.txt", "w"
) as text_file:
    print(f"The height of the droplet: {height}", "\n", file=text_file)
    print(f"The base distance: {base}", "\n", file=text_file)
    print(f"Contac angle: {theta_degree}", "\n", file=text_file)
