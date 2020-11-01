#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 22:19:22 2020

@author: martianface
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns


def point_to_sine(x,z,n):
    """
    According to the point coordinates to generate its dual sine curve
    """
    theta_rho_list = []
    for i in range(n):
        theta_tmp = i*math.pi/n
        rho_tmp = x * np.math.cos(theta_tmp) + z * np.math.sin(theta_tmp)
        theta_rho_list.append([theta_tmp, rho_tmp])
        
    return theta_rho_list

def dist_point_to_line(p,l):
    """
    Calculating the distance bewteen a point and a line represented by (k,b)

    Parameters
    ----------
    p : point
        (x,y) coordinates.
    l : line
        (k,b).

    Returns 
    -------
    d : float value
        distance.

    """
    x = p[0]
    y = p[1]
    k = l[0]
    b = l[1]
    return abs(k*x-y+b)/(k**2+1)**0.5

# ====================== reading and writing data =============================
f_laser_data = open("./31.1.txt", "r") # reading laser data saved as txt file
f_laser_data_filtered = open("./31.1_f.txt", "w") # reading laser data saved as txt file
# laser data as x and z coordinates
xx = []
zz = []
# reading the welding teach program line by line
for line in f_laser_data.readlines():
    raw_data = line.split('  ')
    if len(raw_data) == 4 and raw_data[1] != '-64.3914':
        xx.append(float(raw_data[0]))
        zz.append(float(raw_data[1]))
        f_laser_data_filtered.write(raw_data[0]+' '+raw_data[1]+'\n')
#        print raw_data[0], raw_data[1]
f_laser_data.close()
f_laser_data_filtered.close()
# =============================================================================


# =============================first iteration ================================
# transferring from point to sine
n = 180
theta_rho_list = []
for i in range(len(xx)):
    rho_theta_i = point_to_sine(xx[i],zz[i],n)    
    theta_rho_list.append(rho_theta_i)
# transform to point set
point_set = np.array(theta_rho_list).reshape(len(theta_rho_list)*n,2)

# find the max and min    
rho_max_list = []
rho_min_list = []
for i in range(len(xx)):
    a = np.array(theta_rho_list[i])
    rho_max_list.append(max(a[:,1]))
    rho_min_list.append(min(a[:,1]))
rho_max = max(rho_max_list)
rho_min = min(rho_min_list)

# generate grid
# theta_i = theta_min + i*(theta_max - theta_min)/n
# i = n*(theta_i - theta_min)/(theta_max - theta_min)
# rho_j = rho_min + j*(rho_max - rho_min)/n
# j = n*(rho_i - rho_min)/(rho_max - rho_min)
m = 200 # number of intervals
grid_theta = np.linspace(-0.01, 3.17, m+1)
grid_rho = np.linspace(math.floor(rho_min),math.ceil(rho_max), m+1)
hist_mat = np.zeros((m,m))

theta_interval_min = min(grid_theta)
theta_interval_max = max(grid_theta)
rho_interval_min = min(grid_rho)
rho_interval_max = max(grid_rho)
for point in point_set:
    tmp_i = m*(point[0] - theta_interval_min)/(theta_interval_max - theta_interval_min)
    tmp_j = m*(point[1] - rho_interval_min)/(rho_interval_max - rho_interval_min)
    hist_mat[int(tmp_j//1)][int(tmp_i//1)] += 1

plt.figure()
X,Y = np.meshgrid(grid_theta, grid_rho)
plt.plot(X, Y, color='red', marker='', linestyle='-', linewidth=1)
plt.plot(X.T, Y.T, color='red', marker='', linestyle='-', linewidth=1)
plt.plot(point_set[:,0], point_set[:,1], '.b', alpha=1.0, markersize=0.5)

# plotting heatmap
x = point_set[:,0]
y = point_set[:,1]

gridx = np.linspace(-0.01, 3.17, m+1)
gridy = np.linspace(-40, 48, m+1)

gridx_length = 3.18/m
gridy_length = 88/m

grid, _, _ = np.histogram2d(y, x, bins=[gridy, gridx])

plt.figure()
cmap = plt.get_cmap('jet')
plt.pcolormesh(gridx, gridy, grid, cmap=cmap)
#plt.plot(x, y, 'r.', markersize=0.3)
plt.colorbar()

# another method for plotting heatma
plt.figure()
sns.heatmap(hist_mat)

M = hist_mat.max()
# threshold = int(M*2.3875/10)
# result = np.array(np.where(hist_mat > threshold)) # threshhold
idx = np.where(hist_mat==M) # threshhold
inter_theta = grid_theta[idx[1][0]]
inter_rho = grid_rho[idx[0][0]]
k_1 = -1/math.tan(inter_theta + gridx_length/2)
b_1 = (inter_rho + gridx_length/2)/math.sin(inter_theta + gridx_length/2)
print('line 1')
print(idx)
print(inter_theta, inter_rho)
print(k_1, b_1)


# calculating the distance bewteen points to the fitted line
# generating new data set by removing the fitted points
dist_thresh_1 = 1.1
xxx = []
zzz = []
d = []
i_list = []
for i in range(len(xx)):
    p = [xx[i], zz[i]]
    l = [k_1,b_1]
    d.append(dist_point_to_line(p,l))
    if dist_point_to_line(p,l)>dist_thresh_1:
        i_list.append(i)
        xxx.append(xx[i])
        zzz.append(zz[i])

# ========================== 2nd iteration ===================================
# generate sine data
n = 180
theta_rho_list_2 = []
for i in range(len(xxx)):
    rho_theta_i_2 = point_to_sine(xxx[i],zzz[i],n)    
    theta_rho_list_2.append(rho_theta_i_2)
# transform to point set
point_set_2 = np.array(theta_rho_list_2).reshape(len(theta_rho_list_2)*n,2)

## find the max and min    
#rho_max_list_2 = []
#rho_min_list_2 = []
#for i in range(len(xxx)):
#    a = np.array(theta_rho_list_2[i])
#    rho_max_list_2.append(max(a[:,1]))
#    rho_min_list_2.append(min(a[:,1]))
#rho_max_2 = max(rho_max_list_2)
#rho_min_2 = min(rho_min_list_2)

# generate grid
m = 200
grid_theta_2 = np.linspace(-0.01, 3.17, m+1)
grid_rho_2 = np.linspace(math.floor(rho_min),math.ceil(rho_max), m+1)
hist_mat_2 = np.zeros((m,m))

theta_interval_min_2 = min(grid_theta_2)
theta_interval_max_2 = max(grid_theta_2)
rho_interval_min_2 = min(grid_rho_2)
rho_interval_max_2 = max(grid_rho_2)
for point in point_set_2:
    tmp_i = m*(point[0] - theta_interval_min_2)/(theta_interval_max_2 - theta_interval_min_2)
    tmp_j = m*(point[1] - rho_interval_min_2)/(rho_interval_max_2 - rho_interval_min_2)
    hist_mat_2[int(tmp_j//1)][int(tmp_i//1)] += 1

plt.figure()
X_2,Y_2 = np.meshgrid(grid_theta_2, grid_rho_2)
plt.plot(X_2, Y_2, color='red', marker='', linestyle='-', linewidth=1)
plt.plot(X_2.T, Y_2.T, color='red', marker='', linestyle='-', linewidth=1)
plt.plot(point_set_2[:,0], point_set_2[:,1], '.b', alpha=1.0, markersize=0.5)


# plotting heatmap
x_2 = point_set_2[:,0]
y_2 = point_set_2[:,1]

gridx_2 = np.linspace(-0.01, 3.17, m+1)
gridy_2 = np.linspace(rho_interval_min_2, rho_interval_max_2, m+1)

gridx_length = 3.18/m
gridy_length = (rho_interval_max_2 - rho_interval_min_2)/m

grid_2, _, _ = np.histogram2d(y_2, x_2, bins=[gridy_2, gridx_2])

plt.figure()
cmap = plt.get_cmap('jet')
plt.pcolormesh(gridx_2, gridy_2, grid_2, cmap=cmap)
#plt.plot(x, y, 'r.', markersize=0.3)
plt.colorbar()

# another method for plotting heatma
plt.figure()
sns.heatmap(hist_mat_2)

M = hist_mat_2.max()
# threshold = int(M*2.3875/10)
#result = np.array(np.where(hist_mat_2 > 50)) # threshhold
idx = np.where(hist_mat_2==M) # threshhold
inter_theta_2 = grid_theta_2[idx[1][0]]
inter_rho_2 = grid_rho_2[idx[0][0]]
k_2 = -1/math.tan(inter_theta_2 + gridx_length/2)
b_2 = (inter_rho_2 + gridx_length/2)/math.sin(inter_theta_2 + gridx_length/2)
print('line 2')
print(idx)
print(inter_theta_2, inter_rho_2)
print(k_2, b_2)

# calculating the distance bewteen points to the fitted line
# generating new data set by removing the fitted points
dist_thresh_2 = 1.1
xxxx = []
zzzz = []
d = []
i_list_2 = []
for i in range(len(xxx)):
    p = [xxx[i], zzz[i]]
    l = [k_2,b_2]
    d.append(dist_point_to_line(p, l))
    if dist_point_to_line(p, l)>dist_thresh_2:
        xxxx.append(xxx[i])
        zzzz.append(zzz[i])
        i_list_2.append(i)

# ========================== 3rd iteration ===================================
# generate sine data
n = 180
theta_rho_list_3 = []
for i in range(len(xxxx)):
    rho_theta_i_3 = point_to_sine(xxxx[i],zzzz[i],n)    
    theta_rho_list_3.append(rho_theta_i_3)
# transform to point set
point_set_3 = np.array(theta_rho_list_3).reshape(len(theta_rho_list_3)*n,2)

# find the max and min    
rho_max_list_3 = []
rho_min_list_3 = []
for i in range(len(xxxx)):
    a = np.array(theta_rho_list_3[i])
    rho_max_list_3.append(max(a[:,1]))
    rho_min_list_3.append(min(a[:,1]))
rho_max_3 = max(rho_max_list_3)
rho_min_3 = min(rho_min_list_3)

# generate grid
m = 200
grid_theta_3 = np.linspace(-0.01, 3.17, m+1)
grid_rho_3 = np.linspace(math.floor(rho_min),math.ceil(rho_max), m+1)
hist_mat_3 = np.zeros((m,m))

theta_interval_min_3 = min(grid_theta_3)
theta_interval_max_3 = max(grid_theta_3)
rho_interval_min_3 = min(grid_rho_3)
rho_interval_max_3 = max(grid_rho_3)
for point in point_set_3:
    tmp_i = m*(point[0] - theta_interval_min_3)/(theta_interval_max_3 - theta_interval_min_3)
    tmp_j = m*(point[1] - rho_interval_min_3)/(rho_interval_max_3 - rho_interval_min_3)
    hist_mat_3[int(tmp_j//1)][int(tmp_i//1)] += 1

plt.figure()
X_3,Y_3 = np.meshgrid(grid_theta_3, grid_rho_3)
plt.plot(X_3, Y_3, color='red', marker='', linestyle='-', linewidth=1)
plt.plot(X_3.T, Y_3.T, color='red', marker='', linestyle='-', linewidth=1)
plt.plot(point_set_3[:,0], point_set_3[:,1], '.b', alpha=1.0, markersize=0.5)


# plotting heatmap
x_3 = point_set_3[:,0]
y_3 = point_set_3[:,1]

gridx_3 = np.linspace(-0.01, 3.17, m+1)
gridy_3 = np.linspace(rho_interval_min_3, rho_interval_max_3, m+1)

gridx_length = 3.18/m
gridy_length = (rho_interval_max_3 - rho_interval_min_3)/m

grid_3, _, _ = np.histogram2d(y_3, x_3, bins=[gridy_3, gridx_3])

plt.figure()
cmap = plt.get_cmap('jet')
plt.pcolormesh(gridx_3, gridy_3, grid_3, cmap=cmap)
#plt.plot(x, y, 'r.', markersize=0.3)
plt.colorbar()

# another method for plotting heatma
plt.figure()
sns.heatmap(hist_mat_3)


M = hist_mat_3.max()
# threshold = int(M*2.3875/10)
#result = np.array(np.where(hist_mat_2 > 50)) # threshhold
idx = np.where(hist_mat_3==M) # threshhold
inter_theta_3 = grid_theta_3[idx[1][0]]
inter_rho_3 = grid_rho_3[idx[0][0]]
k_3 = -1/math.tan(inter_theta_3 + gridx_length/2)
b_3 = (inter_rho_3 + gridx_length/2)/math.sin(inter_theta_3 + gridx_length/2)
print('line 3')
print(idx)
print(inter_theta_3, inter_rho_3)
print(k_3, b_3)

# calculating the distance bewteen points to the fitted line
# generating new data set by removing the fitted points
dist_thresh_3 = 1.1
xxxxx = []
zzzzz = []
d = []
for i in range(len(xxxx)):
    p = [xxxx[i], zzzz[i]]
    l = [k_3,b_3]
    d.append(dist_point_to_line(p, l))
    if dist_point_to_line(p, l)>dist_thresh_3:
        xxxxx.append(xxxx[i])
        zzzzz.append(zzzz[i])


# ========================== 4th iteration ===================================
# generate sine data
n = 180
theta_rho_list_4 = []
for i in range(len(xxxxx)):
    rho_theta_i_4 = point_to_sine(xxxxx[i],zzzzz[i],n)    
    theta_rho_list_4.append(rho_theta_i_4)
# transform to point set
point_set_4 = np.array(theta_rho_list_4).reshape(len(theta_rho_list_4)*n,2)

# find the max and min    
rho_max_list_4 = []
rho_min_list_4 = []
for i in range(len(xxxxx)):
    a = np.array(theta_rho_list_4[i])
    rho_max_list_4.append(max(a[:,1]))
    rho_min_list_4.append(min(a[:,1]))
rho_max_4 = max(rho_max_list_4)
rho_min_4 = min(rho_min_list_4)

# generate grid
m = 300
grid_theta_4 = np.linspace(-0.01, 3.17, m+1)
grid_rho_4 = np.linspace(math.floor(rho_min),math.ceil(rho_max), m+1)
hist_mat_4 = np.zeros((m,m))

theta_interval_min_4 = min(grid_theta_4)
theta_interval_max_4 = max(grid_theta_4)
rho_interval_min_4 = min(grid_rho_4)
rho_interval_max_4 = max(grid_rho_4)
for point in point_set_4:
    tmp_i = m*(point[0] - theta_interval_min_4)/(theta_interval_max_4 - theta_interval_min_4)
    tmp_j = m*(point[1] - rho_interval_min_3)/(rho_interval_max_4 - rho_interval_min_4)
    hist_mat_4[int(tmp_j//1)][int(tmp_i//1)] += 1

plt.figure()
X_4,Y_4 = np.meshgrid(grid_theta_4, grid_rho_4)
plt.plot(X_4, Y_4, color='red', marker='', linestyle='-', linewidth=1)
plt.plot(X_4.T, Y_4.T, color='red', marker='', linestyle='-', linewidth=1)
plt.plot(point_set_4[:,0], point_set_4[:,1], '.b', alpha=1.0, markersize=0.5)


# plotting heatmap
x_4 = point_set_4[:,0]
y_4 = point_set_4[:,1]

gridx_4 = np.linspace(-0.01, 3.17, m+1)
gridy_4 = np.linspace(rho_interval_min_4, rho_interval_max_4, m+1)

gridx_length = 3.18/m
gridy_length = (rho_interval_max_4 - rho_interval_min_4)/m

grid_4, _, _ = np.histogram2d(y_4, x_4, bins=[gridy_4, gridx_4])

plt.figure()
cmap = plt.get_cmap('jet')
plt.pcolormesh(gridx_4, gridy_4, grid_4, cmap=cmap)
#plt.plot(x, y, 'r.', markersize=0.3)
plt.colorbar()

# another method for plotting heatma
plt.figure()
sns.heatmap(hist_mat_4)


M = hist_mat_4.max()
# threshold = int(M*2.3875/10)
#result = np.array(np.where(hist_mat_2 > 50)) # threshhold
idx = np.where(hist_mat_4==M) # threshhold
inter_theta_4 = grid_theta_4[idx[1][0]]
inter_rho_4 = grid_rho_4[idx[0][0]]
k_4 = -1/math.tan(inter_theta_4 + gridx_length/2)
b_4 = (inter_rho_4 + gridx_length/2)/math.sin(inter_theta_4 + gridx_length/2)
print('line 4')
print(idx)
print(inter_theta_4, inter_rho_4)
print(k_4, b_4)


# =============================== k, b =======================================
# k_1 = -0.324938
# b_1 = 11.56447

# k_2 = -0.36003
# b_2 = 15.8858

# k_3 = 3.07849
# b_3 = -22.93735

# k_4 = 3.44216
# b_4 = -76.61458

       
plt.show()