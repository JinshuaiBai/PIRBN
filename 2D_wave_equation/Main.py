import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIRBN import PIRBN
from OPT import Adam
import scipy.io
from rbf_net import RBF_Net

### Define the number of sample points
n = 51
ns = n

### Define the sample points' interval
dx = 1./(n-1)

### Initialize sample points' coordinates
xy = np.zeros((ns*ns, 2)).astype(dtype='float32')
k = 0
dx = 1/(ns-1)
for i in range(ns):
    for j in range(ns):
        xy[k, 0] = i * dx
        xy[k, 1] = j * dx
        k = k+1
xy_t = np.hstack([np.linspace(0, 1, n).reshape(n, 1).astype(np.float32), \
                      np.ones((n, 1)).astype(np.float32)])
xy_b = np.hstack([np.linspace(0, 1, n).reshape(n, 1).astype(np.float32), \
                  np.zeros((n, 1)).astype(np.float32)])
xy_l = np.hstack([np.zeros((n, 1)).astype(np.float32), \
              np.linspace(0, 1, n).reshape(n, 1).astype(np.float32)])
xy_r = np.hstack([np.ones((n, 1)).astype(np.float32), \
              np.linspace(0, 1, n).reshape(n, 1).astype(np.float32)])
xy_bound = np.vstack([xy_t, xy_b])
x = [xy, xy_bound, xy_l]
y = [np.sin(np.pi*(xy_l[..., 1, np.newaxis]))+0.5*np.sin(4*np.pi*(xy_l[..., 1, np.newaxis]))]

n_in = 2
n_out = 1
n_neu_x = 61
n_neu_y = 61
c_x = [-0.1, 1.1]
c_y = [-0.1, 1.1]

### Build RBN
rbn = RBF_Net(n_in, n_out, n_neu_x, n_neu_y, 20, c_x, c_y).build()

### Build PIRBN
pirbn = PIRBN(rbn)

### Train PIRBN
opt = Adam(pirbn, x, y, learning_rate = 0.001, maxiter=80001)
result=opt.fit()

### Visualise results
xy = np.zeros((101*101, 2)).astype(dtype='float32')
k = 0
dx = 1/(101-1)
for i in range(101):
    for j in range(101):
        xy[k, 0] = i * dx
        xy[k, 1] = j * dx
        k = k+1
y=rbn(xy)

fig1 = plt.figure(1)
ground_truth = np.cos(2*np.pi*xy[:,0])*np.sin(np.pi*xy[:,1])+0.5*np.cos(8*np.pi*xy[:,0])*np.sin(4*np.pi*xy[:,1])
plt.scatter(xy[:,0],xy[:,1],s=5,c=ground_truth,cmap='jet',vmin=-1.5,vmax=1.5)
plt.axis('equal')
plt.colorbar()
plt.title('Ground Truth')
plt.show()

fig2 = plt.figure(2)
plt.scatter(xy[:,0],xy[:,1],s=5,c=y,cmap='jet',vmin=-1.5,vmax=1.5)
plt.axis('equal')
plt.colorbar()
plt.title('PIRBN')
plt.show()

### Output results
scipy.io.savemat('out.mat', {'xy': xy, 'u': y.numpy()})
