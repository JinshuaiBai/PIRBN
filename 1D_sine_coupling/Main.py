import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIRBN import PIRBN
from OPT import Adam
import scipy.io
from rbn_net import RBN_Net
from Cal_jac import cal_adapt
import matplotlib.pyplot as plt

### Define the number of sample points
ns = 51

### Define the sample points' interval
dx = 2./(ns-1)

### Initialise sample points' coordinates
xy = np.zeros((ns, 1)).astype(np.float32)
for i in range(0, ns):
    xy[i, 0] = i * dx + 20
xy_b = np.array([[20.], [22.]])

x = [xy, xy_b]
y = [-2*np.pi*(22-xy)*np.cos(2*np.pi*xy)+0.5*np.sin(2*np.pi*xy)-np.pi**2*(22-xy)**2*np.sin(2*np.pi*xy) + \
     2*8*np.pi*(xy-20)*np.cos(2*8*np.pi*xy)+0.5*np.sin(2*8*np.pi*xy)-8**2*np.pi**2*(xy-20)**2*np.sin(2*8*np.pi*xy)]

### Set up raidal basis network
n_in = 1
n_out = 1
n_neu = 61
b = 10.
c = [20-0.2, 22+.2]

rbn = RBN_Net(n_in, n_out, n_neu, b, c).build()

### Set up PIRBN
pirbn = PIRBN(rbn)

### Visualise NTK after initialisation
a_g, a_b, jac = cal_adapt(pirbn, x)
a = np.dot(jac, np.transpose(jac))
plt.imshow(a/(np.max(abs(a))), cmap = 'bwr', vmax = 1, vmin = -1)
plt.colorbar()
plt.show()

#%%
opt = Adam(pirbn, x, y, learning_rate = 0.001, maxiter=20001)
result=opt.fit()

### Visualise results
ns = 1001
dx = 2/(ns-1)
xy = np.zeros((ns, 1)).astype(np.float32)
for i in range(0, ns):
    xy[i, 0] = i * dx + 20
y=rbn(xy)
plt.plot(xy,y)
plt.plot(xy,np.sin(2*np.pi*(xy))*(22-xy)**2/4+np.sin(2*8*np.pi*(xy))*(xy-20)**2/4)
plt.legend(['predict','ground truth'])
plt.show()

### Visualise NTK after training
_, _, jac = cal_adapt(pirbn, x)
a = np.dot(jac, np.transpose(jac))
plt.imshow(a/(np.max(abs(a))), cmap='bwr', vmax=1, vmin=-1)
plt.colorbar()
plt.show()

### Save data
scipy.io.savemat('out.mat', {'NTK': a, 'x': xy, 'y': y})
