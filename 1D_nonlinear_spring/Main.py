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
ns = 1001

### Define the sample points' interval
dx = 100./(ns-1)

### Initialise sample points' coordinates
xy = np.zeros((ns, 1)).astype(np.float32)
for i in range(0, ns):
    xy[i, 0] = i * dx
xy_b = np.array([[0.]])

x = [xy, xy_b]
y = [2*np.cos(xy)+3*xy*np.sin(xy)+np.sin(xy*np.sin(xy))]

### Set up raidal basis network
n_in = 1
n_out = 1
n_neu = 1021
b=1.
c = [-1., 101.]

rbn = RBN_Net(n_in, n_out, n_neu, b, c).build()

### Set up PIRBN
pirbn = PIRBN(rbn)

### Train the PIRBN
opt = Adam(pirbn, x, y, learning_rate = 0.001, maxiter=40001)
result=opt.fit()

### Visualise results
ns = 1001
dx = 100/(ns-1)
xy = np.zeros((ns, 1)).astype(np.float32)
for i in range(0, ns):
    xy[i, 0] = i * dx
y=rbn(xy)
plt.plot(xy,y)
plt.plot(xy,xy*np.sin(xy))
plt.legend(['predict','ground truth'])
plt.show()

### Save data
scipy.io.savemat('out.mat', {'x': xy, 'y': y})
