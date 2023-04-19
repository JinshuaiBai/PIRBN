import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIRBN import PIRBN
from OPT import Adam
import scipy.io
from rbf_net import RBF_Net

### function to calculate the ground truth result
def func(x):
    y1 = (2*np.cos(np.pi*x[...,0,np.newaxis]+np.pi/5.)+3/2*np.cos(2*np.pi*x[...,0,np.newaxis]-3*np.pi/5.))
    y2 = (2*np.cos(np.pi*x[...,1,np.newaxis]+np.pi/5.)+3/2*np.cos(2*np.pi*x[...,1,np.newaxis]-3*np.pi/5.))
    y = y1 * y2
    
    return y

### Define the number of sample points
n = 51
ns = n

### Define the sample points' interval
dx = 5./(n-1)

### Initialize sample points' coordinates
xt = np.zeros((ns*ns, 2)).astype(dtype='float32')
k = 0
dx = 5/(ns-1)
for i in range(ns):
    for j in range(ns):
        xt[k, 0] = i * dx + 5
        xt[k, 1] = j * dx + 5
        k = k+1
xt_t = np.hstack([np.linspace(5,10, n).reshape(n, 1).astype(np.float32), \
                  10*np.ones((n,1)).astype(np.float32)])
xt_b = np.hstack([np.linspace(5,10, n).reshape(n, 1).astype(np.float32), \
                  5*np.ones((n,1)).astype(np.float32)])
xt_l = np.hstack([5*np.ones((n,1)).astype(np.float32), \
              np.linspace(5,10, n).reshape(n, 1).astype(np.float32)])
xt_r = np.hstack([10*np.ones((n,1)).astype(np.float32), \
              np.linspace(5,10, n).reshape(n, 1).astype(np.float32)])
xt_lr = np.vstack([xt_l, xt_r])
x = [xt, xt_b, xt_lr]

y_ge  = -0.01*(-2*np.pi**2*np.cos(np.pi*xt[...,0,np.newaxis]+np.pi/5)+6*np.pi**2*np.sin(-2*np.pi*xt[...,0,np.newaxis]+np.pi/10)) * \
             (2*np.cos(np.pi*xt[...,1,np.newaxis]+np.pi/5)-3/2*np.sin(-2*np.pi*xt[...,1,np.newaxis]+np.pi/10)) + \
             (2*np.cos(np.pi*xt[...,0,np.newaxis]+np.pi/5)-3/2*np.sin(-2*np.pi*xt[...,0,np.newaxis]+np.pi/10)) * \
             (-2*np.pi*np.sin(np.pi*xt[...,1,np.newaxis]+np.pi/5)+3*np.pi*np.cos(-2*np.pi*xt[...,1,np.newaxis]+np.pi/10))
y_lr = func(xt_lr)
y_b = func(xt_b)
y= [y_ge, y_b, y_lr]


n_in = 2
n_out = 1
n_neu_x = 61
n_neu_t = 61
c_x = [5-0.5,10+.5]
c_t = [5-0.5,10+.5]

### Build RBN
rbn = RBF_Net(n_in, n_out, n_neu_x, n_neu_t, 5, c_x, c_t).build()

### Build PIRBN
pirbn = PIRBN(rbn)

### Train PIRBN
opt = Adam(pirbn, x, y, learning_rate = 0.001, maxiter=80001)
result=opt.fit()

### Visualise results
xt = np.zeros((101*101, 2)).astype(dtype='float32')
k = 0
dx = 5/(101-1)
for i in range(101):
    for j in range(101):
        xt[k, 0] = i * dx + 5
        xt[k, 1] = j * dx + 5
        k = k+1
y=rbn(xt)

fig1 = plt.figure(1)
ground_truth = func(xt)
plt.scatter(xt[:,0],xt[:,1],s=1,c=ground_truth,cmap='bwr',vmin=-6,vmax=11)
plt.axis('equal')
plt.colorbar()
plt.title('Ground Truth')
plt.show()

fig2 = plt.figure(2)
plt.scatter(xt[:,0],xt[:,1],s=1,c=y,cmap='bwr',vmin=-6,vmax=11)
plt.axis('equal')
plt.colorbar()
plt.title('PIRBN')
plt.show()

### Output results
scipy.io.savemat('out.mat', {'xt': xt, 'u': y.numpy()})
