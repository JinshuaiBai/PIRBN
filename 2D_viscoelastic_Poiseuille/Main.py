import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from PIRBN import PIRBN
from OPT import Adam
import scipy.io
from rbf_net import RBF_Net
import matplotlib.pyplot as plt

### Define the number of sample points
ny = 61
nx = 61
ns = ny

### Define the sample points' interval
dx = 1./(ny-1)

### Initialize sample points' coordinates
ty = np.zeros((nx*ny,2)).astype(dtype='float32')
k = 0
dy = 1/(ny-1)
for i in range(nx):
    for j in range(ny):
        ty[k,0] = i * dy
        ty[k,1] = j * dy
        k=k+1
ty_t = np.hstack([np.linspace(0,1, nx).reshape(nx, 1).astype(np.float32), \
                  1*np.ones((nx,1)).astype(np.float32)])
ty_b = np.hstack([np.linspace(0,1, nx).reshape(nx, 1).astype(np.float32), \
                  0*np.ones((nx,1)).astype(np.float32)])
ty_l = np.hstack([0*np.ones((ny,1)).astype(np.float32), \
              np.linspace(0,1, ny).reshape(ny, 1).astype(np.float32)])
ty_r = np.hstack([np.ones((ny,1)).astype(np.float32), \
              np.linspace(0,1, ny).reshape(ny, 1).astype(np.float32)])
ty_bound = np.vstack([ty_t, ty_b, ty_l])
x = [ty, ty_bound, ty_l]
y=[0.]

n_in=2
n_out=1
n_neu_t=61
n_neu_y=61
c_t=[-0.1,1.1]
c_y=[-0.1,1.1]

rbn_u = RBF_Net(n_in, n_out, n_neu_t, n_neu_y, 5, c_t, c_y).build()
rbn_tau = RBF_Net(n_in, n_out, n_neu_t, n_neu_y, 5, c_t, c_y).build()

pirbn = PIRBN(rbn_u, rbn_tau)
opt = Adam(pirbn, x, y, learning_rate = 0.001, maxiter=10001)
result=opt.fit()

### Visualise results
ny = 51
nt = 201
ns = ny
dy = 1./(ny-1)
ty = np.zeros((nt*ny,2)).astype(dtype='float32')
k = 0
dx = 1/((ny-1))
for i in range(nt):
    for j in range(ny):
        ty[k,0] = i * dx + 0
        ty[k,1] = j * dx - .5
        k=k+1

u=rbn_u(ty)
tau=rbn_tau(ty)
ground_truth_u, ground_truth_tau = func(ty)

fig1 = plt.figure(1)
plt.scatter(ty[:,0],ty[:,1],s=1,c=ground_truth_u,cmap='jet',vmin=0,vmax=1)
plt.axis('equal')
plt.colorbar()
plt.title('Velocity (Ground Truth)')

fig2 = plt.figure(2)
plt.scatter(ty[:,0],ty[:,1],s=1,c=u,cmap='jet',vmin=0,vmax=1)
plt.axis('equal')
plt.colorbar()
plt.title('Velocity (PIRBN)')
plt.show()

fig3 = plt.figure(3)
plt.scatter(ty[:,0],ty[:,1],s=1,c=ground_truth_tau,cmap='jet',vmin=-1,vmax=1)
plt.axis('equal')
plt.colorbar()
plt.title('Shear stress (Ground Truth)')

fig4 = plt.figure(4)
plt.scatter(ty[:,0],ty[:,1],s=1,c=tau,cmap='jet',vmin=-1,vmax=1)
plt.axis('equal')
plt.colorbar()
plt.title('Shear stress (PIRBN)')
plt.show()

scipy.io.savemat('out.mat', {'ty': ty, 'u': u.numpy(), 'tau': tau.numpy()})
