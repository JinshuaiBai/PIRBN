import numpy as np
import tensorflow as tf
from Cal_jac import cal_adapt

class Adam:

    def __init__(self, pirbn, x_train, y_train, learning_rate = 0.001, maxiter=10000):
        # set attributes
        self.pirbn = pirbn
        self.learning_rate = learning_rate
        self.x_train = [ tf.constant(x, dtype=tf.float32) for x in x_train ]
        self.y_train = [ tf.constant(y, dtype=tf.float32) for y in y_train ]
        self.maxiter = maxiter
        self.his_l1 = []
        self.his_l2 = []
        self.his_l3 = []

        self.iter = 0
        self.a_g = tf.constant(1.)
        self.a_b1 = tf.constant(1.)
        self.a_b2 = tf.constant(1.)

    def set_weights(self, flat_weights):
        # get model weights
        shapes = [ w.shape for w in self.pirbn.get_weights() ]
        # compute splitting indices
        split_ids = np.cumsum([ np.prod(shape) for shape in [0] + shapes ])
        # reshape weights
        weights = [ flat_weights[from_id:to_id].reshape(shape)
            for from_id, to_id, shape in zip(split_ids[:-1], split_ids[1:], shapes) ]
        # set weights to the model
        self.pirbn.set_weights(weights)

    @tf.function
    def Loss(self, x, y, a_g, a_b1, a_b2):
        with tf.GradientTape() as g:
            tmp = self.pirbn(x)
            l1 = 0.5 * tf.reduce_mean(tf.square(tmp[0]-y[0]))
            l2 = 0.5 * tf.reduce_mean(tf.square(tmp[1]-y[1]))
            l3 = 0.5 * tf.reduce_mean(tf.square(tmp[2]-y[2]))
            loss = l1 * a_g + l2 * a_b1 + l3 * a_b2
        grads = g.gradient(loss, self.pirbn.trainable_variables)
        return loss, grads, l1, l2, l3

    def evaluate(self, weights):
        # update weights
        self.set_weights(weights)
        # compute loss and gradients for weights
        loss, grads, l1, l2, l3 = self.Loss(self.x_train, self.y_train, self.a_g, self.a_b1, 
                                                        self.a_b2)
        self.his_l1.append(l1.numpy())
        self.his_l2.append(l2.numpy())
        self.his_l3.append(l3.numpy())
        if self.iter % 1000 == 0:
            self.a_g, self.a_b1, self.a_b2 = cal_adapt(self.pirbn, self.x_train)
            print('\ta_g =', self.a_g,'\ta_b1 =', self.a_b1,'\ta_b2 =', self.a_b2)
        if self.iter % 1000 == 0:
            print('Iter: ',self.iter,'\tL1 =',l1.numpy(),'\tL2 =',l2.numpy(),'\tL3 = ',l3.numpy())
        self.iter = self.iter+1
        # convert tf.Tensor to flatten ndarray
        loss = loss.numpy().astype('float64')
        grads = np.concatenate([ g.numpy().flatten() for g in grads ]).astype('float64')

        return loss, grads

    def fit(self):
        print('Optimizer: Adam')
        print('Initializing ...')
        # get initial weights as a flat vector
        initial_weights = np.concatenate(
            [ w.flatten() for w in self.pirbn.get_weights() ])
        print('Optimizer: Adam (maxiter={})'.format(self.maxiter))
        beta1 = 0.9
        beta2 = 0.999
        learning_rate = self.learning_rate
        eps=1e-8
        x0=initial_weights
        x=x0
        m=np.zeros_like(x)
        v=np.zeros_like(x)
        t=0
        b_w = 0
        for i in range(0,self.maxiter):
            loss, g = self.evaluate(x)
            m = (1 - beta1) * g + beta1 * m
            v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
            mhat = m / (1 - beta1**(i + 1))  # bias correction.
            vhat = v / (1 - beta2**(i + 1))
            x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)

        return loss, [self.his_l1, self.his_l2, self.his_l3], b_w
        
        return np.array(self.his_l1)+np.array(self.his_l2)+np.array(self.his_l3), \
               [self.his_l1, self.his_l2, self.his_l3]