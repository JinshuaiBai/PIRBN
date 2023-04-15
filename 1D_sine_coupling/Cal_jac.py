import scipy.optimize
import numpy as np
import tensorflow as tf
import scipy.io

def cal_adapt(pirbn, x):
    lamda_g = 0.
    lamda_b = 0.
    n_neu = len(pirbn.get_weights()[1])

    ### in-domain
    n1 = x[0].shape[0]
    for i in range(n1):
        temp_x = [x[0][np.newaxis,i,...],np.array([[0.]])]
        with tf.GradientTape(persistent = True) as gg:
            y = pirbn(temp_x)
        l1t=gg.gradient(y[0], pirbn.trainable_variables)
        for j in l1t:
            lamda_g = lamda_g + tf.reduce_sum(j**2)/n1
        temp = np.concatenate((l1t[0].numpy(),l1t[1].numpy().reshape((1,n_neu))),axis=1)
        if i==0:
            jac=temp
        else:
            jac=np.concatenate((jac,temp),axis=0)

    ### bound
    n2 = x[1].shape[0]
    for i in range(n2):
        temp_x = [np.array([[0.]]),x[1][tf.newaxis,i,...]]
        with tf.GradientTape(persistent = True) as gg:
            y = pirbn(temp_x)
        l1t=gg.gradient(y[1], pirbn.trainable_variables)
        for j in l1t:
            lamda_b = lamda_b + tf.reduce_sum(j**2)/n2

    ### calculate adapt factors
    temp = lamda_g+lamda_b
    lamda_g = temp/lamda_g
    lamda_b = temp/lamda_b
            
    return lamda_g.numpy(), lamda_b.numpy(), jac