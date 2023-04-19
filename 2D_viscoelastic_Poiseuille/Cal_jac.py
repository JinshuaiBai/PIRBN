import tensorflow as tf

def cal_adapt(pirbn, x):
    lamda_g1 = 0.
    lamda_g2 = 0.
    lamda_b1 = 0.
    lamda_b2 = 0.
    
    # in-domain
    n1 = x[0].shape[0]
    for i in range(n1):
        temp_x = [x[0][tf.newaxis,i,...],tf.zeros((1,2)),tf.zeros((1,2))]
        with tf.GradientTape(persistent = True) as gg:
            y = pirbn(temp_x)
        l1t=gg.gradient(y[0], pirbn.trainable_variables)
        l2t=gg.gradient(y[1], pirbn.trainable_variables)
        for j in l1t:
            lamda_g1 = lamda_g1 + tf.reduce_sum(j**2)/n1
        for j in l2t:
            lamda_g2 = lamda_g2 + tf.reduce_sum(j**2)/n1
            
    # bound
    n2 = x[1].shape[0]
    for i in range(n2):
        temp_x = [tf.zeros((1,2)),x[1][tf.newaxis,i,...],tf.zeros((1,2))]
        with tf.GradientTape(persistent = True) as gg:
            y = pirbn(temp_x)
        l1t=gg.gradient(y[2], pirbn.trainable_variables)
        for j in range(2):
            lamda_b1 = lamda_b1 + tf.reduce_sum(l1t[j]**2)/n2
            
    # left
    n3 = x[2].shape[0]
    for i in range(n3):
        temp_x = [tf.zeros((1,2)),tf.zeros((1,2)),x[2][tf.newaxis,i,...]]
        with tf.GradientTape(persistent = True) as gg:
            y = pirbn(temp_x)
        l1t=gg.gradient(y[3], pirbn.trainable_variables)
        for j in range(2):
            lamda_b2 = lamda_b2 + tf.reduce_sum(l1t[2+j]**2)/n3
    
    # calculate adapt factors
    temp = lamda_g1+lamda_g2+lamda_b1+lamda_b2
    lamda_g1 = temp/lamda_g1
    lamda_g2 = temp/lamda_g2
    lamda_b1 = temp/lamda_b1
    lamda_b2 = temp/lamda_b2
            
    return lamda_g1.numpy(), lamda_g2.numpy(), lamda_b1.numpy(), lamda_b2.numpy()