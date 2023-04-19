import tensorflow as tf

def cal_adapt(pirbn, x):
    lamda_g = 0.
    lamda_b1 = 0.
    lamda_b2 = 0.
    n_neu = len(pirbn.get_weights()[1])
    
    # in-domain
    n1 = x[0].shape[0]
    for i in range(n1):
        temp_x = [x[0][tf.newaxis,i,...],tf.zeros((1,2)),tf.zeros((1,2))]
        with tf.GradientTape(persistent = True) as gg:
            y = pirbn(temp_x)
        l1t=gg.gradient(y[0], pirbn.trainable_variables)
        for j in l1t:
            lamda_g = lamda_g + tf.reduce_sum(j**2)/n1
            
    # bound
    n2 = x[1].shape[0]
    for i in range(n2):
        temp_x = [tf.zeros((1,2)),x[1][tf.newaxis,i,...],tf.zeros((1,2))]
        with tf.GradientTape(persistent = True) as gg:
            y = pirbn(temp_x)
        l1t=gg.gradient(y[1], pirbn.trainable_variables)
        for j in l1t:
            lamda_b1 = lamda_b1 + tf.reduce_sum(j**2)/n2
            
    # left
    n3 = x[2].shape[0]
    for i in range(n3):
        temp_x = [tf.zeros((1,2)),tf.zeros((1,2)),x[2][tf.newaxis,i,...]]
        with tf.GradientTape(persistent = True) as gg:
            y = pirbn(temp_x)
        l1t=gg.gradient(y[2], pirbn.trainable_variables)
        for j in l1t:
            lamda_b2 = lamda_b2 + tf.reduce_sum(j**2)/n3
    
    # calculate adapt factors
    temp = lamda_g+lamda_b1+lamda_b2
    lamda_g = temp/lamda_g
    lamda_b1 = temp/lamda_b1
    lamda_b2 = temp/lamda_b2
            
    return lamda_g.numpy(), lamda_b1.numpy(), lamda_b2.numpy()