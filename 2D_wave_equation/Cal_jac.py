import tensorflow as tf

def cal_adapt(pirbn, x):
    lamda_g = 1.
    lamda_b = 1.
    lamda_l_1 = 0.
    lamda_l_2 = 0.
    n_neu = len(pirbn.get_weights()[1])
    
    # in-domain
    n1 = x[0].shape[0]
    for i in range(n1):
        temp_x = [x[0][tf.newaxis,i,...],x[0][tf.newaxis,i,...],x[0][tf.newaxis,i,...]]
        with tf.GradientTape(persistent = True) as gg:
            y = pirbn(temp_x)
        l1t=gg.gradient(y[0], pirbn.trainable_variables)
        for j in l1t:
            lamda_g = lamda_g + tf.reduce_sum(j**2)/n1
            
    # bound
    n2 = x[1].shape[0]
    for i in range(n2):
        temp_x = [x[1][tf.newaxis,i,...],x[1][tf.newaxis,i,...],x[1][tf.newaxis,i,...]]
        with tf.GradientTape(persistent = True) as gg:
            y = pirbn(temp_x)
        l1t=gg.gradient(y[1], pirbn.trainable_variables)
        for j in l1t:
            lamda_b = lamda_b + tf.reduce_sum(j**2)/n2
            
    # left
    n3 = x[2].shape[0]
    for i in range(n3):
        temp_x = [x[2][tf.newaxis,i,...],x[2][tf.newaxis,i,...],x[2][tf.newaxis,i,...]]
        with tf.GradientTape(persistent = True) as gg:
            y = pirbn(temp_x)
        l1t=gg.gradient(y[2], pirbn.trainable_variables)
        l2t=gg.gradient(y[3], pirbn.trainable_variables)
        for j in l1t:
            lamda_l_1 = lamda_l_1 + tf.reduce_sum(j**2)/n3
        for j in l2t:
            lamda_l_2 = lamda_l_2 + tf.reduce_sum(j**2)/n3
    
    # calculate adapt factors
    temp = lamda_g+lamda_b+lamda_l_1+lamda_l_2
    lamda_g = temp/lamda_g
    lamda_b = temp/lamda_b
    lamda_l_1 = temp/lamda_l_1
    lamda_l_2 = temp/lamda_l_2
            
    return lamda_g.numpy(), lamda_b.numpy(), lamda_l_1.numpy(), lamda_l_2.numpy()