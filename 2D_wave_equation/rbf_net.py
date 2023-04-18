import tensorflow as tf
import numpy as np

class RBF_Net:
    
    def __init__(self, n_in, n_out, n_neu_x, n_neu_y, b, c_x, c_y):
        """
        ================================================================================================================

        This class is to build a radial basis network (RBN).

        ----------------------------------------------------------------------------------------------------------------

        Name        Type                    Info.

        [n_in]      [int]                   : Number of input of the RBN.
        [n_out]     [int]                   : Number of output of the RBN.
        [n_neu]     [int]                   : Number of neurons in the hidden layer.
        [b]         [array of float 32]     : Initial value for hyperparameter b.
        [c_x]       [array of float 32]     : Initial x coordinate value for hyperparameter c.
        [c_y]       [array of float 32]     : Initial y coordinate value for hyperparameter c.
        
        ================================================================================================================
        """
        
        self.n_in = n_in
        self.n_out = n_out
        self.n_neu_x = n_neu_x
        self.n_neu_y = n_neu_y
        self.b = b
        self.c_x = c_x
        self.c_y = c_y
        c = np.zeros((2, n_neu_x*n_neu_y)).astype(dtype='float32')
        k = 0
        dx = (c_x[1] - c_x[0])/(n_neu_x-1)
        dy = (c_y[1] - c_y[0])/(n_neu_y-1)
        for i in range(n_neu_x):
            for j in range(n_neu_y):
                c[0, k] = i * dx + c_x[0]
                c[1, k] = j * dy + c_y[0]
                k = k+1
        self.c = tf.constant(c)
        
    def net(self):
        ### Setup the input layer of the RBN
        x = tf.keras.layers.Input(shape=(self.n_in))
        l1 = RBF_layer1(self.n_neu_x*self.n_neu_y, self.c)
        temp = l1(x)
        y = tf.keras.layers.Dense(self.n_out, kernel_initializer='LecunNormal', use_bias = False)(temp)
    
        ### Combine the input, hidden, and output layers to build up a RBN
        rbn = tf.keras.models.Model(inputs=x, outputs=y)
        
        return rbn
    
    def ini_ab(self, rbn):
        b = np.ones((1, self.n_neu_x*self.n_neu_y))*self.b
        a = rbn.get_weights()[1]
        weights = [b, a]
        rbn.set_weights(weights)

        return rbn
        
    def build(self):
        rbn = self.net()
        rbn = self.ini_ab(rbn)

        return rbn

#%%
class RBF_layer1(tf.keras.layers.Layer):
    """
    ================================================================================================================

    This class is to create the hidden layer of a radial basis network.

    ================================================================================================================
    """
    def __init__(self, n_neu, c):
        super(RBF_layer1, self).__init__()
        self.n_neu = n_neu
        self.c = c
    
    def build(self, input_shape):
        b_init = tf.random_normal_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(1, self.n_neu),
                                 dtype='float32'),
            trainable=True)
        
    def call(self, inputs):  # Defines the computation from inputs to outputs
        t2 = (tf.pow(inputs[..., 0, tf.newaxis], 2)+tf.pow(inputs[..., 1, tf.newaxis], 2))
        D = (tf.pow(self.c[tf.newaxis, 0, ...], 2)+tf.pow(self.c[tf.newaxis, 1, ...], 2))
        t1 = 2*tf.matmul(inputs, self.c)
        return tf.exp((t1-D-t2)*tf.pow(self.b, 2))
    
