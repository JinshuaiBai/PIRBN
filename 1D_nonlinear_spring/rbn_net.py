import tensorflow as tf
import numpy as np

class RBN_Net:
    
    def __init__(self, n_in, n_out, n_neu, b, c):
        """
        ================================================================================================================

        This class is to build a radial basis network (RBN).

        ----------------------------------------------------------------------------------------------------------------

        Name        Type                    Info.

        [n_in]      [int]                   : Number of input of the RBN.
        [n_out]     [int]                   : Number of output of the RBN.
        [n_neu]     [int]                   : Number of neurons in the hidden layer.
        [b]         [array of float 32]     : Initial value for hyperparameter b.
        [c]         [array of float 32]     : Initial value for hyperparameter c.
        
        ================================================================================================================
        """
        
        self.n_in = n_in
        self.n_out = n_out
        self.n_neu = n_neu
        self.c = c
        self.b = b
        
    def net(self):
        ### Setup the input layer of the RBN
        x = tf.keras.layers.Input(shape=(self.n_in))
        l1 = RBF_layer1(self.n_neu, self.c)
        temp = l1(x)
        y = tf.keras.layers.Dense(self.n_out, kernel_initializer='LecunNormal', use_bias = False)(temp)
    
        ### Combine the input, hidden, and output layers to build up a RBN
        rbn = tf.keras.models.Model(inputs=x, outputs=y)
        
        return rbn
    
    def ini_ab(self, rbn):
        b = np.ones((1,self.n_neu))*self.b
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
        w_init = tf.random_normal_initializer()
        self.b = tf.Variable(
            initial_value=w_init(shape=(input_shape[-1], self.n_neu),
                                 dtype='float32'),
            trainable=True)
        
    def call(self, inputs):  # Defines the computation from inputs to outputs
        s = self.b*self.b
        temp_x = tf.matmul(inputs, tf.ones((1,self.n_neu)))
        x0 = tf.reshape(np.array(range(self.n_neu)).astype(dtype='float32'),(1,self.n_neu))*(self.c[1]-self.c[0])/(self.n_neu-1)+self.c[0]
        x_new = (temp_x-x0)*(temp_x-x0)
        return tf.exp(-x_new * s)
    
