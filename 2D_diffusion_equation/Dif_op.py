import tensorflow as tf

class Dif(tf.keras.layers.Layer):
    """
    ====================================================================================================================

    This is the class for calculating the differential terms of the RBN's output with respect to the RBN's input. We
    adopt the GradientTape function provided by the TensorFlow library to do the automatic differentiation.
    This class include 2 functions, including:
        1. __init__()         : Initialise the parameters for differential operator;
        2. call()             : Calculate the differential terms.

    ====================================================================================================================
    """

    def __init__(self, rbn, **kwargs):
        """
        ================================================================================================================

        This function is to initialise for differential operator.

        ----------------------------------------------------------------------------------------------------------------

        Name        Type                    Info.

        [rbn]       [Keras model]           : The radial basis network.
        
        ================================================================================================================
        """
        self.rbn = rbn
        super().__init__(**kwargs)

    def call(self, xt):
        """
        ================================================================================================================

        This function is to calculate the differential terms.

        ----------------------------------------------------------------------------------------------------------------

        Name        Type                    Info.

        [x]         [Keras model]           : The coordinate array;
        [temp]      [Keras tensor]          : The intermediate output from the RBN;
        [u]         [Keras tensor]          : The displacement predictions;
        [u_x]       [Keras tensor]          : The first-order derivative of the u with respect to the x;
        [u_xx]      [Keras tensor]          : The second-order derivative of the u with respect to the x.

        ================================================================================================================
        """

        x, t = (xt[..., i, tf.newaxis] for i in range(xt.shape[-1]))
        with tf.GradientTape(persistent=True) as gg:
            gg.watch(x)
            gg.watch(t)
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)
                g.watch(t)
                U = self.rbn(tf.concat([x, t], axis=-1))
            U_x = g.gradient(U, x)
            U_t = g.gradient(U, t)
            del g
        U_xx = gg.gradient(U_x, x)
        U_tt = gg.gradient(U_t, t)
        del gg
        
        return U_xx, U_x, U_t
