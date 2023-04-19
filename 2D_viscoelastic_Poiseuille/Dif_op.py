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

    def call(self, ty):
        """
        ================================================================================================================

        This function is to calculate the differential terms.

        ----------------------------------------------------------------------------------------------------------------

        Name        Type                    Info.

        [ty]        [Keras tensor]          : The coordinate array;
        [U]         [Keras tensor]          : The displacement predictions;
        [U_t]       [Keras tensor]          : The first-order derivative of the U with respect to the t;
        [U_y]       [Keras tensor]          : The first-order derivative of the U with respect to the y.

        ================================================================================================================
        """

        t, y = (ty[..., i, tf.newaxis] for i in range(ty.shape[-1]))

        with tf.GradientTape(persistent=True) as g:
            g.watch(t)
            g.watch(y)
            U = self.rbn(tf.concat([t, y], axis=-1))
        U_t = g.gradient(U, t)
        U_y = g.gradient(U, y)
        del g
        
        return U_t, U_y
