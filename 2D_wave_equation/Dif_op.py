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

    def call(self, xy):
        """
        ================================================================================================================

        This function is to calculate the differential terms.

        ----------------------------------------------------------------------------------------------------------------

        Name        Type                    Info.

        [xy]        [Keras tensor]          : The coordinate array;
        [temp]      [Keras tensor]          : The intermediate output from the RBN;
        [U]         [Keras tensor]          : The displacement predictions;
        [U_x]       [Keras tensor]          : The first-order derivative of the U with respect to the x;
        [U_xx]      [Keras tensor]          : The second-order derivative of the U with respect to the x;
        [U_yy]      [Keras tensor]          : The second-order derivative of the U with respect to the y.

        ================================================================================================================
        """

        x, y = (xy[..., i, tf.newaxis] for i in range(xy.shape[-1]))
        with tf.GradientTape(persistent=True) as gg:
            gg.watch(x)
            gg.watch(y)
            with tf.GradientTape(persistent=True) as g:
                g.watch(x)
                g.watch(y)
                U = self.rbn(tf.concat([x, y], axis=-1))
            U_x = g.gradient(U, x)
            U_y = g.gradient(U, y)
            del g
        U_xx = gg.gradient(U_x, x)
        U_yy = gg.gradient(U_y, y)
        del gg
        
        return U_xx, U_x, U_yy
