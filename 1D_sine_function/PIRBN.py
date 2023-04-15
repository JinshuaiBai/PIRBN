import tensorflow as tf
from Dif_op import Dif

def PIRBN(rbn):
    """
    ====================================================================================================================

    This function is to initialize a PIRBN.

    ====================================================================================================================
    """

    ### declare PINN's inputs
    xy = tf.keras.layers.Input(shape=(1,))
    xy_b = tf.keras.layers.Input(shape=(1,))
    
    ### initialize the differential operators
    Dif_u = Dif(rbn)
    u_b = rbn(xy_b)
    
    ### obtain partial derivatives of u with respect to x
    _, u_xx = Dif_u(xy)
    
    ### build up the PINN
    pirbn = tf.keras.models.Model(inputs=[xy, xy_b], outputs=[u_xx, u_b])
        
    return pirbn