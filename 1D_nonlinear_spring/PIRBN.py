import tensorflow as tf
from Dif_op import Dif

def PIRBN(rbn):
    """
    ====================================================================================================================

    This function is to initialize a PIRBN.

    ====================================================================================================================
    """

    ### declare PIRBN's inputs
    xy = tf.keras.layers.Input(shape=(1,))
    xy_b = tf.keras.layers.Input(shape=(1,))
    
    ### initialize the differential operators
    Dif_u = Dif(rbn)
    u = rbn(xy)
    u_b = rbn(xy_b)
    
    ### obtain partial derivatives of u with respect to x
    _, u_xx = Dif_u(xy)
    u_b_x, _ = Dif_u(xy_b)
    t = u_xx+4*u+tf.sin(u)
    
    ### build up the PIRBN
    pirbn = tf.keras.models.Model(inputs=[xy, xy_b], outputs=[t, u_b, u_b_x])
        
    return pirbn