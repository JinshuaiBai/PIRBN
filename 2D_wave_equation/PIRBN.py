import tensorflow as tf
from Dif_op import Dif

def PIRBN(rbn):
    """
    ====================================================================================================================

    This function is to initialize a PIRBN.

    ====================================================================================================================
    """

    ### declare PINN's inputs
    xy = tf.keras.layers.Input(shape=(2,))
    xy_b = tf.keras.layers.Input(shape=(2,))
    xy_l = tf.keras.layers.Input(shape=(2,))
    
    ### initialize the differential operators
    Dif_u = Dif(rbn)
    u_b = rbn(xy_b)
    u_l = rbn(xy_l)

    ### obtain partial derivatives of u with respect to x
    u_xx, _, u_yy = Dif_u(xy)
    _, u_x_l, _ = Dif_u(xy_l)
    t = u_xx - 4*u_yy
    
    ### build up the PINN
    pirbn = tf.keras.models.Model(inputs=[xy, xy_b, xy_l], outputs=[t, u_b, u_x_l, u_l])
        
    return pirbn