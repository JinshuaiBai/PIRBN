import tensorflow as tf
from Dif_op import Dif

def PIRBN(rbn):
    """
    ====================================================================================================================

    This function is to initialize a PIRBN.

    ====================================================================================================================
    """

    ### declare PINN's inputs
    xt = tf.keras.layers.Input(shape=(2,))
    xt_b = tf.keras.layers.Input(shape=(2,))
    xt_lr = tf.keras.layers.Input(shape=(2,))
    
    ### initialize the differential operators
    Dif_u = Dif(rbn)
    u_b = rbn(xt_b)
    u_lr = rbn(xt_lr)

    ### obtain partial derivatives of u with respect to x
    u_xx, _, u_t = Dif_u(xt)
    t = u_t - 0.01*u_xx
    
    ### build up the PINN
    pirbn = tf.keras.models.Model(inputs=[xt, xt_b, xt_lr], outputs=[t, u_b, u_lr])
        
    return pirbn