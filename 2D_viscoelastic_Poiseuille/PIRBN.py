import tensorflow as tf
from Dif_op import Dif

def PIRBN(rbn_u, rbn_tau):
    """
    ====================================================================================================================

    This function is to initialize a PIRBN.

    ====================================================================================================================
    """
    
    rho = tf.constant(1./3)
    it0 = tf.constant(0.5)
    f = tf.constant(-1.5)
    lamda = tf.constant(1./3)
    
    ### declare PIRBN's inputs
    ty = tf.keras.layers.Input(shape=(2,))
    ty_b = tf.keras.layers.Input(shape=(2,))
    ty_l = tf.keras.layers.Input(shape=(2,))
    
    ### initialize the differential operators
    Dif_u = Dif(rbn_u)
    Dif_tau = Dif(rbn_tau)
    v_b = rbn_u(ty_b)
    tau_l = rbn_tau(ty_l)
    tau = rbn_tau(ty)
    
    ### obtain partial derivatives of u with respect to x
    u_t, u_y = Dif_u(ty)
    tau_t, tau_y = Dif_tau(ty)
    ge1 = rho*u_t + f - tau_y
    ge2 = lamda * tau_t + tau - it0 * u_y
    
    ### build up the PINN
    pirbn = tf.keras.models.Model(inputs=[ty, ty_b, ty_l], outputs=[ge1, ge2, v_b, tau_l])
        
    return pirbn