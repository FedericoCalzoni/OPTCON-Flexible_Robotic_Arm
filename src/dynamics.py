import numpy as np
from parameters import M1, M2, L1, R1, R2, I1, I2, G, F1, F2, dt

F = np.array([[F1, 0],
              [ 0 ,F2]])

Z_2x2 = np.zeros((2,2))

def compute_inertia_matrix(theta2):
    """Compute the inertia matrix M."""
    cos_theta2 = np.cos(theta2)
    m11 = I1 + I2 + M1 * R1**2 + M2 * (L1**2 + R2**2) + 2 * M2 * L1 * R2 * cos_theta2
    m12 = I2 + M2 * R2**2 + M2 * L1 * R2 * cos_theta2
    m21 = m12
    m22 = I2 + M2 * R2**2
    return np.array([[m11, m12], [m21, m22]])

def compute_coriolis(theta2, dtheta1, dtheta2):
    """Compute the Coriolis and centrifugal forces matrix C."""
    sin_theta2 = np.sin(theta2)
    c1 = -M2 * L1 * R2 * dtheta2 * sin_theta2 * (dtheta2 + 2 * dtheta1)
    c2 = M2 * L1 * R2 * sin_theta2 * dtheta1**2
    return np.array([[c1], [c2]])

def compute_gravity(theta1, theta2):
    """Compute the gravity forces matrix G."""
    sin_theta1 = np.sin(theta1)
    sin_theta1_theta2 = np.sin(theta1 + theta2)
    g1 = G * (M1 * R1 + M2 * L1) * sin_theta1 + G * M2 * R2 * sin_theta1_theta2
    g2 = G * M2 * R2 * sin_theta1_theta2
    return np.array([[g1], [g2]])

def dynamics(x, u, dt=1e-3):

    dtheta1 = x[0].item()
    dtheta2 = x[1].item()
    theta1 = x[2].item()
    theta2 = x[3].item()

    tau1 = u[0].item()
    tau2 = u[1].item()
    #x = np.array([dtheta1, dtheta2, theta1, theta2])

    # Compute matrices
    M = compute_inertia_matrix(theta2)
    M_inv = np.linalg.inv(M)
    C = compute_coriolis(theta2, dtheta1, dtheta2)
    G = compute_gravity(theta1, theta2)
    
    A = np.block([[ -M_inv @ F, Z_2x2 ], 
                  [ np.eye(2), Z_2x2 ]])
       
    
    M_inv_ext = np.block([
        [M_inv, Z_2x2],
        [Z_2x2, Z_2x2]
    ])

    B = M_inv_ext
    
    C_ext = np.block([
        [C],
        [np.zeros((2, 1))]  # Ensure the zeros are in the same shape (2, 1)
    ])
    
    G_ext = np.block([
        [G],
        [np.zeros((2, 1))]
    ])
    
    x_rev = x.reshape(-1, 1) 
    u_rev = u.reshape(-1, 1)

    ct = - M_inv_ext @ (C_ext + G_ext)

    x_dot = A @ x_rev + B @ u_rev + ct
    
    x_new = (x_rev + dt * x_dot).flatten()
    
    #dfx = jacobian_x_dot_wrt_x(dtheta1, dtheta2, theta1, theta2, u[0].item())
    #dfu = jacobian_x_dot_wrt_u(theta2)
    #
    #Hxx = hessian_x_dot_wrt_x(dtheta1, dtheta2, theta1, theta2, u[0].item())
    #Huu = hessian_x_dot_wrt_u()
    #Hxu = hessian_x_dot_wrt_xu(theta2)
        
    return x_new


# TODO: give a better name to this funciton, it is the jacobian of the gravity
def jacobian(theta1, theta2):
    JG_11 = G*M2*R2*np.cos(theta1 + theta2) + G*(L1*M2 + M1*R1)*np.cos(theta1)
    JG_12 = G*M2*R2*np.cos(theta1 + theta2)
    JG_21 = JG_12
    JG_22 = JG_12
    JG_13 = -1
    JG_23 = 0
    return np.array([[JG_11 , JG_12, JG_13], [JG_21 , JG_22, JG_23]])

#def jacobian_x_dot_wrt_x(dtheta1, dtheta2, theta1, theta2, tau1):
#    dfx = np.zeros((4, 4))
#    # Common Subexpressions:
#    tmp0 = pow(R2, 2)
#    tmp1 = M2*tmp0
#    tmp2 = I2 + tmp1
#    tmp3 = F1*tmp2
#    tmp4 = np.sin(theta2)
#    tmp5 = L1*M2
#    tmp6 = R2*tmp5
#    tmp7 = tmp4*tmp6
#    tmp8 = dtheta2*tmp7
#    tmp9 = 2*tmp8
#    tmp10 = np.cos(theta2)
#    tmp11 = tmp10*tmp6
#    tmp12 = tmp11 + tmp2
#    tmp13 = 2*dtheta1
#    tmp14 = tmp13*tmp7
#    tmp15 = pow(L1, 2)
#    tmp16 = M2*tmp15
#    tmp17 = M1*pow(R1, 2)
#    tmp18 = pow(M2, 2)*tmp0
#    tmp19 = tmp15*tmp18
#    tmp20 = I1*I2 + I1*tmp1 + I2*tmp16 + I2*tmp17 + tmp1*tmp17 + tmp19*pow(tmp4, 2)
#    tmp21 = 1.0/tmp20
#    tmp22 = F2*tmp12
#    tmp23 = dtheta2 + tmp13
#    tmp24 = -tmp23*tmp7 - tmp8
#    tmp25 = theta1 + theta2
#    tmp26 = G*np.cos(tmp25)
#    tmp27 = M2*R2
#    tmp28 = tmp26*tmp27
#    tmp29 = G*(M1*R1 + tmp5)
#    tmp30 = tmp28 + tmp29*np.cos(theta1)
#    tmp31 = F2*tmp8
#    tmp32 = G*np.sin(tmp25)
#    tmp33 = L1*pow(dtheta1, 2)
#    tmp34 = tmp32 + tmp33*tmp4
#    tmp35 = L1*tmp18*tmp34*tmp4
#    tmp36 = -dtheta2*tmp11*tmp23 + tmp28
#    tmp37 = tmp10*tmp33 + tmp26
#    tmp38 = -tmp23*tmp8 + tmp27*tmp32 + tmp29*np.sin(theta1)
#    tmp39 = 2*tmp10*tmp19*tmp4/pow(tmp20, 2)
#    tmp40 = I1 + 2*tmp11 + tmp16 + tmp17 + tmp2
#    tmp41 = F2*tmp40
#    tmp42 = tmp27*tmp40
#    # Jacobian Elements:
#    dfx[0, 0] = tmp21*(tmp12*tmp14 + tmp2*tmp9 - tmp3)
#    dfx[0, 1] = tmp21*(-tmp2*tmp24 + tmp22)
#    dfx[0, 2] = tmp21*(tmp12*tmp28 - tmp2*tmp30)
#    dfx[0, 3] = tmp21*(M2*R2*tmp12*tmp37 - tmp2*tmp36 - tmp31 - tmp35) \
#        - tmp39*(-dtheta1*tmp3 + dtheta2*tmp22 + tau1*tmp2 + tmp12*tmp27*tmp34 - tmp2*tmp38)
#    dfx[1, 0] = tmp21*(F1*tmp12 - tmp12*tmp9 - tmp14*tmp40)
#    dfx[1, 1] = tmp21*(tmp12*tmp24 - tmp41)
#    dfx[1, 2] = tmp21*(tmp12*tmp30 - tmp28*tmp40)
#    dfx[1, 3] = tmp21*(-F1*dtheta1*tmp7 + tau1*tmp7 + tmp12*tmp36 + 2*tmp31 \
#        + 2*tmp35 - tmp37*tmp42 - tmp38*tmp7) - tmp39*(F1*dtheta1*tmp12 \
#            - dtheta2*tmp41 - tau1*tmp12 + tmp12*tmp38 - tmp34*tmp42)
#    dfx[2, 0] = 1
#    dfx[3, 1] = 1
#    return dfx
#
#def jacobian_x_dot_wrt_u(theta2):
#    dfu = np.zeros((4, 4))
#    # Common Subexpressions:
#    tmp0 = pow(R2, 2)
#    tmp1 = M2*tmp0
#    tmp2 = I2 + tmp1
#    tmp3 = pow(L1, 2)
#    tmp4 = M2*tmp3
#    tmp5 = M1*pow(R1, 2)
#    tmp6 = 1.0/(I1*I2 + I1*tmp1 + I2*tmp4 + I2*tmp5 \
#        + pow(M2, 2)*tmp0*tmp3*pow(np.sin(theta2), 2) + tmp1*tmp5)
#    tmp7 = L1*M2*R2*np.cos(theta2)
#    tmp8 = tmp6*(-tmp2 - tmp7)
#    # Jacobian Elements:
#    dfu[0, 0] = tmp2*tmp6
#    dfu[0, 1] = tmp8
#    dfu[1, 0] = tmp8
#    dfu[1, 1] = tmp6*(I1 + tmp2 + tmp4 + tmp5 + 2*tmp7)
#    return dfu

import numpy as np

def compute_jacobian_simplified(x_new, u):
    dtheta1 = x_new[0].item()
    dtheta2 = x_new[1].item()
    theta1 = x_new[2].item()
    theta2 = x_new[3].item()
    tau1 = u[0].item()
    tau2 = u[1].item()
    A = compute_jacobian_x(tau1, tau2, dtheta1, dtheta2, theta1, theta2)
    B = compute_jacobian_u(theta2)
    return A, B

def compute_jacobian_x(tau1, tau2, dtheta1, dtheta2, theta1, theta2):
    # Calcoli intermedi comuni
    sin_theta2 = np.sin(theta2)
    cos_theta2 = np.cos(theta2)
    sin_theta1_theta2 = np.sin(theta1 + theta2)
    cos_theta1_theta2 = np.cos(theta1 + theta2)

    denom = (I1 * I2 + I1 * M2 * R2**2 + I2 * L1**2 * M2 + I2 * M1 * R1**2 +
             L1**2 * M2**2 * R2**2 * sin_theta2**2 + M1 * M2 * R1**2 * R2**2)

    # Matrice Jacobiana inizializzata a zero
    jacobian_x_new_wrt_x = np.zeros((4, 4))

    # Elemento [0,0]
    jacobian_x_new_wrt_x[0, 0] = (dt * (-F1 * (I2 + M2 * R2**2) +
                                        2 * L1 * M2 * R2 * dtheta1 * (I2 + L1 * M2 * R2 * cos_theta2 + M2 * R2**2) * sin_theta2 +
                                        2 * L1 * M2 * R2 * dtheta2 * (I2 + M2 * R2**2) * sin_theta2) /
                                  denom + 1)

    # Elemento [0,1]
    jacobian_x_new_wrt_x[0, 1] = (dt * (F2 * (I2 + L1 * M2 * R2 * cos_theta2 + M2 * R2**2) -
                                        (I2 + M2 * R2**2) * (-L1 * M2 * R2 * dtheta2 * sin_theta2 -
                                                             L1 * M2 * R2 * (2 * dtheta1 + dtheta2) * sin_theta2)) /
                                  denom)

    # Elemento [0,2]
    jacobian_x_new_wrt_x[0, 2] = (dt * (G * M2 * R2 * (I2 + L1 * M2 * R2 * cos_theta2 + M2 * R2**2) * cos_theta1_theta2 -
                                        (I2 + M2 * R2**2) * (G * M2 * R2 * cos_theta1_theta2 +
                                                             G * (L1 * M2 + M1 * R1) * np.cos(theta1))) /
                                  denom)

    # Elemento [0,3]
    jacobian_x_new_wrt_x[0, 3] = (-2 * L1**2 * M2**2 * R2**2 * dt * (
        -F1 * dtheta1 * (I2 + M2 * R2**2) +
        F2 * dtheta2 * (I2 + L1 * M2 * R2 * cos_theta2 + M2 * R2**2) +
        M2 * R2 * (G * sin_theta1_theta2 + L1 * dtheta1**2 * sin_theta2) * (I2 + L1 * M2 * R2 * cos_theta2 + M2 * R2**2) +
        tau1 * (I2 + M2 * R2**2) - tau2 * (I2 + L1 * M2 * R2 * cos_theta2 + M2 * R2**2) -
        (I2 + M2 * R2**2) * (G * M2 * R2 * sin_theta1_theta2 + G * (L1 * M2 + M1 * R1) * np.sin(theta1) -
                             L1 * M2 * R2 * dtheta2 * (2 * dtheta1 + dtheta2) * sin_theta2)) * sin_theta2 * cos_theta2 /
        denom**2 +
        dt * (-F2 * L1 * M2 * R2 * dtheta2 * sin_theta2 -
              L1 * M2**2 * R2**2 * (G * sin_theta1_theta2 + L1 * dtheta1**2 * sin_theta2) * sin_theta2 +
              L1 * M2 * R2 * tau2 * sin_theta2 +
              M2 * R2 * (G * cos_theta1_theta2 + L1 * dtheta1**2 * cos_theta2) * (I2 + L1 * M2 * R2 * cos_theta2 + M2 * R2**2) -
              (I2 + M2 * R2**2) * (G * M2 * R2 * cos_theta1_theta2 -
                                   L1 * M2 * R2 * dtheta2 * (2 * dtheta1 + dtheta2) * cos_theta2)) /
        denom)

    # Elemento [1,0]
    jacobian_x_new_wrt_x[1, 0] = (dt * (F1 * (I2 + L1 * M2 * R2 * cos_theta2 + M2 * R2**2) -
                                        2 * L1 * M2 * R2 * dtheta1 * (I1 + I2 + L1**2 * M2 + 2 * L1 * M2 * R2 * cos_theta2 +
                                                                      M1 * R1**2 + M2 * R2**2) * sin_theta2 -
                                        2 * L1 * M2 * R2 * dtheta2 * (I2 + L1 * M2 * R2 * cos_theta2 + M2 * R2**2) * sin_theta2) /
                                  denom)

    # Elemento [1,1]
    jacobian_x_new_wrt_x[1, 1] = (dt * (-F2 * (I1 + I2 + L1**2 * M2 + 2 * L1 * M2 * R2 * cos_theta2 + M1 * R1**2 + M2 * R2**2) +
                                        (-L1 * M2 * R2 * dtheta2 * sin_theta2 -
                                         L1 * M2 * R2 * (2 * dtheta1 + dtheta2) * sin_theta2) *
                                        (I2 + L1 * M2 * R2 * cos_theta2 + M2 * R2**2)) /
                                  denom + 1)

    # Elemento [1,2]
    jacobian_x_new_wrt_x[1, 2] = (dt * (-G * M2 * R2 * (I1 + I2 + L1**2 * M2 + 2 * L1 * M2 * R2 * cos_theta2 + M1 * R1**2 + M2 * R2**2) * cos_theta1_theta2 +
                                        (G * M2 * R2 * cos_theta1_theta2 + G * (L1 * M2 + M1 * R1) * np.cos(theta1)) *
                                        (I2 + L1 * M2 * R2 * cos_theta2 + M2 * R2**2)) /
                                  denom)

    # Elemento [1,3]
    jacobian_x_new_wrt_x[1, 3] = (dt * (-F2 * L1 * M2 * R2 * dtheta2 * sin_theta2 -
                                        L1 * M2**2 * R2**2 * (G * sin_theta1_theta2 + L1 * dtheta1**2 * sin_theta2) * sin_theta2 +
                                        L1 * M2 * R2 * tau2 * sin_theta2 +
                                        M2 * R2 * (G * cos_theta1_theta2 + L1 * dtheta1**2 * cos_theta2) * (I2 + L1 * M2 * R2 * cos_theta2 + M2 * R2**2) -
                                        (I2 + M2 * R2**2) * (G * M2 * R2 * cos_theta1_theta2 -
                                                             L1 * M2 * R2 * dtheta2 * (2 * dtheta1 + dtheta2) * cos_theta2)) /
                                  denom)

    # Elementi rimanenti
    jacobian_x_new_wrt_x[2, 0] = dt
    jacobian_x_new_wrt_x[2, 1] = 0
    jacobian_x_new_wrt_x[2, 2] = 1
    jacobian_x_new_wrt_x[2, 3] = 0
    jacobian_x_new_wrt_x[3, 0] = 0
    jacobian_x_new_wrt_x[3, 1] = dt
    jacobian_x_new_wrt_x[3, 2] = 0
    jacobian_x_new_wrt_x[3, 3] = 1

    return jacobian_x_new_wrt_x

def compute_jacobian_u(theta2):
    # Calcoli intermedi comuni
    sin_theta2 = np.sin(theta2)
    cos_theta2 = np.cos(theta2)
    pow_sin_theta2 = np.sin(theta2)**2
    pow_cos_theta2 = np.cos(theta2)**2

    denom = (I1 * I2 + I1 * M2 * R2**2 + I2 * L1**2 * M2 + I2 * M1 * R1**2 +
             L1**2 * M2**2 * R2**2 * pow_sin_theta2 + M1 * M2 * R1**2 * R2**2)

    # Matrice Jacobiana inizializzata a zero
    jacobian_x_new_wrt_u = np.zeros((4, 4))

    # Elemento [0,0]
    jacobian_x_new_wrt_u[0, 0] = dt * (I2 + M2 * R2**2) / denom

    # Elemento [0,1]
    jacobian_x_new_wrt_u[0, 1] = dt * (-I2 - L1 * M2 * R2 * cos_theta2 - M2 * R2**2) / denom

    # Elemento [1,0]
    jacobian_x_new_wrt_u[1, 0] = dt * (-I2 - L1 * M2 * R2 * cos_theta2 - M2 * R2**2) / denom

    # Elemento [1,1]
    jacobian_x_new_wrt_u[1, 1] = dt * (I1 + I2 + L1**2 * M2 + 2 * L1 * M2 * R2 * cos_theta2 + M1 * R1**2 + M2 * R2**2) / denom

    # Elementi rimanenti
    jacobian_x_new_wrt_u[0, 2] = 0
    jacobian_x_new_wrt_u[0, 3] = 0
    jacobian_x_new_wrt_u[1, 2] = 0
    jacobian_x_new_wrt_u[1, 3] = 0
    jacobian_x_new_wrt_u[2, 0] = 0
    jacobian_x_new_wrt_u[2, 1] = 0
    jacobian_x_new_wrt_u[2, 2] = 0
    jacobian_x_new_wrt_u[2, 3] = 0
    jacobian_x_new_wrt_u[3, 0] = 0
    jacobian_x_new_wrt_u[3, 1] = 0
    jacobian_x_new_wrt_u[3, 2] = 0
    jacobian_x_new_wrt_u[3, 3] = 0

    return jacobian_x_new_wrt_u



def hessian_x_dot_wrt_x(dtheta1, dtheta2, theta1, theta2, tau1):
    Hxx = np.zeros((4, 4, 4))
    # Common Subexpressions:
    tmp0 = np.cos(theta2)
    tmp1 = L1*M2
    tmp2 = R2*tmp1
    tmp3 = tmp0*tmp2
    tmp4 = pow(R2, 2)
    tmp5 = M2*tmp4
    tmp6 = I2 + tmp5
    tmp7 = tmp3 + tmp6
    tmp8 = pow(L1, 2)
    tmp9 = M2*tmp8
    tmp10 = M1*pow(R1, 2)
    tmp11 = np.sin(theta2)
    tmp12 = pow(tmp11, 2)
    tmp13 = pow(M2, 2)
    tmp14 = tmp13*tmp4
    tmp15 = tmp14*tmp8
    tmp16 = tmp12*tmp15
    tmp17 = I1*I2 + I1*tmp5 + I2*tmp10 + I2*tmp9 + tmp10*tmp5 + tmp16
    tmp18 = 1.0/tmp17
    tmp19 = tmp11*tmp2
    tmp20 = 2*tmp19
    tmp21 = tmp18*tmp20
    tmp22 = tmp21*tmp7
    tmp23 = tmp21*tmp6
    tmp24 = dtheta2*tmp3
    tmp25 = 2*tmp6
    tmp26 = 2*dtheta1
    tmp27 = tmp26*tmp7
    tmp28 = F1*tmp6
    tmp29 = dtheta2*tmp19
    tmp30 = 2*tmp15
    tmp31 = pow(tmp17, -2)
    tmp32 = tmp0*tmp11*tmp31
    tmp33 = tmp30*tmp32
    tmp34 = tmp18*(-tmp16*tmp26 + tmp24*tmp25 + tmp27*tmp3) - tmp33*(tmp19*tmp27 + tmp25*tmp29 - tmp28)
    tmp35 = F2*tmp19
    tmp36 = dtheta2 + tmp26
    tmp37 = -tmp24 - tmp3*tmp36
    tmp38 = F2*tmp7
    tmp39 = -tmp19*tmp36 - tmp29
    tmp40 = tmp18*(-tmp35 - tmp37*tmp6) - tmp33*(tmp38 - tmp39*tmp6)
    tmp41 = theta1 + theta2
    tmp42 = np.sin(tmp41)
    tmp43 = G*tmp42
    tmp44 = M2*R2
    tmp45 = tmp43*tmp44
    tmp46 = tmp45*tmp7
    tmp47 = G*(M1*R1 + tmp1)
    tmp48 = tmp45 + tmp47*np.sin(theta1)
    tmp49 = -tmp48
    tmp50 = G*np.cos(tmp41)
    tmp51 = L1*tmp14
    tmp52 = tmp11*tmp51
    tmp53 = tmp50*tmp52
    tmp54 = tmp44*tmp50
    tmp55 = tmp47*np.cos(theta1) + tmp54
    tmp56 = tmp18*(G*M2*R2*tmp42*tmp6 - tmp46 - tmp53) - tmp33*(tmp54*tmp7 - tmp55*tmp6)
    tmp57 = dtheta2*tmp35
    tmp58 = L1*pow(dtheta1, 2)
    tmp59 = tmp11*tmp58 + tmp43
    tmp60 = tmp51*tmp59
    tmp61 = tmp11*tmp60
    tmp62 = -tmp24*tmp36 + tmp54
    tmp63 = tmp0*tmp58 + tmp50
    tmp64 = 4*tmp15*tmp32
    tmp65 = F2*tmp24
    tmp66 = tmp0*tmp60
    tmp67 = tmp52*tmp63
    tmp68 = -tmp29*tmp36
    tmp69 = -tmp45 - tmp68
    tmp70 = -tmp59
    tmp71 = 2*tmp16
    tmp72 = tmp48 + tmp68
    tmp73 = -dtheta1*tmp28 + dtheta2*tmp38 + tau1*tmp6 + tmp44*tmp59*tmp7 - tmp6*tmp72
    tmp74 = tmp31*tmp73
    tmp75 = pow(tmp0, 2)
    tmp76 = tmp30*tmp75
    tmp77 = 8*pow(L1, 4)*pow(M2, 4)*pow(R2, 4)*tmp12*tmp75/pow(tmp17, 3)
    tmp78 = I1 + tmp10 + 2*tmp3 + tmp6 + tmp9
    tmp79 = -tmp22
    tmp80 = 2*tmp7
    tmp81 = tmp26*tmp78
    tmp82 = F1*tmp19
    tmp83 = tmp18*(4*dtheta1*tmp12*tmp13*tmp4*tmp8 + 2*dtheta2*tmp12*tmp13*tmp4*tmp8 \
        - tmp24*tmp80 - tmp3*tmp81 - tmp82) - tmp33*(F1*tmp7 - tmp19*tmp81 - tmp29*tmp80)
    tmp84 = F2*tmp78
    tmp85 = tmp18*(-tmp19*tmp39 + 2*tmp35 + tmp37*tmp7) - tmp33*(tmp39*tmp7 - tmp84)
    tmp86 = tmp45*tmp78
    tmp87 = tmp18*(-tmp19*tmp55 - tmp46 + 2*tmp53 + tmp86) - tmp33*(-tmp54*tmp78 + tmp55*tmp7)
    tmp88 = tmp44*tmp78
    tmp89 = F1*dtheta1*tmp7 - dtheta2*tmp84 - tau1*tmp7 - tmp59*tmp88 + tmp7*tmp72
    tmp90 = tmp31*tmp89
    # Hessian Elements:
    Hxx[0,0,0] = tmp22
    Hxx[0,0,1] = tmp23
    Hxx[0,0,3] = tmp34
    Hxx[0,1,0] = tmp23
    Hxx[0,1,1] = tmp23
    Hxx[0,1,3] = tmp40
    Hxx[0,2,2] = tmp18*(-tmp46 - tmp49*tmp6)
    Hxx[0,2,3] = tmp56
    Hxx[0,3,0] = tmp34
    Hxx[0,3,1] = tmp40
    Hxx[0,3,2] = tmp56
    Hxx[0,3,3] = tmp18*(M2*R2*tmp7*tmp70 - tmp6*tmp69 - tmp65 - tmp66 - 2*tmp67) \
        - tmp64*(M2*R2*tmp63*tmp7 - tmp57 - tmp6*tmp62 - tmp61) + tmp71*tmp74 + tmp73*tmp77 - tmp74*tmp76
    Hxx[1,0,0] = -tmp21*tmp78
    Hxx[1,0,1] = tmp79
    Hxx[1,0,3] = tmp83
    Hxx[1,1,0] = tmp79
    Hxx[1,1,1] = tmp79
    Hxx[1,1,3] = tmp85
    Hxx[1,2,2] = tmp18*(tmp49*tmp7 + tmp86)
    Hxx[1,2,3] = tmp87
    Hxx[1,3,0] = tmp83
    Hxx[1,3,1] = tmp85
    Hxx[1,3,2] = tmp87
    Hxx[1,3,3] = tmp18*(-F1*dtheta1*tmp3 + tau1*tmp3 - tmp20*tmp62 - tmp3*tmp72 \
        + 2*tmp65 + 2*tmp66 + 4*tmp67 + tmp69*tmp7 - tmp70*tmp88) - tmp64*(-dtheta1*tmp82 \
            + tau1*tmp19 - tmp19*tmp72 + 2*tmp57 + 2*tmp61 + tmp62*tmp7 - tmp63*tmp88) \
                + tmp71*tmp90 - tmp76*tmp90 + tmp77*tmp89
    return Hxx
                
def hessian_x_dot_wrt_u():
    Huu = np.zeros((4, 4, 4))
    return Huu

def hessian_x_dot_wrt_xu(theta2):
    Hxu = np.zeros((4, 4, 4))
    # Common Subexpressions:
    tmp0 = pow(R2, 2)
    tmp1 = M2*tmp0
    tmp2 = I2 + tmp1
    tmp3 = np.cos(theta2)
    tmp4 = np.sin(theta2)
    tmp5 = pow(L1, 2)
    tmp6 = M2*tmp5
    tmp7 = M1*pow(R1, 2)
    tmp8 = pow(M2, 2)*tmp0*tmp5
    tmp9 = I1*I2 + I1*tmp1 + I2*tmp6 + I2*tmp7 + tmp1*tmp7 + pow(tmp4, 2)*tmp8
    tmp10 = 2*tmp3*tmp4*tmp8/pow(tmp9, 2)
    tmp11 = 1.0/tmp9
    tmp12 = L1*M2*R2
    tmp13 = tmp12*tmp3
    tmp14 = L1*M2*R2*tmp11*tmp4 - tmp10*(-tmp13 - tmp2)
    # Mixed Hessian Elements:
    Hxu[0,3,0] = -tmp10*tmp2
    Hxu[0,3,1] = tmp14
    Hxu[1,3,0] = tmp14
    Hxu[1,3,1] = -tmp10*(I1 + 2*tmp13 + tmp2 + tmp6 + tmp7) - 2*tmp11*tmp12*tmp4
    return Hxu
    
def jacobian_x_dot_wrt_x_simplified(x, u):
    return jacobian_x_dot_wrt_x(x[0], x[1], x[2], x[3], u[0])

def jacobian_x_dot_wrt_u_simplified(x):
    return jacobian_x_dot_wrt_u(x[3])

def hessian_x_dot_wrt_x_simplified(x, u):
    return hessian_x_dot_wrt_x(x[0], x[1], x[2], x[3], u[0])

def hessian_x_dot_wrt_u_simplified():
    return hessian_x_dot_wrt_u()

def hessian_x_dot_wrt_xu_simplified(x):
    return hessian_x_dot_wrt_xu(x[3])
