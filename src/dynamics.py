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

def dynamics(x, u):

    dtheta1 = x[0].item()
    dtheta2 = x[1].item()
    theta1 = x[2].item()
    theta2 = x[3].item()

    tau1 = u[0].item()
    
    xx = np.array([dtheta1, dtheta2, theta1, theta2])
    xx = xx[:,None]
    uu = np.array([[tau1],
                   [0],
                   [0],
                   [0]])

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
    ct = - M_inv_ext @ (C_ext + G_ext)

    x_dot = A @ xx + B @ uu + ct
    
    x_new = (xx + dt * x_dot).flatten()
        
    return x_new

def jacobian_gravity(theta1, theta2):
    JG_11 = G*M2*R2*np.cos(theta1 + theta2) + G*(L1*M2 + M1*R1)*np.cos(theta1)
    JG_12 = G*M2*R2*np.cos(theta1 + theta2)
    JG_21 = JG_12
    JG_22 = JG_12
    JG_13 = -1
    JG_23 = 0
    return np.array([[JG_11 , JG_12, JG_13], [JG_21 , JG_22, JG_23]])

def jacobian_x_new_wrt_x(x, u):
    dtheta1 = x[0].item()
    dtheta2 = x[1].item()
    theta1 = x[2].item()
    theta2 = x[3].item()
    tau1 = u[0].item()

    # Common Subexpressions:
    tmp0 = pow(R2, 2)
    tmp1 = M2*tmp0
    tmp2 = I2 + tmp1
    tmp3 = F1*tmp2
    tmp4 = np.sin(theta2)
    tmp5 = L1*M2
    tmp6 = R2*tmp5
    tmp7 = tmp4*tmp6
    tmp8 = dtheta2*tmp7
    tmp9 = 2*tmp8
    tmp10 = np.cos(theta2)
    tmp11 = tmp10*tmp6
    tmp12 = tmp11 + tmp2
    tmp13 = 2*dtheta1
    tmp14 = tmp13*tmp7
    tmp15 = pow(L1, 2)
    tmp16 = M2*tmp15
    tmp17 = M1*pow(R1, 2)
    tmp18 = pow(M2, 2)*tmp0
    tmp19 = tmp15*tmp18
    tmp20 = I1*I2 + I1*tmp1 + I2*tmp16 + I2*tmp17 + tmp1*tmp17 + tmp19*pow(tmp4, 2)
    tmp21 = 1.0/tmp20
    tmp22 = dt*tmp21
    tmp23 = F2*tmp12
    tmp24 = dtheta2 + tmp13
    tmp25 = -tmp24*tmp7 - tmp8
    tmp26 = theta1 + theta2
    tmp27 = G*np.cos(tmp26)
    tmp28 = M2*R2
    tmp29 = tmp27*tmp28
    tmp30 = G*(M1*R1 + tmp5)
    tmp31 = tmp29 + tmp30*np.cos(theta1)
    tmp32 = F2*tmp8
    tmp33 = G*np.sin(tmp26)
    tmp34 = L1*pow(dtheta1, 2)
    tmp35 = tmp33 + tmp34*tmp4
    tmp36 = L1*tmp18*tmp35*tmp4
    tmp37 = -dtheta2*tmp11*tmp24 + tmp29
    tmp38 = tmp10*tmp34 + tmp27
    tmp39 = -tmp24*tmp8 + tmp28*tmp33 + tmp30*np.sin(theta1)
    tmp40 = 2*dt*tmp10*tmp19*tmp4/pow(tmp20, 2)
    tmp41 = I1 + 2*tmp11 + tmp16 + tmp17 + tmp2
    tmp42 = F2*tmp41
    tmp43 = tmp28*tmp41
    # Jacobian Elements:
    dfx = np.zeros((4, 4))
    dfx[0,0] = tmp22*(tmp12*tmp14 + tmp2*tmp9 - tmp3) + 1
    dfx[0,1] = tmp22*(-tmp2*tmp25 + tmp23)
    dfx[0,2] = tmp22*(tmp12*tmp29 - tmp2*tmp31)
    dfx[0,3] = dt*tmp21*(M2*R2*tmp12*tmp38 - tmp2*tmp37 - tmp32 - tmp36) - tmp40*(-dtheta1*tmp3 + dtheta2*tmp23 + tau1*tmp2 + tmp12*tmp28*tmp35 - tmp2*tmp39)
    dfx[1,0] = tmp22*(F1*tmp12 - tmp12*tmp9 - tmp14*tmp41)
    dfx[1,1] = tmp22*(tmp12*tmp25 - tmp42) + 1
    dfx[1,2] = tmp22*(tmp12*tmp31 - tmp29*tmp41)
    dfx[1,3] = dt*tmp21*(-F1*dtheta1*tmp7 + tau1*tmp7 + tmp12*tmp37 + 2*tmp32 + 2*tmp36 - tmp38*tmp43 - tmp39*tmp7) - tmp40*(F1*dtheta1*tmp12 - dtheta2*tmp42 - tau1*tmp12 + tmp12*tmp39 - tmp35*tmp43)
    dfx[2,0] = dt
    dfx[2,2] = 1
    dfx[3,1] = dt
    dfx[3,3] = 1
    return dfx

def jacobian_x_new_wrt_u(x):
    theta2 = x[3].item()
    # Common Subexpressions:
    tmp0 = pow(R2, 2)
    tmp1 = M2*tmp0
    tmp2 = I2 + tmp1
    tmp3 = pow(L1, 2)
    tmp4 = M1*pow(R1, 2)
    tmp5 = dt/(I1*I2 + I1*tmp1 + I2*M2*tmp3 + I2*tmp4 + pow(M2, 2)*tmp0*tmp3*pow(np.sin(theta2), 2) + tmp1*tmp4)
    # Jacobian Elements:
    dfu = np.zeros((4, 1))
    dfu[0,0] = tmp2*tmp5
    dfu[1,0] = tmp5*(-L1*M2*R2*np.cos(theta2) - tmp2)
    dfu[2,0] = 0
    dfu[3,0] = 0

    return dfu



def hessian_x_new_wrt_x(x, u):
    Hxx = np.zeros((4, 4, 4))
    dtheta1 = x[0].item()
    dtheta2 = x[1].item()
    theta1 = x[2].item()
    theta2 = x[3].item()
    tau1 = u[0].item()
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
    tmp19 = dt*tmp18
    tmp20 = tmp11*tmp2
    tmp21 = 2*tmp20
    tmp22 = tmp19*tmp21
    tmp23 = tmp22*tmp7
    tmp24 = tmp22*tmp6
    tmp25 = dtheta2*tmp3
    tmp26 = 2*tmp6
    tmp27 = 2*dtheta1
    tmp28 = tmp27*tmp7
    tmp29 = F1*tmp6
    tmp30 = dtheta2*tmp20
    tmp31 = 2*tmp15
    tmp32 = dt/pow(tmp17, 2)
    tmp33 = tmp0*tmp11*tmp32
    tmp34 = tmp31*tmp33
    tmp35 = dt*tmp18*(-tmp16*tmp27 + tmp25*tmp26 + tmp28*tmp3) - tmp34*(tmp20*tmp28 + tmp26*tmp30 - tmp29)
    tmp36 = F2*tmp20
    tmp37 = dtheta2 + tmp27
    tmp38 = -tmp25 - tmp3*tmp37
    tmp39 = F2*tmp7
    tmp40 = -tmp20*tmp37 - tmp30
    tmp41 = dt*tmp18*(-tmp36 - tmp38*tmp6) - tmp34*(tmp39 - tmp40*tmp6)
    tmp42 = theta1 + theta2
    tmp43 = np.sin(tmp42)
    tmp44 = G*tmp43
    tmp45 = M2*R2
    tmp46 = tmp44*tmp45
    tmp47 = tmp46*tmp7
    tmp48 = G*(M1*R1 + tmp1)
    tmp49 = tmp46 + tmp48*np.sin(theta1)
    tmp50 = -tmp49
    tmp51 = G*np.cos(tmp42)
    tmp52 = L1*tmp14
    tmp53 = tmp11*tmp52
    tmp54 = tmp51*tmp53
    tmp55 = tmp45*tmp51
    tmp56 = tmp48*np.cos(theta1) + tmp55
    tmp57 = dt*tmp18*(G*M2*R2*tmp43*tmp6 - tmp47 - tmp54) - tmp34*(tmp55*tmp7 - tmp56*tmp6)
    tmp58 = dtheta2*tmp36
    tmp59 = L1*pow(dtheta1, 2)
    tmp60 = tmp11*tmp59 + tmp44
    tmp61 = tmp52*tmp60
    tmp62 = tmp11*tmp61
    tmp63 = -tmp25*tmp37 + tmp55
    tmp64 = tmp0*tmp59 + tmp51
    tmp65 = 4*tmp15*tmp33
    tmp66 = F2*tmp25
    tmp67 = tmp0*tmp61
    tmp68 = tmp53*tmp64
    tmp69 = -tmp30*tmp37
    tmp70 = -tmp46 - tmp69
    tmp71 = -tmp60
    tmp72 = 2*tmp16
    tmp73 = tmp49 + tmp69
    tmp74 = -dtheta1*tmp29 + dtheta2*tmp39 + tau1*tmp6 + tmp45*tmp60*tmp7 - tmp6*tmp73
    tmp75 = tmp32*tmp74
    tmp76 = pow(tmp0, 2)
    tmp77 = tmp31*tmp76
    tmp78 = 8*pow(L1, 4)*pow(M2, 4)*pow(R2, 4)*dt*tmp12*tmp76/pow(tmp17, 3)
    tmp79 = I1 + tmp10 + 2*tmp3 + tmp6 + tmp9
    tmp80 = -tmp23
    tmp81 = 2*tmp7
    tmp82 = tmp27*tmp79
    tmp83 = F1*tmp20
    tmp84 = dt*tmp18*(4*dtheta1*tmp12*tmp13*tmp4*tmp8 + 2*dtheta2*tmp12*tmp13*tmp4*tmp8 - tmp25*tmp81 - tmp3*tmp82 - tmp83) - tmp34*(F1*tmp7 - tmp20*tmp82 - tmp30*tmp81)
    tmp85 = F2*tmp79
    tmp86 = dt*tmp18*(-tmp20*tmp40 + 2*tmp36 + tmp38*tmp7) - tmp34*(tmp40*tmp7 - tmp85)
    tmp87 = tmp46*tmp79
    tmp88 = dt*tmp18*(-tmp20*tmp56 - tmp47 + 2*tmp54 + tmp87) - tmp34*(-tmp55*tmp79 + tmp56*tmp7)
    tmp89 = tmp45*tmp79
    tmp90 = F1*dtheta1*tmp7 - dtheta2*tmp85 - tau1*tmp7 - tmp60*tmp89 + tmp7*tmp73
    tmp91 = tmp32*tmp90
    # Hessian Elements:
    # Hessian 0:
    Hxx[0,0,0] = tmp23
    Hxx[0,0,1] = tmp24
    Hxx[0,0,2] = 0
    Hxx[0,0,3] = tmp35
    Hxx[0,1,0] = tmp24
    Hxx[0,1,1] = tmp24
    Hxx[0,1,2] = 0
    Hxx[0,1,3] = tmp41
    Hxx[0,2,0] = 0
    Hxx[0,2,1] = 0
    Hxx[0,2,2] = tmp19*(-tmp47 - tmp50*tmp6)
    Hxx[0,2,3] = tmp57
    Hxx[0,3,0] = tmp35
    Hxx[0,3,1] = tmp41
    Hxx[0,3,2] = tmp57
    Hxx[0,3,3] = tmp19*(M2*R2*tmp7*tmp71 - tmp6*tmp70 - tmp66 - tmp67 - 2*tmp68) - tmp65*(M2*R2*tmp64*tmp7 - tmp58 - tmp6*tmp63 - tmp62) + tmp72*tmp75 + tmp74*tmp78 - tmp75*tmp77
    # Hessian 1:
    Hxx[1,0,0] = -tmp22*tmp79
    Hxx[1,0,1] = tmp80
    Hxx[1,0,2] = 0
    Hxx[1,0,3] = tmp84
    Hxx[1,1,0] = tmp80
    Hxx[1,1,1] = tmp80
    Hxx[1,1,2] = 0
    Hxx[1,1,3] = tmp86
    Hxx[1,2,0] = 0
    Hxx[1,2,1] = 0
    Hxx[1,2,2] = tmp19*(tmp50*tmp7 + tmp87)
    Hxx[1,2,3] = tmp88
    Hxx[1,3,0] = tmp84
    Hxx[1,3,1] = tmp86
    Hxx[1,3,2] = tmp88
    Hxx[1,3,3] = tmp19*(-F1*dtheta1*tmp3 + tau1*tmp3 - tmp21*tmp63 - tmp3*tmp73 + 2*tmp66 + 2*tmp67 + 4*tmp68 + tmp7*tmp70 - tmp71*tmp89) - tmp65*(-dtheta1*tmp83 + tau1*tmp20 - tmp20*tmp73 + 2*tmp58 + 2*tmp62 + tmp63*tmp7 - tmp64*tmp89) + tmp72*tmp91 - tmp77*tmp91 + tmp78*tmp90
    # Hessian 2:
    Hxx[2,0,0] = 0
    Hxx[2,0,1] = 0
    Hxx[2,0,2] = 0
    Hxx[2,0,3] = 0
    Hxx[2,1,0] = 0
    Hxx[2,1,1] = 0
    Hxx[2,1,2] = 0
    Hxx[2,1,3] = 0
    Hxx[2,2,0] = 0
    Hxx[2,2,1] = 0
    Hxx[2,2,2] = 0
    Hxx[2,2,3] = 0
    Hxx[2,3,0] = 0
    Hxx[2,3,1] = 0
    Hxx[2,3,2] = 0
    Hxx[2,3,3] = 0
    # Hessian 3:
    Hxx[3,0,0] = 0
    Hxx[3,0,1] = 0
    Hxx[3,0,2] = 0
    Hxx[3,0,3] = 0
    Hxx[3,1,0] = 0
    Hxx[3,1,1] = 0
    Hxx[3,1,2] = 0
    Hxx[3,1,3] = 0
    Hxx[3,2,0] = 0
    Hxx[3,2,1] = 0
    Hxx[3,2,2] = 0
    Hxx[3,2,3] = 0
    Hxx[3,3,0] = 0
    Hxx[3,3,1] = 0
    Hxx[3,3,2] = 0
    Hxx[3,3,3] = 0
    return Hxx
                
def hessian_x_new_wrt_u():
    Huu = np.zeros((4, 1))
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
