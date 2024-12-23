import numpy as np
import dynamics as dyn


def solve_ltv_LQR(A0, B0, Q0, R0, S0, QT, T, x0, q0, r0, qT):
	
    try:
        # check if matrix is (.. x .. x T) - 3 dimensional array 
        x_size, lA = A0.shape[1:]
    except ValueError:
        # if not 3 dimensional array, make it (.. x .. x 1)
        A0 = A0[:,:,None]
        x_size, lA = A0.shape[1:]

    try:  
        u_size, lB = B0.shape[1:]
    except ValueError:
        B0 = B0[:,:,None]
        u_size, lB = B0.shape[1:]

    try:
        nQ, lQ = Q0.shape[1:]
    except ValueError:
        Q0 = Q0[:,:,None]
        nQ, lQ = Q0.shape[1:]

    try:
        nR, lR = R0.shape[1:]
    except ValueError:
        R0 = R0[:,:,None]
        nR, lR = R0.shape[1:]

    try:
        nSi, nSs, lS = S0.shape
    except ValueError:
        S0 = S0[:,:,None]
        nSi, nSs, lS = S0.shape

    # Check dimensions consistency -- safety
    if nQ != x_size:
        print("Matrix Q does not match number of states")
        exit()
    if nR != u_size:
        print("Matrix R does not match number of inputs")
        exit()
    if nSs != x_size:
        print("Matrix S does not match number of states")
        exit()
    if nSi != u_size:
        print("Matrix S does not match number of inputs")
        exit()


    if lA < T:
        A0 = A0.repeat(T, axis=2)
    if lB < T:
        B0 = B0.repeat(T, axis=2)
    if lQ < T:
        Q0 = Q0.repeat(T, axis=2)
    if lR < T:
        R0 = R0.repeat(T, axis=2)
    if lS < T:
        S0 = S0.repeat(T, axis=2)


    # Check for affine terms

    if q0 is not None or r0 is not None or qT is not None:
        print("Augmented term!")

    K_star = np.zeros((u_size, x_size, T))
    sigma_star = np.zeros((u_size, T))
    P = np.zeros((x_size, x_size, T))
    p = np.zeros((x_size, T))

    Q = Q0
    R = R0
    S = S0
    
    q = q0
    r = r0

    A = A0
    B = B0

    delta_x_star = np.zeros((x_size, T))
    delta_u_star = np.zeros((u_size, T))

    delta_x_star[:,0] = x0.flatten()
    
    P[:,:,-1] = QT
    p[:,-1] = qT
  
    # Solve Riccati equation
    for t in reversed(range(T-1)):
        Qt = Q[:,:,t]
        qt = q[:,t][:,None]
        Rt = R[:,:,t]
        rt = r[:,t][:,None]
        At = A[:,:,t]
        Bt = B[:,:,t]
        St = S[:,:,t]
        Pt_plus_1 = P[:,:,t+1]
        pt_plus_1 = p[:, t+1][:,None]

        Mt_inv = np.linalg.inv(Rt + Bt.T @ Pt_plus_1 @ Bt)
        mt = rt + Bt.T @ pt_plus_1
        
        Pt = At.T @ Pt_plus_1 @ At - (Bt.T @ Pt_plus_1 @ At + St).T @ Mt_inv @ (Bt.T @ Pt_plus_1 @ At + St) + Qt
        pt = At.T @ pt_plus_1 - (Bt.T @ Pt_plus_1 @ At + St).T @ Mt_inv @ mt + qt

        P[:,:,t] = Pt
        p[:,t] = pt.squeeze()


    # Evaluate KK
    
    for t in range(T-1):
        Qt = Q[:,:,t]
        qt = q[:,t][:,None]
        Rt = R[:,:,t]
        rt = r[:,t][:,None]
        At = A[:,:,t]
        Bt = B[:,:,t]
        St = S[:,:,t]

        Pt_plus_1 = P[:,:,t+1]
        pt_plus_1 = p[:,t+1][:,None]

        # Check positive definiteness

        Mt_inv = np.linalg.inv(Rt + Bt.T @ Pt_plus_1 @ Bt)
        mt = rt + Bt.T @ pt_plus_1

        # TODO: add a regularization step here

        K_star[:,:,t] = -Mt_inv @ (Bt.T @ Pt_plus_1 @ At + St)
        sigma_t = -Mt_inv @ mt

        sigma_star[:,t] = sigma_t.squeeze()

    for t in range(T-1):
        # Trajectory

        delta_u_star[:, t] = K_star[:,:,t] @ delta_x_star[:, t] + sigma_star[:,t]
        delta_x_star[:,t+1] = A[:,:,t] @ delta_x_star[:,t] + B[:,:,t] @ delta_u_star[:, t]

    return K_star, sigma_star, delta_x_star


def compute_LQR_trajectory(A0, B0, Q0, R0, S0, QT, T, x0, q0, r0, qT, u0, step_size = 0.1, max_iter = 100):
    
    u = np.zeros((u0.shape[0], T, max_iter))
    x = np.zeros((x0.shape[0], T, max_iter))
    
    print(f'Size u: {u.shape}, Size x: {x.shape}')
    
    u[:,:,0] = u0
    x[:,:,1] = x0
        
    for k in range(max_iter-1):
        
        K, sigma, delta_x = solve_ltv_LQR(A0, B0, Q0, R0, S0, QT, T, x0, q0, r0, qT)        
        
        print(f'Size K: {K.shape}, Size sigma: {sigma.shape}, Size delta_x: {delta_x.shape}')
        
        for t in range(T-1):
            u[:,t,k+1] = u[:,t,k] + step_size * (sigma[:,t] + K[:,:,t] @ delta_x[:,t]) + K[:,:,t] @ (x[:, t, k+1]-x[:,t,k]- step_size * delta_x[:,t])
            x[:,t+1,k+1] = dyn.dynamics(x[:,t,k+1][:, None], u[:,t,k+1][:, None])[0].flatten()
            
    return x, u

# Test
A0 = np.array([[0.9, 0.1, 0.0, 0.0],
               [0.0, 0.9, 0.1, 0.0],
               [0.0, 0.0, 0.9, 0.1],
               [0.1, 0.0, 0.0, 0.9]])
B0 = np.array([[0.1, 0.0, 0.0, 0.0],
               [0.0, 0.1, 0.0, 0.0],
               [0.0, 0.0, 0.1, 0.0],
               [0.0, 0.0, 0.0, 0.1]])
Q0 = np.eye(4)
R0 = np.eye(4)
S0 = np.zeros((4, 4))
QT = np.eye(4) * 10
T = 100
x0 = np.array([1, 0, 0, 0])
u0 = np.zeros((4, 1))

q0 = np.zeros((4, T))
r0 = np.zeros((4, T))
qT = np.zeros(4)

x, u = compute_LQR_trajectory(A0, B0, Q0, R0, S0, QT, T, x0[:,None], q0, r0, qT, u0)
print(x)
print(u)