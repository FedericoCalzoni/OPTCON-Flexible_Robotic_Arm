#
# Gradient method for Optimal Control
# Discrete-time nonlinear dynamics
# Lorenzo Sforni
# Bologna, 22/11/2022
#

import numpy as np
import sympy as sp

dim_x = 2 # state dimension
dim_u = 1 # input dimension

dt = 1e-3 # discretization stepsize - Forward Euler

# Dynamics parameters

mm = 0.4 # mass
ll = 0.2 # length
kk = 1 # friction
gg = 9.81 # gravity
KKeq = mm*gg*ll # K for equilibrium


def dynamics(xx,uu):
  """
    Nonlinear dynamics of a pendulum

    Args
      - xx \in \R^2 state at time t
      - uu \in \R^1 input at time t

    Return 
      - next state xx_{t+1}
      - gradient of f wrt x, at xx,uu
      - gradient of f wrt u, at xx,uu
  
  """
  xx = xx[:,None] # reshape to column vector
  uu = uu[:,None]
  
  print("initial xx", xx)
  print("initial uu", uu)

  xxp = np.zeros((dim_x,1)) # next state
  print("xxp:", xxp)
  
  # Dynamics (continuous)
  # x1_dot = x2
  # x2_dot = - g/l * sin(x1) - b*x2 + 1/M * u(t)
  
  # Dynamics (discrete)
  # x1_{t+1} = x1_t + dt * x2_t
  # x2_{t+1} = x2_t + dt * (- g/l * sin(x1_t) - b*x2_t + 1/M * u(t))

  xxp[0] = xx[0,0] + dt * xx[1,0]
  xxp[1] = xx[1,0] + dt * (- gg / ll * np.sin(xx[0,0]) - kk / (mm * ll) * xx[1,0] + 1 / (mm * (ll ** 2)) * uu[0,0])

  # Gradient

  dfx = np.zeros((dim_x, dim_x))
  dfu = np.zeros((dim_u, dim_x))

  #df1
  dfx[0,0] = 1
  dfx[1,0] = dt

  dfu[0,0] = 0

  #df2

  dfx[0,1] = dt*-gg / ll * np.cos(xx[0,0])
  dfx[1,1] = 1 + dt*(- kk / (mm * ll))

  dfu[0,1] = dt / (mm * (ll ** 2))

  xxp = xxp.squeeze()

  return xxp, dfx, dfu

def main():
  # Initial state and input
  xx = np.array([np.pi / 4, 0])  # initial state [theta, theta_dot]
  uu = np.array([0])  # initial input
  
  print("Initial state:", xx)
  print("Initial input:", uu)

  # Call the dynamics function
  xxp, dfx, dfu = dynamics(xx, uu)

  # Print the results
  print("Next state:", xxp)
  print("Gradient wrt state:", dfx)
  print("Gradient wrt input:", dfu)

if __name__ == "__main__":
  main()


