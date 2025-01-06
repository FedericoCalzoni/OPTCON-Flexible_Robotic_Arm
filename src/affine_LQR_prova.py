import numpy as np
import parameters as param
import dynamics as dyn
import cost
import matplotlib.pyplot as plt

def Affine_problem(x_reference, u_reference):
    max_iter = 10000
    size = x_reference.shape[0]
    TT = x_reference.shape[1]
    l = np.array((max_iter))
    x_optimal = np.zeros((size, TT))
    u_optimal = np.zeros((size, TT))
    