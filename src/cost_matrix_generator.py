import numpy as np
import parameters as pm

def matrices_elaburator():
    x_size = pm.Qt.shape[0]
    u_size = pm.Rt.shape[0]
    Qt_elab = np.zeros((x_size,x_size))
    QT_elab = np.zeros((x_size,x_size))
    Rt_elab = np.zeros((u_size))

    

