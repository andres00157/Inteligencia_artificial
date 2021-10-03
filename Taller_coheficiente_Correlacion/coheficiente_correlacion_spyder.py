import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


x = np.arange(1,6.5,0.5)

x = np.array([np.ones_like(x),x])

y = np.array([0.169610271922408, 0.283395812542308, 0.386358737510785, 0.470227872390909, 0.433281293764675, 0.600267648212653, 0.738338980436742, 0.790315020494445, 0.877464268422459, 0.84356446225183, 0.96443891694455])
def costo(parametos,x,y):
    J = (sum((np.transpose(parametos) @ x-y)**2))/(2*x.shape[1])
    return J

def gradiente(teta,x,y):
    grad = ((np.transpose(teta) @ x-y) @ np.transpose(x))/x.shape[1]
    return grad

def calculo_recta(x,y,theta_inicial,alpha,error_min,max_iteracion):
    parametros = theta_inicial
    for i in range(max_iteracion):
        parametros = parametros- alpha * gradiente(parametros,x,y)
        J = costo(parametros,x,y)
        if(J<error_min):
            break
    return parametros, J

def calc_Recta(tetha,x):
    y_est = np.transpose(x) @ tetha
    return y_est

def R_cuadrado(x,y,tetha):
    y_est = calc_Recta(tetha,x)
    y_prom = sum(y)/y.shape[0]
    R_cua= 1-(sum((y-y_est)**2))/(sum((y-y_prom)**2))
    return R_cua
    


parametros_min, J_min = calculo_recta(x,y,np.array([1,1]),0.032,0.000000001,10000)
r = R_cuadrado(x,y,parametros_min)
print("Parametros hallados:",parametros_min)
print("Coheficiente de correlacion:",r)


