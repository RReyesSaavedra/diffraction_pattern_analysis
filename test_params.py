
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, j1

fixed_params = [-0.6,0.000013,0.000018,0.439]
f = 0.09
x_array = np.array(range(-225,225))

#this function leaves all constants fixed, except for frequency.
def fixed_bessel(x_array, f):
    global fixed_params
    a,b,c,d = fixed_params
    value = a * np.exp(b * x_array**2)
    return value

def airy_pattern(r, I0, beta):
    # Evitar división por cero
    with np.errstate(divide='ignore', invalid='ignore'):
        intensity = (2 * j1(beta * r) / (beta * r))**2
        intensity[r == 0] = 1  # Definir el centro como intensidad máxima
    return I0 * intensity

I0_fit = 255
beta_fit = 0.1

I_fit = airy_pattern(x_array, I0_fit, beta_fit)

plt.plot(x_array, I_fit, linewidth=1)
plt.xlabel('eje x en pixels')
plt.ylabel('Valor de intensidad normalizado')
plt.title('Airy')
plt.show()  

y_array = fixed_bessel(x_array,f)

plt.plot(x_array, y_array, linewidth=1)
plt.xlabel('eje x en pixels')
plt.ylabel('Valor de intensidad normalizado')
plt.title('simple bessel - +')
plt.show()  

fixed_params = [-0.6,-0.000013,0.000018,0.439]
y_array = fixed_bessel(x_array,f)

plt.plot(x_array, y_array, linewidth=1)
plt.xlabel('eje x en pixels')
plt.ylabel('Valor de intensidad normalizado')
plt.title('simple bessel - -')
plt.show()  

fixed_params = [0.6,0.000013,0.000018,0.439]
y_array = fixed_bessel(x_array,f)

plt.plot(x_array, y_array, linewidth=1)
plt.xlabel('eje x en pixels')
plt.ylabel('Valor de intensidad normalizado')
plt.title('simple bessel + +')
plt.show()  

fixed_params = [0.6,-0.000013,0.000018,0.439]
y_array = fixed_bessel(x_array,f)

plt.plot(x_array, y_array, linewidth=1)
plt.xlabel('eje x en pixels')
plt.ylabel('Valor de intensidad normalizado')
plt.title('simple bessel + -')
plt.show()  

