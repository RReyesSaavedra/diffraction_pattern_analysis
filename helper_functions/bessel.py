import cv2
import numpy as np
import matplotlib.pyplot as plt


def bessel_patern(img, x, y):
    # Convertir las imágenes a escala de grises    
    imagen1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    xi = int(x-x_dimension/2)
    xf = int(x+x_dimension/2)

    red_255 = imagen1[:, :, 0]
    y_temp = red_255[y, xi:xf]/255    
    #bessel = average_image[y, xi:xf]
    
    return  y_temp

y_data_all_frames = y_arrays

    
from scipy.optimize import curve_fit
from scipy.special import jv


# Define the modified Bessel function to fit
def modified_cosine_function(x, a, b, c, d, e):
    return a*np.exp(b*x**2)* np.cos(c * x) - d*abs(x**2) + e
    
    
# Define the modified Bessel function to fit
def modified_bessel_function(x, a, b, c, d, e):
    return a*np.exp(b*x**2)* jv(0, c * x) - d*abs(x**2) + e

# Function to fit the Bessel function with stopping criteria
def fit_bessel_function(params, x_array, y_array, plot_label):
    tol=1e-6
    max_iter=5
    prev_residual = np.inf
    no_improvement_count = 0
    
    # Plot the original data and the fitted Bessel function
    #plt.plot(x_array, modified_bessel_function(x_array, *params), label='Fitted Bessel Function', linewidth=2)
    #plt.xlabel('eje x en pixels')
    #plt.ylabel('Valor de intensidad normalizado')
    #plt.title(f'patrón tipo Bessel Original')
    #plt.show()   

    for i in range(max_iter):
        popt, _ = curve_fit(modified_bessel_function, x_array, y_array, p0=params)
        residual = np.sum((modified_bessel_function(x_array, *popt) - y_array) ** 2)
        prev_residual = residual
        params = popt
        
    # Plot the original data and the fitted Bessel function
    plt.plot(x_array, y_array, label='Original Data', linewidth=1)
    plt.plot(x_array, modified_bessel_function(x_array, *popt), label='Fitted Bessel Function', linewidth=2)
    plt.xlabel('eje x en pixels')
    plt.ylabel('Valor de intensidad normalizado')
    plt.title(plot_label)
    plt.show()    

    return popt


# Convert x_data and y_data_all_frames_all_frames to numpy arrays if they are not already
x_data = np.array(x_data)
y_data_all_frames = np.array(y_data_all_frames)

# Calculating estimated parameters
number_of_peaks = 7  # As seen in your plot
x_range = x_data.max() - x_data.min()
frequency_c = number_of_peaks / x_range
amplitude_a = y_data_all_frames.max()

# You can start with b and d as small values or set them based on experience
b_initial = 1e-5
d_initial = 1e-5 

# New initial parameters based on estimation
initial_params = [amplitude_a, -b_initial, frequency_c*5, d_initial, 0.4]
#initial_params = [amplitude_a, -b_initial, frequency_c*5, d_initial, 0.4]

print(f'Initial parameters: {initial_params}')

adjusted_bessel_function = np.zeros_like(x_data)
for i, x in enumerate(x_data):
    adjusted_bessel_function[i] = modified_bessel_function(x, *initial_params)

plt.plot(x_data, adjusted_bessel_function, linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Original Bessel function')
plt.show()

# Fit the modified Bessel function to the data
n_concentraciones = np.shape(y_data_all_frames)[0]

fixed_params = []

def fixed_bessel_function(x_array,y_array,f):
    global fixed_params
    global a,b,c,d,e
    label='finding fixed parameters'
    if len(fixed_params) == 0:
        fixed_params = fit_bessel_function(initial_params, x_array, y_array, label)
        a,b,c,d,e = fixed_params
    else:
        c=f
    value = a * np.exp(b * x_array**2) * jv(0, c * x_array) - d * np.abs(x_array**2) + e
    
    return value

def fit_bessel_function_simple(f0, x_array, y_array, plot_label):
    max_iter = 5
    # Wrapper function to fit only `e` using curve_fit
    def wrapper_bessel_function(x_array, e):
        return fixed_bessel_function(x_array, y_array, e)

    for i in range(max_iter):
        popt, pcov = curve_fit(wrapper_bessel_function, x_array, y_array, p0=[f0])
        
        f0 = popt[0] # Update param with the optimized one
    
    f_n = f0
    # Plot the original data and the fitted Bessel function
    plt.plot(x_array, y_array, label='Original Data', linewidth=1)
    plt.plot(x_array, fixed_bessel_function(x_array, y_array,f_n), label=f'Fitted Bessel Function. {plot_label}', linewidth=2)
    plt.xlabel('eje x en pixels')
    plt.ylabel('Valor de intensidad normalizado')
    plt.title(plot_label)
    plt.legend()
    plt.show()
    return f_n

def get_frequencies():
    global fixed_params 
    frequencies = []
    f_inicial=1
    for i in range(n_concentraciones):
        index = int(i/2)
        label = f"Molaridad {Molaridad[index]}"
        f = fit_bessel_function_simple(f_inicial, x_data, y_data_all_frames[i], label)
        frequencies.append(f)
    
    return frequencies

frequencies = get_frequencies()

print(f"\nFrequencies: {frequencies}")
diff, error = diff_value(frequencies, 2)

print(f'\nDiferencia porcentual entre concentraciones {diff}')
print(f'\n\nError porcentual entre fotos de mismas concentraciones {error}')

frequencies = np.array(frequencies)  # Ejemplo de frecuencias en Hz
periodos = 1/frequencies

"""
plt.scatter(Molaridad, periodos/periodos[0]*100)
plt.xlabel('Molaridad')
plt.ylabel('Periodo Espacial')
plt.title("Molaridad vs Periodicidad")
plt.grid(True)
plt.show()
"""