import cv2
import numpy as np
import matplotlib.pyplot as plt
import config    
from scipy.optimize import curve_fit
from scipy.special import jv, j1


# Define the modified Bessel function to fit
# def modified_cosine_function(x, a, b, c, d, e):
#     return a*np.exp(b*x**2)* np.cos(c * x) - d*abs(x**2) + e
    
# Define the modified Bessel function to fit
# Define a 10th-degree polynomial function to fit
def polynomial_function(x, *coeffs):
    return sum(c * x**(2*i) for i, c in enumerate(coeffs))


## Function to fit the polynomial with stopping criteria
def fit_polynomial(params, x_array, y_array, plot_label):
    max_iter = 5

    for i in range(max_iter):
        popt, _ = curve_fit(polynomial_function, x_array, y_array, p0=params)
    
    print(f'NEW POLYNOMIAL COEFFICIENTS: {popt}')
    
    # Plot the original data and the fitted polynomial function
    plt.plot(x_array, y_array, label='Original Data', linewidth=1)
    plt.plot(x_array, polynomial_function(x_array, *popt), label='Fitted Polynomial', linewidth=2)
    plt.xlabel('eje x en pixels')
    plt.ylabel('Valor de intensidad normalizado')
    plt.title(plot_label)
    plt.legend()
    plt.show()
    return popt

# Define the modified Bessel function to fit
def modified_bessel(x, a, f, c):
    return (a*jv(0, f * x) + c)**2

#this function leaves all constants fixed, except for frequency.
def fixed_bessel(x_array, f):
    global fixed_params
    a,b,c,d = fixed_params
    value = a * np.exp(b * x_array**2) + jv(0, f * x_array) - c * np.abs(x_array**2) + d
    return value

# Function to fit the Bessel function with stopping criteria
def fit_bessel(params, x_array, y_array, plot_label):
    popt, _ = curve_fit(modified_bessel, x_array, y_array, p0=params)
    print(f'NEW PARAMS: {popt}')
    new_params = popt
    #print(f'FIXED PARAMS: {new_params} \t FRECUENCY: {f_temp}')
    # Plot the original data and the fitted Bessel function
    plt.plot(x_array, y_array, label='Original Data', linewidth=1)
    plt.plot(x_array, modified_bessel(x_array, *popt), label='Fitted Bessel Function', linewidth=2)
    plt.xlabel('eje x en pixels')
    plt.ylabel('Valor de intensidad normalizado')
    plt.title(plot_label)
    plt.show()    
    return new_params

def fit_fixed_bessel(f0, x_array, y_array, plot_label):
    popt, pcov = curve_fit(fixed_bessel, x_array, y_array, p0=[f0])
    f0 = popt[0] # Update param with the optimized one
    # Plot the original data and the fitted Bessel function
    plt.plot(x_array, y_array, label='Original Data', linewidth=1)
    plt.plot(x_array, fixed_bessel(x_array, f0), label=f'Fitted Bessel Function. {plot_label}', linewidth=2)
    plt.xlabel('eje x en pixels')
    plt.ylabel('Valor de intensidad normalizado')
    plt.title(plot_label)
    plt.legend()
    plt.show()
    return f0

def process_array(image):
    img_array = image.img_array
    centroid = image.centroid
    #  START: the following code is to align all images peaks
    x_size = np.shape(img_array)[0]
    x_range = int(x_size*0.9)
    xi = centroid[0] - int(x_range/2)
    xf = centroid[0] + int(x_range/2)
    print('CENTROIDE', centroid, 'X: ', centroid[0], 'xrange: ', xi, ' - ', xf)
    y_array = img_array[centroid[1],:]
    y_array = y_array[xi:xf]
    x_array = np.array(range(-int(x_range/2),int(x_range/2)))
    y_min = min(y_array)
    y_array = (y_array - y_min)
    y_max = max(y_array)
    y_array = y_array /y_max
    # y_array = (y_array[xi:centroid[0]] - y_min)/y_max
    # print('NEW DIMENSIONS',np.shape(y_array))
    # y_array = np.concatenate((y_array, y_array[::-1]))
    print('NEW DIMENSIONS',np.shape(y_array))
    new_array = []
    n = np.size(y_array)
    d = config.avg_delta
    for i in range(n):
        if i >= d and i < n - d:
            yf = np.mean(y_array[i - d:i + d + 1])
        else:
            yf = y_array[i]
        new_array.append(yf)
    return new_array, x_range

def airy_pattern(r, I0, beta):
    # Evitar división por cero
    with np.errstate(divide='ignore', invalid='ignore'):
        intensity = (2 * j1(beta * r) / (beta * r))**2
        intensity[r == 0] = 1  # Definir el centro como intensidad máxima
    return I0 * intensity

def fit_airy(r,I_exp,label):
    # 🎯 Ajuste de la función Airy
    params, covariance = curve_fit(airy_pattern, r, I_exp, p0=[1, 2])
    I0_fit, beta_fit = params

    # 📊 Generar la curva ajustada
    I_fit = airy_pattern(r, I0_fit, beta_fit)

    # 📈 GRAFICAR RESULTADOS
    plt.plot(r, I_exp, 'o', label='Datos experimentales', markersize=4)
    plt.plot(r, I_fit, '-', label='Ajuste Airy', color='red')
    plt.xlabel('Radio r')
    plt.ylabel('Intensidad')
    plt.title(f'Ajuste del Patrón de Airy {label}')
    plt.legend()
    plt.grid()
    plt.show()

    # 📢 Imprimir parámetros ajustados
    print(f"I0 ajustado: {I0_fit:.4f}")
    print(f"Beta ajustado: {beta_fit:.4f}")

    return beta_fit, I0_fit


def fit_bessel_square(r,I_exp,label):
    # 🎯 Ajuste de la función Airy
    params, covariance = curve_fit(modified_bessel, r, I_exp, p0=[1, 2])
    I0_fit, beta_fit = params

    # 📊 Generar la curva ajustada
    I_fit = modified_bessel(r, I0_fit, beta_fit)

    # 📈 GRAFICAR RESULTADOS
    plt.plot(r, I_exp, 'o', label='Datos experimentales', markersize=4)
    plt.plot(r, I_fit, '-', label='Ajuste Airy', color='red')
    plt.xlabel('Radio r')
    plt.ylabel('Intensidad')
    plt.title(f'Ajuste del Patrón de Airy {label}')
    plt.legend()
    plt.grid()
    plt.show()

    # 📢 Imprimir parámetros ajustados
    print(f"I0 ajustado: {I0_fit:.4f}")
    print(f"Beta ajustado: {beta_fit:.4f}")

    return beta_fit, I0_fit

def bessel_analysis(processed_images,labels):
    y_arrays = []
    num_images = 0
    for img in processed_images:
        y_array, x_range = process_array(img)
        y_arrays.append(y_array) #append resulting array to list of arrays
        num_images += 1

    x_array = np.array(range(-int(x_range/2),int(x_range/2)))
    #Estimating inital params
    # a*np.exp(b*x**2)* jv(0, c * x) - d*abs(x**2) + e
    number_of_peaks = 7  # As seen in your plot or image
    frequency_c = number_of_peaks / x_range
    amplitude_a = max(y_arrays[0])
    c = 0.2
    frequencies = []
    global fixed_params
    temp_params = [amplitude_a,frequency_c*4, c]
    for i in range(num_images):
        temp_params = fit_bessel(temp_params,x_array,y_arrays[i],labels[i])
        print(f'TEMP parameters: {temp_params}')
        frequencies.append(temp_params[1])

    plt.plot(range(np.size(frequencies)),frequencies, linewidth=2)
    plt.xlabel('Sample number')
    plt.ylabel('Frequency of adjusted Bessel function')
    plt.title('Change of frequency by sample')
    plt.show()

    error = 0
    diff = 0

    periodos = []
    for i, f in enumerate(frequencies):
        if (i+1)%2==0:
            periodos.append(1/f)

    plt.scatter(config.molaridades, periodos/periodos[0]*100)
    plt.xlabel('Molaridad')
    plt.ylabel('Periodo Espacial')
    plt.title("Molaridad vs Periodicidad")
    plt.grid(True)
    plt.show()
    return error, diff

def airy_analysis(processed_images,labels):
    y_arrays = []
    num_images = 0
    for img in processed_images:
        y_array, x_range = process_array(img)
        y_arrays.append(y_array) #append resulting array to list of arrays
        num_images += 1

    x_array = np.array(range(-int(x_range/2),int(x_range/2)))
    betas = []
    for i in range(num_images):
        Beta, Intenisty = fit_airy(x_array,y_arrays[i],labels[i])
        betas.append(Beta)
    print(f'BETAS: {betas}')

    periodos = []
    for i, f in enumerate(betas):
        if (i+1)%2==0:
            periodos.append(1/f)

    plt.scatter(config.molaridades, periodos/periodos[0]*100)
    plt.xlabel('Molaridad')
    plt.ylabel('Periodo Espacial')
    plt.title("Molaridad vs Periodicidad")
    plt.grid(True)
    plt.show()

    return 0, 0

def polynomial_analysis(processed_images, labels):
    y_arrays = []
    num_images = 0
    for img in processed_images:
        y_array, x_range = process_array(img)
        y_arrays.append(y_array) #append resulting array to list of arrays
        num_images += 1

    x_array = np.array(range(-int(x_range/2),int(x_range/2)))
    
    # Initial parameters for a 10th-degree polynomial (all coefficients set to small values)
    initial_params = []

    for i in range(config.num_terms):
        c = 0.1
        initial_params.append(c**i)

    print(f'Initial polynomial parameters: {initial_params}')

    for i in range(num_images):
        popt = fit_polynomial(initial_params, x_array, y_arrays[i], labels[i])

    return 0, 0
