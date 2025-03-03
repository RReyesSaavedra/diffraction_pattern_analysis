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

# Define the modified Bessel function to fit
def modified_bessel(x, a, f, c):
    return a*(jv(0, f * x))**2 + c

#this function leaves all constants fixed, except for frequency.
def fixed_bessel(x_array, f):
    global a, c
    value = a * (jv(0, f * x))**2 + c
    return value

def airy_pattern(r, I0, beta):
    # Evitar división por cero
    with np.errstate(divide='ignore', invalid='ignore'):
        intensity = (2 * j1(beta * r) / (beta * r))**2
        intensity[r == 0] = 1  # Definir el centro como intensidad máxima
    return I0 * intensity

## Function to fit the polynomial with stopping criteria
def fit_polynomial(params, x_array, y_array, plot_label):
    popt, _ = curve_fit(polynomial_function, x_array, y_array, p0=params)
    print(f'NEW POLYNOMIAL COEFFICIENTS: {popt}')
    
    # Plot the original data and the fitted polynomial function
    # plt.plot(x_array, y_array, label='Original Data', linewidth=1)
    # plt.plot(x_array, polynomial_function(x_array, *popt), label='Fitted Polynomial', linewidth=2)
    # plt.xlabel('eje x en pixels')
    # plt.ylabel('Valor de intensidad normalizado')
    # plt.title(plot_label)
    # plt.legend()
    # plt.show()
    return popt

# Function to fit the Bessel function with stopping criteria
def fit_bessel(params, x_array, y_array, plot_label):
    new_params, _ = curve_fit(modified_bessel, x_array, y_array, p0=params)
    print(f'NEW PARAMS: {new_params}')
    # Plot the original data and the fitted Bessel function
    # plt.plot(x_array, y_array, label='Original Data', linewidth=1)
    # plt.plot(x_array, modified_bessel(x_array, *popt), label='Fitted Bessel Function', linewidth=2)
    # plt.xlabel('eje x en pixels')
    # plt.ylabel('Valor de intensidad normalizado')
    # plt.title(plot_label)
    # plt.show()    
    return new_params

def fit_fixed_bessel(f0, x_array, y_array, plot_label):
    popt, pcov = curve_fit(fixed_bessel, x_array, y_array, p0=[f0])
    f0 = popt[0] # Update param with the optimized one
    # Plot the original data and the fitted Bessel function
    # plt.plot(x_array, y_array, label='Original Data', linewidth=1)
    # plt.plot(x_array, fixed_bessel(x_array, f0), label=f'Fitted Bessel Function. {plot_label}', linewidth=2)
    # plt.xlabel('eje x en pixels')
    # plt.ylabel('Valor de intensidad normalizado')
    # plt.title(plot_label)
    # plt.legend()
    # plt.show()
    return f0

def fit_airy(params,x_array, y_array,label):
    # Ajuste de la función Airy
    params, covariance = curve_fit(airy_pattern, x_array, y_array, p0=params)
    I0_fit, beta_fit = params

    # Imprimir parámetros ajustados
    print(f"I0 ajustado: {I0_fit:.4f}")
    print(f"Beta ajustado: {beta_fit:.4f}")
    # GRAFICAR RESULTADOS
    # plt.plot(r, I_exp, 'o', label='Datos experimentales', markersize=4)
    # plt.plot(r, airy_pattern(r, I0_fit, beta_fit), '-', label='Ajuste Airy', color='red')
    # plt.xlabel('Radio r')
    # plt.ylabel('Intensidad')
    # plt.title(f'Ajuste del Patrón de Airy {label}')
    # plt.legend()
    # plt.grid()
    # plt.show()
    return I0_fit, beta_fit

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
    y_min = min(y_array)
    y_array = (y_array - y_min)
    y_max = max(y_array)
    #y_array = y_array /y_max
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

def bessel_adjustment(processed_images,labels):
    y_arrays = []
    num_images = 0
    for img in processed_images:
        y_array, x_range = process_array(img)
        y_arrays.append(y_array) #append resulting array to list of arrays
        num_images += 1

    x_array = np.array(range(-int(x_range/2),int(x_range/2)))
    #Estimating inital params
    number_of_peaks = 2*config.visible_rings + 1
    frequency_c = number_of_peaks / x_range
    amplitude_a = max(y_arrays[0])
    c = min(y_array[0])
    amplitudes = []
    betas = []
    temp_params = [amplitude_a,frequency_c*4, c]
    for i in range(num_images):
        temp_params = fit_bessel(temp_params,x_array,y_arrays[i],labels[i])
        amplitud, beta, c = temp_params
        print(f'New C: {c}')
        amplitudes.append(amplitud)
        betas.append(beta)

    plt.plot(range(np.size(betas)),betas, linewidth=2)
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

def adjustments_compair(processed_images,labels):
    y_arrays = []
    num_images = 0
    for img in processed_images:
        y_array, x_range = process_array(img)
        y_arrays.append(y_array) #append resulting array to list of arrays
        num_images += 1

    x_array = np.array(range(-int(x_range/2),int(x_range/2)))
    
    #Estimating polinomial initial params
    poly_params = []
    for i in range(config.num_terms):
        c = 0.1
        poly_params.append(c**i)

    #Estimating bessel initial params
    number_of_peaks = 2*config.visible_rings + 1
    frequency_c = number_of_peaks / x_range
    amplitude_a = max(y_arrays[0])
    c = min(y_arrays[0])
    bessel_params = [amplitude_a,frequency_c*4, c]

    #Estimating Airy initial params
    airy_params = [amplitude_a,0.1]

    for i in range(num_images):
        y_array = y_arrays[i]
        I0_fit, beta_fit = fit_airy(airy_params, x_array, y_array, labels[i])
        poly_params = fit_polynomial(poly_params, x_array, y_array, labels[i])
        bessel_params = fit_bessel(bessel_params, x_array, y_array, labels[i])

        plt.plot(x_array, y_array, 'o', label='Datos experimentales', markersize=2)
        plt.plot(x_array, polynomial_function(x_array, *poly_params), label='Ajuste Polynomial', linewidth=1)
        plt.plot(x_array, modified_bessel(x_array, *bessel_params), label='Ajuste Bessel', linewidth=1)
        plt.plot(x_array, airy_pattern(x_array, I0_fit, beta_fit), label='Ajuste Airy', linewidth=1)
        plt.xlabel('eje x en pixeles')
        plt.ylabel('Intensidad')
        plt.title(f'Compairing different function adjustments {labels[i]}')
        plt.legend()
        plt.grid()
        plt.show()

    return 0, 0

