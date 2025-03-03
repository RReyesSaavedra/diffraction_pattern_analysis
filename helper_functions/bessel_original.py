import cv2
import numpy as np
import matplotlib.pyplot as plt
import config    
from scipy.optimize import curve_fit
from scipy.special import jv


# Define the modified Bessel function to fit
# def modified_cosine_function(x, a, b, c, d, e):
#     return a*np.exp(b*x**2)* np.cos(c * x) - d*abs(x**2) + e
    
# Define the modified Bessel function to fit
def modified_bessel(x, a, b, c, d, e):
    return a*np.exp(b*x**2)* jv(0, c * x) - d*abs(x**2) + e

#this function leaves all constants fixed, except for frequency.
def fixed_bessel(x_array, f):
    global fixed_params
    a,b,c,d = fixed_params
    value = a * np.exp(b * x_array**2) * jv(0, f * x_array) - c * np.abs(x_array**2) + d
    return value

# Function to fit the Bessel function with stopping criteria
def fit_bessel(params, x_array, y_array, plot_label):
    max_iter=5
    # tol=1e-6
    # prev_residual = np.inf
    # no_improvement_count = 0
    # Plot the original data and the fitted Bessel function
    # plt.plot(x_array, modified_bessel_function(x_array, *params), label='Fitted Bessel Function', linewidth=2)
    # plt.xlabel('eje x en pixels')
    # plt.ylabel('Valor de intensidad normalizado')
    # plt.title(f'patrón tipo Bessel Original')
    # plt.show()   

    for i in range(max_iter):
        popt, _ = curve_fit(modified_bessel, x_array, y_array, p0=params)
        residual = np.sum((modified_bessel(x_array, *popt) - y_array) ** 2)
        prev_residual = residual
        
    print(f'NEW PARAMS: {popt}')
    a,b,f_temp,c,d = popt
    new_params = [a,b,c,d]
    print(f'FIXED PARAMS: {new_params} \t FRECUENCY: {f_temp}')
    # Plot the original data and the fitted Bessel function
    plt.plot(x_array, y_array, label='Original Data', linewidth=1)
    plt.plot(x_array, modified_bessel(x_array, *popt), label='Fitted Bessel Function', linewidth=2)
    plt.xlabel('eje x en pixels')
    plt.ylabel('Valor de intensidad normalizado')
    plt.title(plot_label)
    plt.show()    

    return new_params, f_temp

def fit_fixed_bessel(f0, x_array, y_array, plot_label):
    max_iter = 5

    for i in range(max_iter):
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


def bessel_analysis(processed_images,labels):
    y_arrays = []
    num_images = 0
    for img in processed_images:
        img_array = img.img_array
        centroid = img.centroid
        #  START: the following code is to align all images peaks
        x_size = np.shape(img_array)[0]
        x_range = int(x_size*0.9)
        xi = centroid[0] - int(x_range/2)
        xf = centroid[0] + int(x_range/2)
        print('CENTROIDE', centroid, 'X: ', centroid[0], 'xrange: ', xi, ' - ', xf)
        y_array = img_array[centroid[1],:]
        y_max = max(y_array)
        y_array = y_array[xi:xf]/y_max
        print('NEW DIMENSIONS',np.shape(y_array))
        #END
        y_arrays.append(y_array) #append resulting array to list of arrays
        num_images += 1

    x_array = np.array(range(-int(x_range/2),int(x_range/2)))
    #Estimating inital params
    # a*np.exp(b*x**2)* jv(0, c * x) - d*abs(x**2) + e
    number_of_peaks = 7  # As seen in your plot or image
    frequency_c = number_of_peaks / x_range
    amplitude_a = max(y_arrays[0])
    # You can start with b and d as small values or set them based on experience
    b_initial = 1e-5
    d_initial = 1e-5
    initial_params = [amplitude_a, -b_initial, frequency_c*4, d_initial, 0.4]
    print(f'Initial parameters: {initial_params}')

    frequencies = []
    global fixed_params
    for i in range(num_images):
        if i==0:
            fixed_params, f_temp = fit_bessel(initial_params,x_array,y_arrays[i],labels[i])

        f_temp = fit_fixed_bessel(f_temp, x_array, y_arrays[i], labels[i])
        frequencies.append(f_temp)

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

