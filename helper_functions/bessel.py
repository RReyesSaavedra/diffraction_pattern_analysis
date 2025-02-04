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
        params = popt
        
    print(f'NEW PARAMS: {params}')
    # Plot the original data and the fitted Bessel function
    plt.plot(x_array, y_array, label='Original Data', linewidth=1)
    plt.plot(x_array, modified_bessel(x_array, *popt), label='Fitted Bessel Function', linewidth=2)
    plt.xlabel('eje x en pixels')
    plt.ylabel('Valor de intensidad normalizado')
    plt.title(plot_label)
    plt.show()    

    return popt

def fit_fixed_bessel(f0, x_array, y_array, plot_label):
    max_iter = 5
    # Wrapper function to fit only `e` using curve_fit
    def wrapper_bessel_function(x_array, e):
        return fixed_bessel(x_array, y_array, e)

    for i in range(max_iter):
        popt, pcov = curve_fit(wrapper_bessel_function, x_array, y_array, p0=[f0])
        
        f0 = popt[0] # Update param with the optimized one
    
    f_n = f0
    # Plot the original data and the fitted Bessel function
    plt.plot(x_array, y_array, label='Original Data', linewidth=1)
    plt.plot(x_array, fixed_bessel(x_array, y_array,f_n), label=f'Fitted Bessel Function. {plot_label}', linewidth=2)
    plt.xlabel('eje x en pixels')
    plt.ylabel('Valor de intensidad normalizado')
    plt.title(plot_label)
    plt.legend()
    plt.show()
    return f_n


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
    number_of_peaks = 7  # As seen in your plot or image
    frequency_c = number_of_peaks / x_range
    amplitude_a = max(y_arrays[0])
    print(amplitude_a)
    # You can start with b and d as small values or set them based on experience
    b_initial = 1e-5
    d_initial = 1e-5 
    initial_params = [amplitude_a, -b_initial, frequency_c*4, d_initial, 0.4]
    print(f'Initial parameters: {initial_params}')

    frequencies = []
    global fixed_params
    for i in range(num_images):
        if i==0:
            fixed_params = fit_bessel(initial_params,x_array,y_arrays[i],labels[i])

        frequency = fit_fixed_bessel(fixed_bessel, *fixed_params)
        frequencies.append(frequency)



    # adjusted_bessel_function = np.zeros_like(x_array)
    # for i, x in enumerate(x_array):
    #     adjusted_bessel_function[i] = modified_bessel(x, *initial_params)

    # plt.plot(x_array, adjusted_bessel_function, linewidth=2)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Original Bessel function')
    # plt.show()

    # # Fit the modified Bessel function to the data
    # n_concentraciones = np.shape(y_data_all_frames)[0]

    # fixed_params = []
    return 0

"""
plt.scatter(Molaridad, periodos/periodos[0]*100)
plt.xlabel('Molaridad')
plt.ylabel('Periodo Espacial')
plt.title("Molaridad vs Periodicidad")
plt.grid(True)
plt.show()
"""