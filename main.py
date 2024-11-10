import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from helper_functions import vectorization
#from fourier import fourier_analysis
#from bessel_adjustment import bessel_analysis

#bellow are the only values that need to be modified
folder_path = '/Users/Rudy/Documents/img_KCl_base'
sample_size = 2
num_samples = 6
molaridades = [0.000001, 0.1875, 0.375, 0.75, 1.5, 3] #0 values will cause errors

imgs_names = sorted(os.listdir(folder_path))
img_paths = []

#getting the complete path of the images
for name in imgs_names:
    if name != '.DS_Store':
        path = os.path.join(folder_path, name)
        img_paths.append(path)

expected_n = num_samples*sample_size
num_images = np.size(img_paths)

error = False
if expected_n == num_images:
    #before starting we check all images can be read succesfully
    c = 0
    for s in range(num_samples):
        for i in range(sample_size):
            temp_img = cv2.imread(img_paths[c])
            if temp_img is None:
                print(f"\nError reading image: {img_paths[c]}")
                error = True
            else:
                print(f"\nImage '{img_paths[c]}' was read successfully")
            c += 1
    #If there were no errors reading the images we continue to analyse the images
    if not error: 
        option = 0
        while option not in [1,2,3]:
            option = int(input('\nThese are the available options for anlyzing your pattern images: \n1.Fourier\n2.Bright areas and Bessel\n3.All\n\nPlease enter a number: '))
            if option not in [1,2,3]:
                print(option,' is not a valid option. Please enter 1, 2 or 3')
else:
    print(f'Total number of photos on {folder_path}: {num_images}. \nExpected number of photos: {sample_size}x{num_samples}={expected_n}. Check sample size and number of samples values.')
    error = True

if error:
    print('\nPlease revise and try again.')
else:
    if option == 1:
        fourier_analysis(img_paths,sample_size,num_samples)
    elif option ==2:
        error = vectorization(img_paths,sample_size,num_samples)
        #y_arrays, centroid = area_analysis(img_paths,sample_size,num_samples)
        #bessel_analysis(img_paths,sample_size,num_samples,y_arrays, centroid)
    elif option ==3:
        fourier_analysis(img_paths,sample_size,num_samples)
        y_arrays, centroid = area_analysis(img_paths,sample_size,num_samples)
        bessel_analysis(img_paths,sample_size,num_samples,y_arrays, centroid)

        
# dark_zone_area = []
# circle_area = []
# first_ring_area = []
# centroids = []

# for k in range(n_frames):
#     contador = k+1
#     frame_name = imgs_names[k]
#     video.set(cv2.CAP_PROP_POS_FRAMES, k)  # Set the frame position
#     boolean, frame = video.read()  # Read the frame
#     if boolean:
#         # Process the image and get the properties
#         dark_zone_area_i, circle_area_i, first_ring_area_i, centroid = img_proccesing_main(frame, frame_name)
        
#         # Append the results outside the function
#         dark_zone_area.append(dark_zone_area_i)
#         circle_area.append(circle_area_i)
#         first_ring_area.append(first_ring_area_i)
#         centroids.append(centroid)

# total_bright_area = [circle + first_ring for circle, first_ring in zip(circle_area, first_ring_area)]




# total_bright_area = [circle + first_ring for circle, first_ring in zip(circle_area, first_ring_area)]

# def err_value_4bam(areas):
#     error_array = np.zeros(int(n_frames/2))
#     for i in range(int(n_frames/2)):
#         j = 2*i
#         k = 2*i+1
#         error_array[i] = 100 * abs(areas[j] - areas[k]) / areas[j]
#     return error_array

# error_values_bright_area = err_value_4bam(total_bright_area)
# print(f'\n\nESTIMATED PERCENTAGE OF ERROR OF EACH PULSE: {error_values_bright_area}')
# BA_MEASUREMENT_ERROR = np.mean(error_values_bright_area)
# print(f'\n\nESTIMATED MEASUREMENT ERROR: {BA_MEASUREMENT_ERROR}')

# def promediar_areas4y5(areas):
#     areas_prom = np.zeros(int(n_frames/2))
#     for i in range(int(n_frames/2)):
#         j = 2*i
#         k = 2*i+1
#         areas_prom[i] = (areas[j] + areas[k]) / 2
#     return areas_prom

# dz_area_prom = promediar_areas4y5(dark_zone_area)

# Molaridad = [0.000001, 0.1875, 0.375, 0.75, 1.5, 3]

# plt.scatter(Molaridad, dz_area_prom)
# plt.title('Dark Zone Average Area (Frame 4 and 5)')
# plt.xlabel('Molarity')
# plt.ylabel('Area in Pixels')
# plt.grid(True)
# plt.show()

# bz_area_prom = promediar_areas4y5(total_bright_area)

# plt.scatter(Molaridad, bz_area_prom/bz_area_prom[0]*100)
# plt.title('Bright Zone Average Area (Frame 4 and 5)')
# plt.xlabel('Molarity')
# plt.ylabel('Area in Pixels')
# plt.grid(True)
# plt.show()


# #area_luminosa_entre_M = bz_area_prom/Molaridad
# #print(f'\n\nRelación área luminosa y Molaridad: {area_luminosa_entre_M}')

# #n_of_values is the number of photos per Molarity
# def diff_value(values,sample_size,method_label):
#     n = np.size(values)
#     num_of_samples = int(n/sample_size)
#     diff_array = np.zeros(num_of_samples)
#     std_dev = np.zeros(num_of_samples)
#     mean_value = np.zeros(num_of_samples)
    
#     print(f'all values:  \t {values}')

#     for i in range(num_of_samples):
#         ia =int(i*sample_size)
#         ib =ia+sample_size
#         temp_array = np.array(values[ia:ib])
#         print(f'temp array: {temp_array}')
#         std_dev[i] = np.std(temp_array)
#         mean_value[i] = np.mean(temp_array)
#         if not i == 0:
#             diff_array[i] = mean_value[i]- mean_value[i-1]
#             print(diff_array[i])
            
        
#     print('\n'+method_label+'\n')
#     print(mean_value)
#     print(diff_array)
#     print(std_dev)
#     print('goodbye')

        
    
#     return diff_array, std_dev, mean_value

# diff_bza, error_bza, mean_bza = diff_value(total_bright_area,2,"Método de las areas brillosas")

# # Extract x and y coordinates
# x_coords = [cX for cX, cY in centroids]
# print(x_coords)
# y_coords = [cY for cX, cY in centroids]
# print(y_coords)


# # Plot the centroids and arrows using matplotlib
# plt.figure(figsize=(10, 10))

# for i in range(len(centroids) - 1):
#     cX1, cY1 = centroids[i]
#     cX2, cY2 = centroids[i + 1]
#     plt.scatter(cX1, cY1, color='blue')
#     plt.text(cX1, cY1, str(i+1), color='blue', fontsize=12)
#     plt.arrow(cX1, cY1, cX2 - cX1, cY2 - cY1, head_width=0.5, head_length=0.5, fc='green', ec='green')

# # Plot the last point
# cX_last, cY_last = centroids[-1]
# plt.scatter(cX_last, cY_last, color='blue')
# plt.text(cX_last, cY_last, str(len(centroids)), color='blue', fontsize=12)
# plt.title("Centroids of Contours with Arrows")
# plt.xlabel("X Coordinate")
# plt.ylabel("Y Coordinate")
# plt.gca().invert_yaxis()  # Invert y axis to match the image coordinate system
# plt.show()

# global x_dimension

# def bessel_patern(img, x, y):
#     # Convertir las imágenes a escala de grises    
#     imagen1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
#     xi = int(x-x_dimension/2)
#     xf = int(x+x_dimension/2)

#     red_255 = imagen1[:, :, 0]
#     y_temp = red_255[y, xi:xf]/255    
#     #bessel = average_image[y, xi:xf]
    
#     return  y_temp

# y_data_all_frames = []

# for k in range(int(n_frames/2)):
#     contador = k+1
#     video.set(cv2.CAP_PROP_POS_FRAMES,2*k); #colocamos el número de frame que deseamos leer
#     boolean, frame1 = video.read() #Leemos el frame
#     video.set(cv2.CAP_PROP_POS_FRAMES,2*k+1); #colocamos el número de frame que deseamos leer
#     boolean, frame2 = video.read() #Leemos el frame
#     if k == 0: #solo necesitamos definir x_data una vez
#         x_dimension = np.shape(frame)[0] - 60 #extraemos pixeles porque el centro se mueve y tenemos que ajustar las funciones al mismo centro
#         x_data = np.linspace(-int(x_dimension/20),int(x_dimension/20),x_dimension)
#     y_temp = bessel_patern(frame1,x_coords[k],y_coords[k])
#     y_data_all_frames.append(y_temp)
     
#     y_temp2 = bessel_patern(frame2,x_coords[k],y_coords[k])
#     y_data_all_frames.append(y_temp2)

    
# from scipy.optimize import curve_fit
# from scipy.special import jv


# # Define the modified Bessel function to fit
# def modified_cosine_function(x, a, b, c, d, e):
#     return a*np.exp(b*x**2)* np.cos(c * x) - d*abs(x**2) + e
    
    
# # Define the modified Bessel function to fit
# def modified_bessel_function(x, a, b, c, d, e):
#     return a*np.exp(b*x**2)* jv(0, c * x) - d*abs(x**2) + e

# # Function to fit the Bessel function with stopping criteria
# def fit_bessel_function(params, x_array, y_array, plot_label):
#     tol=1e-6
#     max_iter=5
#     prev_residual = np.inf
#     no_improvement_count = 0
    
#     # Plot the original data and the fitted Bessel function
#     #plt.plot(x_array, modified_bessel_function(x_array, *params), label='Fitted Bessel Function', linewidth=2)
#     #plt.xlabel('eje x en pixels')
#     #plt.ylabel('Valor de intensidad normalizado')
#     #plt.title(f'patrón tipo Bessel Original')
#     #plt.show()   

#     for i in range(max_iter):
#         popt, _ = curve_fit(modified_bessel_function, x_array, y_array, p0=params)
#         residual = np.sum((modified_bessel_function(x_array, *popt) - y_array) ** 2)
#         prev_residual = residual
#         params = popt
        
#     # Plot the original data and the fitted Bessel function
#     plt.plot(x_array, y_array, label='Original Data', linewidth=1)
#     plt.plot(x_array, modified_bessel_function(x_array, *popt), label='Fitted Bessel Function', linewidth=2)
#     plt.xlabel('eje x en pixels')
#     plt.ylabel('Valor de intensidad normalizado')
#     plt.title(plot_label)
#     plt.show()    

#     return popt


# # Convert x_data and y_data_all_frames_all_frames to numpy arrays if they are not already
# x_data = np.array(x_data)
# y_data_all_frames = np.array(y_data_all_frames)

# # Calculating estimated parameters
# number_of_peaks = 7  # As seen in your plot
# x_range = x_data.max() - x_data.min()
# frequency_c = number_of_peaks / x_range
# amplitude_a = y_data_all_frames.max()

# # You can start with b and d as small values or set them based on experience
# b_initial = 1e-5
# d_initial = 1e-5 

# # New initial parameters based on estimation
# initial_params = [amplitude_a, -b_initial, frequency_c*5, d_initial, 0.4]
# #initial_params = [amplitude_a, -b_initial, frequency_c*5, d_initial, 0.4]

# print(f'Initial parameters: {initial_params}')

# adjusted_bessel_function = np.zeros_like(x_data)
# for i, x in enumerate(x_data):
#     adjusted_bessel_function[i] = modified_bessel_function(x, *initial_params)

# plt.plot(x_data, adjusted_bessel_function, linewidth=2)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('Original Bessel function')
# plt.show()

# # Fit the modified Bessel function to the data
# n_concentraciones = np.shape(y_data_all_frames)[0]

# fixed_params = []

# def fixed_bessel_function(x_array,y_array,f):
#     global fixed_params
#     global a,b,c,d,e
#     label='finding fixed parameters'
#     if len(fixed_params) == 0:
#         fixed_params = fit_bessel_function(initial_params, x_array, y_array, label)
#         a,b,c,d,e = fixed_params
#     else:
#         c=f
#     value = a * np.exp(b * x_array**2) * jv(0, c * x_array) - d * np.abs(x_array**2) + e
    
#     return value

# def fit_bessel_function_simple(f0, x_array, y_array, plot_label):
#     max_iter = 5
#     # Wrapper function to fit only `e` using curve_fit
#     def wrapper_bessel_function(x_array, e):
#         return fixed_bessel_function(x_array, y_array, e)

#     for i in range(max_iter):
#         popt, pcov = curve_fit(wrapper_bessel_function, x_array, y_array, p0=[f0])
        
#         f0 = popt[0] # Update param with the optimized one
    
#     f_n = f0
#     # Plot the original data and the fitted Bessel function
#     plt.plot(x_array, y_array, label='Original Data', linewidth=1)
#     plt.plot(x_array, fixed_bessel_function(x_array, y_array,f_n), label=f'Fitted Bessel Function. {plot_label}', linewidth=2)
#     plt.xlabel('eje x en pixels')
#     plt.ylabel('Valor de intensidad normalizado')
#     plt.title(plot_label)
#     plt.legend()
#     plt.show()
#     return f_n

# def get_frequencies():
#     global fixed_params 
#     frequencies = []
#     f_inicial=1
#     for i in range(n_concentraciones):
#         index = int(i/2)
#         label = f"Molaridad {Molaridad[index]}"
#         f = fit_bessel_function_simple(f_inicial, x_data, y_data_all_frames[i], label)
#         frequencies.append(f)
    
#     return frequencies

# frequencies = get_frequencies()

# print(f"\nFrequencies: {frequencies}")
# diff, error = diff_value(frequencies, 2)

# print(f'\nDiferencia porcentual entre concentraciones {diff}')
# print(f'\n\nError porcentual entre fotos de mismas concentraciones {error}')

# frequencies = np.array(frequencies)  # Ejemplo de frecuencias en Hz
# periodos = 1/frequencies

# """
# plt.scatter(Molaridad, periodos/periodos[0]*100)
# plt.xlabel('Molaridad')
# plt.ylabel('Periodo Espacial')
# plt.title("Molaridad vs Periodicidad")
# plt.grid(True)
# plt.show()
# """