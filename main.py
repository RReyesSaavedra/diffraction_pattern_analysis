import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from helper_functions import vectorization, area_analysis
#from fourier import fourier_analysis
#from bessel_adjustment import bessel_analysis

#bellow are the only values that need to be modified
folder_path = '/Users/rodolfo.reyes/Documents/img_KCl_base'
global sample_size, num_samples
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
            #option = int(input('\nThese are the available options for anlyzing your pattern images: \n1.Fourier\n2.Bright areas and Bessel\n3.All\n\nPlease enter a number: '))
            option = 2  #option testing
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
        processed_images = vectorization(img_paths)
        error, diff = area_analysis(processed_images)
        error, diff = bessel_analysis(processed_images)
    elif option ==3:
        processed_images = vectorization(img_paths)
        error, diff = area_analysis(processed_images)
        error, diff = bessel_analysis(processed_images)
        error, diff = fourier_analysis(processed_images)
        
