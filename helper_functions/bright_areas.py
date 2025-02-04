import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import config

#cada imagen será un objeto con 5 propiedades
class ProcessedImage:
    def __init__(self, dark_zone_area_i, circle_area_i, first_ring_area_i, centroid, img_array):
        self.dark_zone_area_i = dark_zone_area_i
        self.circle_area_i = circle_area_i
        self.first_ring_area_i = first_ring_area_i 
        self.centroid = centroid
        self.img_array = img_array

    def __repr__(self):
        return (f"ProcessedImage(dark_zone_area_i={self.dark_zone_area_i}, "
                f"circle_area_i={self.circle_area_i}, "
                f"first_ring_area_i={self.first_ring_area_i}, "
                f"centroid={self.centroid})")


"blockSize" # en el método adaptative threshold es el tamaño en pixeles del cuadro de exploración
"C" # Factor multiplicativo flotante que puede admitir más o menos sensibilidad de deteccion
"morph_iterations determines the iterations of morphological operations"
def draw_contours(image, blockSize, C, morph_iterations):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    equalized = cv2.equalizeHist(gray)  # Histogram equalization to improve contrast
    blurred = cv2.GaussianBlur(equalized, (21, 21), 0)  # Apply Gaussian blur
    # 21 seems like a good number, any less and there is too much noise, lines are to wavy

    # Apply adaptive thresholding (MEAN gives better results than GAUSSIAN)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize, C)

    #uncomment if you want to see the effects of the filter
    #plt.imshow(adaptive_thresh, cmap='gray')
    #plt.title(f'testing mean treash C={C}, morph_iter={morph_iterations}')
    #plt.show()

    # Use morphological operations with different kernels
    kernel_dilate = np.ones((5, 5), np.uint8)
    kernel_erode = np.ones((3, 3), np.uint8)
    morph = cv2.dilate(adaptive_thresh, kernel_dilate, iterations=morph_iterations)
    morph = cv2.erode(morph, kernel_erode, iterations=morph_iterations)
    #going past 3 iterations moves the centroide too much 

    # Find contours with the improved edge map
    contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.RETR_TREE: Retrieves all contours and reconstructs a full hierarchy of nested contours.
    #other options are: cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP
    
    # Filter contours by circularity
    circularity_threshold=0.7
    circular_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        #remove all non wanted contours, that are either too small or not circular enough
        if circularity > circularity_threshold and area>3000:
            circular_contours.append(contour)

    # Filter and draw only the most inner rings (the two largest contours by area)
    all_contours_sorted = sorted(circular_contours, key=cv2.contourArea)
    
    return all_contours_sorted


def sort_contours(dz1_contours, dz2_contours, img_x, label, wanted_contours):
    #the first 2 inner contours are obtained with different draw_contours parameters for better results,
    #the rest are all obtained with same fixed parameters.
    dz1_inner_contours = dz1_contours[:2]
    all_contours = dz1_inner_contours.copy()
    area1 = cv2.contourArea(all_contours[0])
    area2 = cv2.contourArea(all_contours[1])

    for contour in dz2_contours:
        area_dummy = cv2.contourArea(contour)
        diff_porcentual1 = 100*abs(area1 - area_dummy)/area_dummy
        diff_porcentual2 = 100*abs(area2 - area_dummy)/area_dummy
        
        #this if is to make sure we don't append two similar contours
        #we need to check against both of the contours already present on "all_contous variable"
        if diff_porcentual1 > 15 and diff_porcentual2 > 15:
            all_contours.append(contour)
    
    #We need to consider the case where we don't have the number of contours required.
    total_contours = len(all_contours)
    if wanted_contours > total_contours:
        print(f'After processing image {label}, we were only able to draw {total_contours}/{wanted_contours} contours')
        final_contours = all_contours
        boolean_break = True
    else:
        final_contours = all_contours[:wanted_contours]
        boolean_break = False

    # cv2.drawContours(img_x, final_contours, -1, (0, 255, 0), 1)
    # plt.figure()
    # plt.imshow(img_x)
    # plt.title(label)
    # plt.axis('off')
    # plt.show()
    
    return final_contours, boolean_break


def patern_properties(img_array, sorted_contours):
    (cx0, cy0), radius0 = cv2.minEnclosingCircle(sorted_contours[0])
    (cx1, cy1), radius1 = cv2.minEnclosingCircle(sorted_contours[1])

    x_1 = int(cx0-radius1) 
    x_origin = int(cx0)
    y_origin = int(cy0)
    y_temp = img_array[y_origin, x_1:x_origin]/255
    y_mean = np.mean(y_temp)
    y_min = np.min(y_temp)
    #threshold value is going to a combiation of min value and average value
    thresh = 0.66*y_min + 0.34*y_mean

    x_0 = int((cx0-radius0)*0.85) #we multipliy by 0.85 to avoid including dark areas at the edges
    first_ring_array = img_array[y_origin, x_0:x_origin]/255

    #FINISH CODE
    if any(first_ring_array) < thresh:
        print("this contour is not the contour of the circle, it's the contour of an outer ring")

    area_c1 = cv2.contourArea(sorted_contours[0])
    area_c2 = cv2.contourArea(sorted_contours[1])
    area_c3 = cv2.contourArea(sorted_contours[2])
    area_c4 = cv2.contourArea(sorted_contours[3])
    
    # Calculate areas
    dark_zone_area_i = (area_c4 - area_c3) + (area_c2 - area_c1)
    circle_area_i = area_c1
    first_ring_area_i = area_c3 - area_c2
    
    M = cv2.moments(sorted_contours[0])
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroid = (cX, cY)
    else:
        print('No centroid found')
        centroid = (250, 250)
    
    # Return calculated properties
    return dark_zone_area_i, circle_area_i, first_ring_area_i, centroid

def img_processing(image_BGR, img_name, wanted_contours):
    image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
    #gray = cv2.cvtColor(image_RGB, cv2.COLOR_RGB2GRAY) 
    
    # Create copies for R, G, and B channels, we keep arrays three dimentional to convert to gray scale alter
    image_red = np.zeros_like(image_RGB)
    image_green = np.zeros_like(image_RGB)
    image_blue = np.zeros_like(image_RGB)
    image_red[:, :, 0] = image_RGB[:, :, 0]    # Assign the red channel
    image_green[:, :, 1] = image_RGB[:, :, 1]    # Assign the green channel
    image_blue[:, :, 2] = image_RGB[:, :, 2]    # Assign the blue channel

    #select what chanel you want to use:
    image = image_red
    img_array = image[:, :, 0] #don't forget to change index if you change channel
    
    # For first dark ring:
    blockSize = 45
    C = 2
    morph_iterations = 0    
    dz1_contours = draw_contours(image, blockSize, C, morph_iterations)
    
    # For second dark ring:
    blockSize = 45
    C = 3
    morph_iterations = 2
    dz2_contours = draw_contours(image, blockSize, C, morph_iterations)

    contours_sorted, break_boolean = sort_contours(dz1_contours, dz2_contours, image, img_name, wanted_contours)
    # Obtener propiedades del patrón
    dark_zone_area_i, circle_area_i, first_ring_area_i, centroid = patern_properties(img_array, contours_sorted)

    #Declaring array again since it was modified while getting centroid and properties
    image_red = image_RGB[:, :, 0] #keeping only red channel
    img_array = image_red   

    # Devolver objeto de clase
    return ProcessedImage(dark_zone_area_i, circle_area_i, first_ring_area_i, centroid, img_array)

def vectorization(img_paths):
    processed_images = []  # Lista de objetos `ProcessedImage`

    for path in img_paths:
        temp_BGR = cv2.imread(path)
        img_name = os.path.splitext(os.path.basename(path))[0]

        # Procesar imagen y agregar el objeto resultante a la lista
        processed_img = img_processing(temp_BGR, img_name, config.wanted_contours)
        processed_images.append(processed_img)

    # Devolver resultados como lista de objetos
    return processed_images

def plot_centroids(centroids):
   # Plot the centroids and arrows using matplotlib
    plt.figure(figsize=(10, 10))

    for i in range(len(centroids) - 1):
        cX1, cY1 = centroids[i]
        cX2, cY2 = centroids[i + 1]
        plt.scatter(cX1, cY1, color='blue')
        plt.text(cX1, cY1, str(i+1), color='blue', fontsize=12)
        plt.arrow(cX1, cY1, cX2 - cX1, cY2 - cY1, head_width=0.5, head_length=0.5, fc='green', ec='green')

    # Plot the last point
    cX_last, cY_last = centroids[-1]
    plt.scatter(cX_last, cY_last, color='blue')
    plt.text(cX_last, cY_last, str(len(centroids)), color='blue', fontsize=12)
    plt.title("Centroids of Contours with Arrows")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().invert_yaxis()  # Invert y axis to match the image coordinate system
    plt.show()

def area_analysis(processed_images):
    # Calculate total bright area
    total_bright_area = [img.circle_area_i + img.first_ring_area_i for img in processed_images]
    y = np.array(total_bright_area) / total_bright_area[0] * 100
    labels = [str(round(m,3))+'M' for m in config.molaridades]

    # Create scatter plot
    x_array = range(1,len(total_bright_area)+1)
    plt.scatter(x_array, y, label="Bright Area Porcentage")

    # Set y-axis limit to the max value
    plt.ylim(min(y)*0.8, max(y) * 1.1)  # Extends 10% above max value

    # Draw vertical lines and place labels
    for i in range(1, config.num_samples+1):
        x = i * config.sample_size + 0.5
        plt.plot([x, x], [0, max(y) * 1.3], 'o-')   # Keep your original label

        # Place label slightly to the left and at the bottom
        plt.text(x - 0.2, min(y*0.9), labels[i - 1], color='red', fontsize=10, 
                horizontalalignment='right', verticalalignment='bottom')

    # Aesthetics
    plt.title('Bright Zone Area')
    plt.xlabel('Sample number')
    plt.ylabel('Percentage of first sample area')
    plt.xticks(x_array)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Extract centroids from each ProcessedImage object
    centroids = [obj.centroid for obj in processed_images]
    # Now call the function
    plot_centroids(centroids)

    error = 0
    diff = [0,0,0]
    return error, diff

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