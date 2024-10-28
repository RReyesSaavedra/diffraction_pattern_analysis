import cv2
import numpy as np
import matplotlib.pyplot as plt

# class Image:

#     def __init__(self, image, blockSize, C, morph_iterations):
#         self.blockSize = blockSize
#         self.C = C
#         self.morph_iterations = morph_iterations
#         self.image = image

def draw_contours(image, blockSize, C, morph_iterations):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    equalized = cv2.equalizeHist(gray)  # Histogram equalization to improve contrast
    blurred = cv2.GaussianBlur(equalized, (21, 21), 0)  # Apply Gaussian blur
    # 21 seems like a good number, any less and there is too much noise, lines are to wavy

    # Apply adaptive thresholding (MEAN gives better results than GAUSSIAN)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize, C)

    #uncomment if you want to see the effects of the filter
    # plt.imshow(adaptive_thresh, cmap='gray')
    # plt.title('testing gaussian')
    # plt.show()

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
    area1 = cv2.contourArea(final_contours[0])
    area2 = cv2.contourArea(final_contours[1])

    for contour in dz2_contours:
        area_dummy = cv2.contourArea(contour)
        diff_porcentual1 = 100*abs(area1 - area_dummy)/area_dummy
        diff_porcentual2 = 100*abs(area2 - area_dummy)/area_dummy
        
        #this if is to make sure we don't append two similar contours
        if diff_porcentual1 > 15 and diff_porcentual2 > 15:
            all_contours.append(contour)
    
    #We need to consider the case where we don't have the number of contours required.
    total_contours = np.size(all_contours)
    if wanted_contours > total_contours:
        print(f'After processing image {label}, we were only able to draw {total_contours}/{wanted_contours} contours')
        final_contours = all_contours
    else:
        final_contours = all_contours[:wanted_contours]
        
    cv2.drawContours(img_x, final_contours, -1, (0, 255, 0), 1)
    plt.figure()
    plt.imshow(img_x)
    plt.title(label)
    plt.axis('off')
    plt.show()
    
    return final_contours, boolean_break


def patern_properties(img, sorted_contours):
    (cx, cy), radius = cv2.minEnclosingCircle(sorted_contours[0])
    xi = int(cx-radius*0.9) #we multipliy by 0.9 to avoid including dark areas
    xf = cx

    y_temp = img[cy, xi:xf]/255
    y_min = np.min(y_temp)
    y_max = np.max(y_temp)
    if (y_max - y_min) > 0.25:
        
        





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

def img_proccesing(image_BGR, img_name):
    image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
    
    # Create copies for R, G, and B channels
    image_red = np.zeros_like(image_RGB)
    image_green = np.zeros_like(image_RGB)
    image_blue = np.zeros_like(image_RGB)

    # Assign the red channel
    image_red[:, :, 0] = image_RGB[:, :, 0]

    # Assign the green channel
    image_green[:, :, 1] = image_RGB[:, :, 1]

    # Assign the blue channel
    image_blue[:, :, 2] = image_RGB[:, :, 2]

    # For first dark ring:
    blockSize = 45
    C = 2
    morph_iterations = 0
    
    dz1_contours = draw_contours(image_RGB, blockSize, C, morph_iterations)
    
    # For second dark ring:
    blockSize = 45
    C = 3
    morph_iterations = 2
    
    dz2_contours = draw_contours(image_RGB, blockSize, C, morph_iterations)
    contours_sorted, break_boolean = sort_contours(dz1_contours, dz2_contours, image_RGB, img_name)
    
    # Return pattern properties
    return patern_properties(contours_sorted)

video = cv2.VideoCapture(path_video)
n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
dark_zone_area = []
circle_area = []
first_ring_area = []
centroids = []

print(f"Video with path: {path_video} read successfully\n\nFrames to be analyzed: {n_frames}")

global contador

for k in range(n_frames):
    contador = k+1
    frame_name = imgs_names[k]
    video.set(cv2.CAP_PROP_POS_FRAMES, k)  # Set the frame position
    boolean, frame = video.read()  # Read the frame
    if boolean:
        # Process the image and get the properties
        dark_zone_area_i, circle_area_i, first_ring_area_i, centroid = img_proccesing(frame, frame_name)
        
        # Append the results outside the function
        dark_zone_area.append(dark_zone_area_i)
        circle_area.append(circle_area_i)
        first_ring_area.append(first_ring_area_i)
        centroids.append(centroid)







def drive(self):
    print("This "+self.model+" is driving")

def stop(self):
    print("This "+self.model+" is stopped")