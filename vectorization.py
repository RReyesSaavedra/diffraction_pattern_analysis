import cv2
import numpy as np
import matplotlib.pyplot as plt



"blockSize" # en el método adaptative threshold es el tamaño en pixeles del cuadro de exploración
"C" # Factor multiplicativo flotante que puede admitir más o menos sensibilidad de deteccion
"morph_iterations determines the iterations of morphological operations"
def draw_contours(image, blockSize, C, morph_iterations):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Histogram equalization to improve contrast
    equalized = cv2.equalizeHist(gray)

    # Apply an even higher Gaussian blur
    blurred = cv2.GaussianBlur(equalized, (21, 21), 0)
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
    #cv2.RETR_TREE: Retrieves all contours and reconstructs a full hierarchy of nested contours. This is useful for understanding the parent-child relationship between contours.
    #other options are: cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP
    
    # Filter contours by circularity
    circularity_threshold=0.7
    
    #circular_contours = contours
    
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

def sort_contours(dz1_contours, dz2_contours, img_x, label):
    
    dz1_inner_contours = dz1_contours[:2]
    # Compare these contours with dz2 contours
    final_contours = dz1_inner_contours.copy()
    area1 = cv2.contourArea(final_contours[0])
    area2 = cv2.contourArea(final_contours[1])
    
    for i, contour in enumerate(dz2_contours):
        if len(final_contours) >= 4:
            break
        area_dummy = cv2.contourArea(contour)
        diff_porcentual1 = 100*abs(area1 - area_dummy)/area_dummy
        diff_porcentual2 = 100*abs(area2 - area_dummy)/area_dummy
        
        if diff_porcentual1 > 15 and diff_porcentual2 > 15:
            final_contours.append(contour)
    
    #We need to consider the case where we are not able to draw the 4 contours required.
    boolean_break = False
    if len(final_contours) < 4:
        boolean_break = True
        
    cv2.drawContours(img_x, final_contours, -1, (0, 255, 0), 1)
    plt.figure()
    plt.imshow(img_x)
    plt.title(label)
    plt.axis('off')
    plt.show()
    
    return final_contours, boolean_break


def patern_properties(sorted_contours):
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