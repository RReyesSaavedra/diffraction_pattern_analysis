sample_size = 2
num_samples = 6
molaridades = [0.000001, 0.1875, 0.375, 0.75, 1.5, 3] #0 values will cause errors
rings_to_analyze = 2 #number of rings to analyze, they most have very clear contours (blury rings won't be detected)
visible_rings = 3 # Number of rings seen in in your image
channel = 'green' #possible values: red, blue, green, gray, r+g
num_terms = 5 #5 seems to work best for polynomial adjustment
avg_delta = 3
