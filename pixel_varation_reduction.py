import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def kmeans_segmentation(input_folder, output_folder, k=8):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    file_list = os.listdir(input_folder)

    for file_name in file_list:
        # Read the image
        image_path = os.path.join(input_folder, file_name)
        image = cv2.imread(image_path)

        # Resize the image
        #image = cv2.resize(img, (984, 1010))

        # Reshape the image into a 2D array of pixels and 3 color values (RGB)
        pixel_vals_reshape = image.reshape((-1, 1))

        # Convert to float type
        pixel_vals = np.float32(pixel_vals_reshape)

        # Set criteria for k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.95)
        k = 3

        # Perform k-means clustering
        retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Convert data into 8-bit values
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]

        # Reshape data into the original image dimensions
        segmented_image = segmented_data.reshape((image.shape))

        # Save the segmented image
        output_path = os.path.join(output_folder, file_name)
        cv2.imwrite(output_path, segmented_image)

if __name__ == "__main__":
    input_folder = '/Clustered_tiff_images/cluster_1'
    output_folder = '/input_seg_images/cluster_1'

    kmeans_segmentation(input_folder, output_folder)
    print('cluster 1 is done')

if __name__ == "__main__":
    input_folder = '/Clustered_tiff_images/cluster_2'
    output_folder = '/input_seg_images/cluster_2'


    kmeans_segmentation(input_folder, output_folder)
    print('cluster 2 is done')

if __name__ == "__main__":
    input_folder = '/Clustered_tiff_images/cluster_3'
    output_folder = '/input_seg_images/cluster_3'

    kmeans_segmentation(input_folder, output_folder)
    print('cluster 3 is done')