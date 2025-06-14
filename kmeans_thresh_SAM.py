import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from skimage import io, filters
from PIL import Image
# %matplotlib inline
import sys
import subprocess
import os
import random
import pandas as pd
import csv
from scipy.stats import sem
from scipy.stats import t
# from matplotlib.ticker import FuncFormatter


using_colab = False  # Set this variable as needed

if using_colab:
    import torch
    import torchvision
    print("PyTorch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA is available:", torch.cuda.is_available())

    # Install packages using subprocess
    packages_to_install = ["opencv-python", "matplotlib"]
    for package in packages_to_install:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    # Create a directory
    subprocess.check_call(["mkdir", "images"])

    # Download images
    subprocess.check_call(["wget", "-P", "images", "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/truck.jpg"])
    subprocess.check_call(["wget", "-P", "images", "https://raw.githubusercontent.com/facebookresearch/segment-anything/main/notebooks/images/groceries.jpg"])

    # Download a file
    subprocess.check_call(["wget", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"])

# Install 'segment-anything' package from GitHub
github_repo_url = "https://github.com/facebookresearch/segment-anything.git"
subprocess.check_call([sys.executable, "-m", "pip", "install", f'git+{github_repo_url}'])


if not os.path.exists("images"):
    os.makedirs("images")

if not os.path.exists("sam_vit_h_4b8939.pth"):
    subprocess.check_call(["wget", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"])

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(2), np.array([0.6])], axis=0)
    else:
        color = np.array([255/255, 0/255, 0.6])
        #color = np.array([30/255, 144/255, 255/255, 0.6])
        #print(color.shape)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, cmap = 'gray')
    # grayscale_mask = np.mean(mask_image, axis=-1)
    # ax.imshow(grayscale_mask, cmap='gray', alpha=0.6)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

# Define the show_mask function to display masks
def show_mask(mask, ax):
    ax.imshow(mask, cmap='viridis')  # You can adjust the colormap as needed

# Define the save_mask function to save masks as numpy arrays
def save_mask(mask, score, index, folder_path):
    filename = f"mask_score_index_{index}.npy"
    file_path = os.path.join(folder_path, filename)
    np.save(file_path, mask)

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

##For the saved ground truth in the drive
#only thewhite pixels with 255
def get_white_pixel_coordinates(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        coordinates = []

        for y in range(height):
            for x in range(width):
                pixel_value = img.getpixel((x, y))
                if pixel_value == 255:
                    coordinates.append((x, y))

    return coordinates
def generate_ones_array(length):
    return [1] * length

#cluster_1
def main():
    image_path = "/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/centers/thresholded_centroid_cluster_1.png"  # Replace this with the actual path of your image file
    #image_path = th5
    white_pixel_coordinates = get_white_pixel_coordinates(image_path)
    print(len(white_pixel_coordinates))
    #print(white_pixel_coordinates)
    point_array = generate_ones_array(len(white_pixel_coordinates))

    print(len(point_array))
    return white_pixel_coordinates, point_array


if __name__ == "__main__":
    white_pixels, points = main()

# Assuming 'white_pixels' and 'points' are your original data arrays
num_points = len(white_pixels)
#indices = np.random.choice(num_points, size = int(0.1*num_points), replace=False)
indices = np.random.choice(num_points, size = 100, replace=False)
cord_100 = np.array(white_pixels)[indices]
print(cord_100.shape)

labels_100 = np.array(points)[indices]


image_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/input_seg_images/cluster_1'
# Iterate over each file in the folder

for filename in os.listdir(image_folder):
    if filename.endswith('.tif'):  # Assuming your images have a .tif extension
        # Load and process the image
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image_path = os.path.join(image_folder, filename)



        # Read the input image
        #data_arr = io.imread(input_image_path)

        # Create the output filename by extracting the base name and adding the target directory
        base_filename = os.path.splitext(filename)[0]

        # Create a predictor for each image
        predictor = SamPredictor(sam)
        predictor.set_image(image)
        #masks
        masks,scores, logits = predictor.predict(
            # point_labels = labels,
            # point_coords = low_dense_area,
            point_labels = labels_100,
            point_coords= cord_100,
            #mask_input= image_array,
            multimask_output= True,
            return_logits = False
        )
        #print("mask shape:", masks.shape)
        #Mask prediction by SAM

        # # Define the show_mask function to display masks
        # def show_mask(mask, ax):
        #     ax.imshow(mask, cmap='viridis')  # You can adjust the colormap as needed

        # # Define the save_mask function to save masks as numpy arrays
        # def save_mask(mask, score, index, folder_path):
        #     filename = f"mask_score_index_{index}.npy"
        #     file_path = os.path.join(folder_path, filename)
        #     np.save(file_path, mask)

        # Create a folder to save the mask files
        folder_path = "/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/pred_masks_np_arrays_seg/cluster_1"
        csv_folder = "/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6"
        os.makedirs(csv_folder, exist_ok=True)

        # Create a CSV file to store scores
        csv_filename = "mask6_scores_1.csv"
        csv_path = os.path.join(csv_folder, csv_filename)

        # Initialize a list to store scores
        mask_scores = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if i == 1:  # Check score for i == 1 only
                if score > 0.9:  #if score > 0.85 for sample 4
                    selected_index = 0
                else:
                    selected_index = 1
                
                # Save mask and score value for selected_index
                save_mask(masks[selected_index], scores[selected_index], selected_index, folder_path)
                print('index:',selected_index )
                print('socre:', scores[selected_index])
                
                # Append the selected_index and its corresponding score to the mask_scores list
                mask_scores.append([selected_index, scores[selected_index]])

        # Now, you can load the mask with index 1 without specifying the score
                loaded_mask = np.load(os.path.join(folder_path, f"mask_score_index_{selected_index}.npy"))
        #print(np.unique(loaded_mask))


                # Save the scores to a CSV file
                with open(csv_path, mode='w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(["Index", "Score"])
                    csv_writer.writerows(mask_scores)

        # Provided data_array
        da = loaded_mask
        # Convert the data_array to uint8 data type in the range [0, 255]
        data_array = np.clip(da, 0, 255)
        data_arr = data_array.astype(np.uint8)

        v = data_arr > 0
        data_arr[v] = 255

        # Save the final mask in the folder
        output_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/pred_mask_final_seg/cluster_1'
        output_filename = f"{base_filename}.png"
        output_path = os.path.join(output_folder, output_filename)

        # Save the image
        io.imsave(output_path, data_arr)


# Specify the paths to the two folders
output_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/pred_mask_final_seg/cluster_1'
#ground_truth_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/GT_final/cluster_1' #for sample 4 only 
ground_truth_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/GT_flip/cluster_1' #for sample 6 only 
# Create lists of file names in the two folders
output_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.png')])
ground_truth_files = sorted([f for f in os.listdir(ground_truth_folder) if f.endswith('.png')])

# Initialize a dictionary to store Dice coefficients and image pairs
dice_coefficients = {}
image_pairs = []

# Check if the number of images in both folders is the same
if len(output_files) != len(ground_truth_files):
    print("Error: The number of images in the two folders does not match.")
else:
    # Loop through ground truth and predicted images in order
    for ground_truth_file in ground_truth_files:
        # Extract the last number from the ground truth file name
        ground_truth_number = int(ground_truth_file.split("_")[-1].split(".")[0])

        # Search for the matching predicted file by comparing last numbers
        matching_predicted_file = None
        for predicted_file in output_files:
            predicted_number = int(predicted_file.split("_")[-1].split(".")[0])
            if ground_truth_number == predicted_number:
                matching_predicted_file = predicted_file
                break

        if matching_predicted_file:
            # Load ground truth and predicted images
            ground_truth = cv2.imread(os.path.join(ground_truth_folder, ground_truth_file), cv2.IMREAD_GRAYSCALE)
            predicted = cv2.imread(os.path.join(output_folder, matching_predicted_file), cv2.IMREAD_GRAYSCALE)

            # Calculate Dice coefficient
            intersection = np.sum(np.logical_and(ground_truth == 255, predicted == 255))
            total_pixels_gt = np.sum(ground_truth == 255)
            total_pixels_pred = np.sum(predicted == 255)

            dice_coefficient = 2.0 * intersection / (total_pixels_gt + total_pixels_pred)

            # Store the result in the dictionary
            image_pairs.append((ground_truth_file, matching_predicted_file))
            dice_coefficients[(ground_truth_file, matching_predicted_file)] = dice_coefficient
            print('dice coefficient:', dice_coefficient)
        else:
            print(f"No matching predicted file found for {ground_truth_file}")


# Specify the folder where you want to save the CSV file
csv_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/CSV_6_seg'

# Save Dice coefficients and image pairs to a CSV file in the specified folder
csv_filename = os.path.join(csv_folder, 'sample_6_k_means_seg_cluster_1.csv')

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Ground Truth Image', 'Predicted Image', 'Dice Coefficient'])
    for pair in image_pairs:
        dice = dice_coefficients.get(pair, 0.0)
        csv_writer.writerow([pair[0], pair[1], dice])

print("Dice coefficients saved to", csv_filename)

#cluster_2
def main():
    image_path = "/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/centers/thresholded_centroid_cluster_2.png"  # Replace this with the actual path of your image file
    #image_path = th5
    white_pixel_coordinates = get_white_pixel_coordinates(image_path)
    print(len(white_pixel_coordinates))
    #print(white_pixel_coordinates)
    point_array = generate_ones_array(len(white_pixel_coordinates))

    print(len(point_array))
    return white_pixel_coordinates, point_array


if __name__ == "__main__":
    white_pixels, points = main()

# Assuming 'white_pixels' and 'points' are your original data arrays
num_points = len(white_pixels)
#indices = np.random.choice(num_points, size = int(0.1*num_points), replace=False)
indices = np.random.choice(num_points, size =10000, replace=False)
cord_100 = np.array(white_pixels)[indices]
print(cord_100.shape)

labels_100 = np.array(points)[indices]
image_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/input_seg_images/cluster_2'
# Iterate over each file in the folder

for filename in os.listdir(image_folder):
    if filename.endswith('.tif'):  # Assuming your images have a .tif extension
        # Load and process the image
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image_path = os.path.join(image_folder, filename)



        # Read the input image
        #data_arr = io.imread(input_image_path)

        # Create the output filename by extracting the base name and adding the target directory
        base_filename = os.path.splitext(filename)[0]

        # Create a predictor for each image
        predictor = SamPredictor(sam)
        predictor.set_image(image)
        #masks
        masks,scores, logits = predictor.predict(
            # point_labels = labels,
            # point_coords = low_dense_area,
            point_labels = labels_100,
            point_coords= cord_100,
            #mask_input= image_array,
            multimask_output= True,
            return_logits = False
        )
        #print("mask shape:", masks.shape)
        #Mask prediction by SAM

        # # Define the show_mask function to display masks
        # def show_mask(mask, ax):
        #     ax.imshow(mask, cmap='viridis')  # You can adjust the colormap as needed

        # # Define the save_mask function to save masks as numpy arrays
        # def save_mask(mask, score, index, folder_path):
        #     filename = f"mask_score_index_{index}.npy"
        #     file_path = os.path.join(folder_path, filename)
        #     np.save(file_path, mask)

        # Create a folder to save the mask files
        folder_path = "/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/pred_masks_np_arrays_seg/cluster_2"
        csv_folder = "/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6"
        os.makedirs(csv_folder, exist_ok=True)

        # Create a CSV file to store scores
        csv_filename = "mask6_scores_1.csv"
        csv_path = os.path.join(csv_folder, csv_filename)

        # Initialize a list to store scores
        mask_scores = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if i == 1:  # Check score for i == 1 only
                if score > 0.9:
                    selected_index = 0
                else:
                    selected_index = 1
                
                # Save mask and score value for selected_index
                save_mask(masks[selected_index], scores[selected_index], selected_index, folder_path)
                print('index:',selected_index )
                print('socre:', scores[selected_index])
                
                # Append the selected_index and its corresponding score to the mask_scores list
                mask_scores.append([selected_index, scores[selected_index]])

        # Now, you can load the mask with index 1 without specifying the score
                loaded_mask = np.load(os.path.join(folder_path, f"mask_score_index_{selected_index}.npy"))
        #print(np.unique(loaded_mask))


                # Save the scores to a CSV file
                with open(csv_path, mode='w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(["Index", "Score"])
                    csv_writer.writerows(mask_scores)

        # Provided data_array
        da = loaded_mask
        # Convert the data_array to uint8 data type in the range [0, 255]
        data_array = np.clip(da, 0, 255)
        data_arr = data_array.astype(np.uint8)

        v = data_arr > 0
        data_arr[v] = 255

        # Save the final mask in the folder
        output_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/pred_mask_final_seg/cluster_2'
        output_filename = f"{base_filename}.png"
        output_path = os.path.join(output_folder, output_filename)

        # Save the image
        io.imsave(output_path, data_arr)


# Specify the paths to the two folders
output_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/pred_mask_final_seg/cluster_2'
#ground_truth_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/GT_final/cluster_2' #sample 4 only
ground_truth_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/GT_flip/cluster_2' #sample 6 only 
# Create lists of file names in the two folders
output_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.png')])
ground_truth_files = sorted([f for f in os.listdir(ground_truth_folder) if f.endswith('.png')])

# Initialize a dictionary to store Dice coefficients and image pairs
dice_coefficients = {}
image_pairs = []

# Check if the number of images in both folders is the same
if len(output_files) != len(ground_truth_files):
    print("Error: The number of images in the two folders does not match.")
else:
    # Loop through ground truth and predicted images in order
    for ground_truth_file in ground_truth_files:
        # Extract the last number from the ground truth file name
        ground_truth_number = int(ground_truth_file.split("_")[-1].split(".")[0])

        # Search for the matching predicted file by comparing last numbers
        matching_predicted_file = None
        for predicted_file in output_files:
            predicted_number = int(predicted_file.split("_")[-1].split(".")[0])
            if ground_truth_number == predicted_number:
                matching_predicted_file = predicted_file
                break

        if matching_predicted_file:
            # Load ground truth and predicted images
            ground_truth = cv2.imread(os.path.join(ground_truth_folder, ground_truth_file), cv2.IMREAD_GRAYSCALE)
            predicted = cv2.imread(os.path.join(output_folder, matching_predicted_file), cv2.IMREAD_GRAYSCALE)

            # Calculate Dice coefficient
            intersection = np.sum(np.logical_and(ground_truth == 255, predicted == 255))
            total_pixels_gt = np.sum(ground_truth == 255)
            total_pixels_pred = np.sum(predicted == 255)

            dice_coefficient = 2.0 * intersection / (total_pixels_gt + total_pixels_pred)

            # Store the result in the dictionary
            image_pairs.append((ground_truth_file, matching_predicted_file))
            dice_coefficients[(ground_truth_file, matching_predicted_file)] = dice_coefficient
            print('dice coefficient:', dice_coefficient)
        else:
            print(f"No matching predicted file found for {ground_truth_file}")


# Specify the folder where you want to save the CSV file
csv_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/CSV_6_seg'

# Save Dice coefficients and image pairs to a CSV file in the specified folder
csv_filename = os.path.join(csv_folder, 'sample_6_k_means_seg_cluster_2.csv')

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Ground Truth Image', 'Predicted Image', 'Dice Coefficient'])
    for pair in image_pairs:
        dice = dice_coefficients.get(pair, 0.0)
        csv_writer.writerow([pair[0], pair[1], dice])

print("Dice coefficients saved to", csv_filename)

#cluster_3
def main():
    image_path = "/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/centers/thresholded_centroid_cluster_3.png"  # Replace this with the actual path of your image file
    #image_path = th5
    white_pixel_coordinates = get_white_pixel_coordinates(image_path)
    print(len(white_pixel_coordinates))
    #print(white_pixel_coordinates)
    point_array = generate_ones_array(len(white_pixel_coordinates))

    print(len(point_array))
    return white_pixel_coordinates, point_array


if __name__ == "__main__":
    white_pixels, points = main()

# Assuming 'white_pixels' and 'points' are your original data arrays
num_points = len(white_pixels)
#indices = np.random.choice(num_points, size = int(0.1*num_points), replace=False)
indices = np.random.choice(num_points, size =10000, replace=False)
cord_100 = np.array(white_pixels)[indices]
print(cord_100.shape)

labels_100 = np.array(points)[indices]
image_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/input_seg_images/cluster_3'
# Iterate over each file in the folder

for filename in os.listdir(image_folder):
    if filename.endswith('.tif'):  # Assuming your images have a .tif extension
        # Load and process the image
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_image_path = os.path.join(image_folder, filename)



        # Read the input image
        #data_arr = io.imread(input_image_path)

        # Create the output filename by extracting the base name and adding the target directory
        base_filename = os.path.splitext(filename)[0]

        # Create a predictor for each image
        predictor = SamPredictor(sam)
        predictor.set_image(image)
        #masks
        masks,scores, logits = predictor.predict(
            # point_labels = labels,
            # point_coords = low_dense_area,
            point_labels = labels_100,
            point_coords= cord_100,
            #mask_input= image_array,
            multimask_output= True,
            return_logits = False
        )
        #print("mask shape:", masks.shape)
        #Mask prediction by SAM

        # # Define the show_mask function to display masks
        # def show_mask(mask, ax):
        #     ax.imshow(mask, cmap='viridis')  # You can adjust the colormap as needed

        # # Define the save_mask function to save masks as numpy arrays
        # def save_mask(mask, score, index, folder_path):
        #     filename = f"mask_score_index_{index}.npy"
        #     file_path = os.path.join(folder_path, filename)
        #     np.save(file_path, mask)

        # Create a folder to save the mask files
        folder_path = "/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/pred_masks_np_arrays_seg/cluster_3"
        csv_folder = "/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6"
        os.makedirs(csv_folder, exist_ok=True)

        # Create a CSV file to store scores
        csv_filename = "mask6_scores_1.csv"
        csv_path = os.path.join(csv_folder, csv_filename)

        # Initialize a list to store scores
        mask_scores = []
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if i == 1:  # Check score for i == 1 only
                if score > 0.9:
                    selected_index = 0
                else:
                    selected_index = 1
                
                # Save mask and score value for selected_index
                save_mask(masks[selected_index], scores[selected_index], selected_index, folder_path)
                print('index:',selected_index )
                print('socre:', scores[selected_index])
                
                # Append the selected_index and its corresponding score to the mask_scores list
                mask_scores.append([selected_index, scores[selected_index]])

        # Now, you can load the mask with index 1 without specifying the score
                loaded_mask = np.load(os.path.join(folder_path, f"mask_score_index_{selected_index}.npy"))
        #print(np.unique(loaded_mask))


                # Save the scores to a CSV file
                with open(csv_path, mode='w', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(["Index", "Score"])
                    csv_writer.writerows(mask_scores)

        # Provided data_array
        da = loaded_mask
        # Convert the data_array to uint8 data type in the range [0, 255]
        data_array = np.clip(da, 0, 255)
        data_arr = data_array.astype(np.uint8)

        v = data_arr > 0
        data_arr[v] = 255

        # Save the final mask in the folder
        output_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/pred_mask_final_seg/cluster_3'
        output_filename = f"{base_filename}.png"
        output_path = os.path.join(output_folder, output_filename)

        # Save the image
        io.imsave(output_path, data_arr)


# Specify the paths to the two folders
output_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/pred_mask_final_seg/cluster_3'
#ground_truth_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/GT_final/cluster_3' #sample 4 only
ground_truth_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/GT_flip/cluster_3' #sample 6 only 
# Create lists of file names in the two folders
output_files = sorted([f for f in os.listdir(output_folder) if f.endswith('.png')])
ground_truth_files = sorted([f for f in os.listdir(ground_truth_folder) if f.endswith('.png')])

# Initialize a dictionary to store Dice coefficients and image pairs
dice_coefficients = {}
image_pairs = []

# Check if the number of images in both folders is the same
if len(output_files) != len(ground_truth_files):
    print("Error: The number of images in the two folders does not match.")
else:
    # Loop through ground truth and predicted images in order
    for ground_truth_file in ground_truth_files:
        # Extract the last number from the ground truth file name
        ground_truth_number = int(ground_truth_file.split("_")[-1].split(".")[0])

        # Search for the matching predicted file by comparing last numbers
        matching_predicted_file = None
        for predicted_file in output_files:
            predicted_number = int(predicted_file.split("_")[-1].split(".")[0])
            if ground_truth_number == predicted_number:
                matching_predicted_file = predicted_file
                break

        if matching_predicted_file:
            # Load ground truth and predicted images
            ground_truth = cv2.imread(os.path.join(ground_truth_folder, ground_truth_file), cv2.IMREAD_GRAYSCALE)
            predicted = cv2.imread(os.path.join(output_folder, matching_predicted_file), cv2.IMREAD_GRAYSCALE)

            # Calculate Dice coefficient
            intersection = np.sum(np.logical_and(ground_truth == 255, predicted == 255))
            total_pixels_gt = np.sum(ground_truth == 255)
            total_pixels_pred = np.sum(predicted == 255)

            dice_coefficient = 2.0 * intersection / (total_pixels_gt + total_pixels_pred)

            # Store the result in the dictionary
            image_pairs.append((ground_truth_file, matching_predicted_file))
            dice_coefficients[(ground_truth_file, matching_predicted_file)] = dice_coefficient
            print('dice coefficient:', dice_coefficient)
        else:
            print(f"No matching predicted file found for {ground_truth_file}")


# Specify the folder where you want to save the CSV file
csv_folder = '/home/jovyan/stephenie-storage/israt_files/SAM/Kmeans/sample_6/CSV_6_seg'

# Save Dice coefficients and image pairs to a CSV file in the specified folder
csv_filename = os.path.join(csv_folder, 'sample_6_k_means_seg_cluster_3.csv')

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Ground Truth Image', 'Predicted Image', 'Dice Coefficient'])
    for pair in image_pairs:
        dice = dice_coefficients.get(pair, 0.0)
        csv_writer.writerow([pair[0], pair[1], dice])

print("Dice coefficients saved to", csv_filename)
