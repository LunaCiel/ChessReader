import cv2
import albumentations as A
import os
import shutil

def blur_and_save(aug, image, output_path):
    image = aug(image=image)['image']  # Apply blur
    cv2.imwrite(output_path, image)  # Save new image

# Initialize blur augmentation
blur = A.Compose([A.Blur(p=1.0)])

# Initialize directories
data_dir = 'data'
blurry_data_dir = 'blurry_data'

# Create blurry_data_dir if not exist
os.makedirs(blurry_data_dir, exist_ok=True)

# Generate list of numbers for image files
image_numbers = list(range(1000))

# Iterate over the first 1000 image numbers
for num in image_numbers:
    image_name = f"{num}.png"
    text_name = f"{num}.txt"

    # Define paths
    image_path = os.path.join(data_dir, image_name)
    label_path = os.path.join(data_dir, text_name)
    blurry_image_path = os.path.join(blurry_data_dir, f"{num}_blurry.png")
    blurry_label_path = os.path.join(blurry_data_dir, text_name)

    # Load, blur, and save image
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is None:  # Check if image was loaded correctly
            print(f"Could not load image {image_path}. Skipping this image.")
        else:
            # Apply blur and save new image
            blur_and_save(blur, image, blurry_image_path)
            print(f"Processed and saved image {image_name}")
    else:
        print(f"Image file {image_path} does not exist. Skipping this image.")

    # Copy label file
    if os.path.exists(label_path):
        shutil.copy(label_path, blurry_label_path)
        print(f"Copied label file {text_name}")
    else:
        print(f"Label file {label_path} does not exist. Skipping this label.")