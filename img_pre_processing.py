import os
import cv2
import numpy as np
import pandas as pd

# Paths and configuration
RAW_IMAGE_PATH = '../data/raw_test/'
PATCHES_PATH = '../data/patches/'
CSV_OUTPUT_PATH = '../data/patch_metadata.csv'
IMAGE_SIZE = (256, 256)  # Standard size for resizing images
PATCH_SIZE = (64, 64)    # Size of each patch
METADATA_PATH = '../data/csv/two_class_metadata.csv'

# Load and resize images
def load_images(image_path, target_size):
    image_files = [f for f in os.listdir(image_path) if f.endswith(('.jpg', '.png'))]
    images = []
    for file in image_files:
        image = cv2.imread(os.path.join(image_path, file))
        if image is None:
            continue
        resized_image = cv2.resize(image, target_size)
        images.append(resized_image)
    return images, image_files

# Segment the resulting image into patches
def segment_into_patches(image, patch_size):
    h, w, _ = image.shape
    patches = []
    for i in range(0, h, patch_size[0]):
        for j in range(0, w, patch_size[1]):
            patch = image[i:i+patch_size[0], j:j+patch_size[1], :]
            patches.append(((i, j), patch))  # Store patch with its top-left corner coordinates
    return patches

# Tag each patch with a sequence ID and coordinates
def tag_patches(patches):
    tagged_patches = []
    for idx, (coords, patch) in enumerate(patches):
        tagged_patches.append((idx, coords, patch))
    return tagged_patches

# Embed lesion metadata with coordinates
def embed_metadata(tagged_patches, metadata, file_name):
    data_entries = []
    base_file_name = os.path.splitext(file_name)[0]  # Remove file extension
    for tag, (x, y), patch in tagged_patches:
        entry = {
            'patch_id': f"{base_file_name}_{tag}",
            'patch_location': f"({x},{y})",  # Include coordinates
            'file_name': file_name,
            **metadata
        }
        data_entries.append(entry)
    return data_entries

# Load metadata
def load_metadata(metadata_path):
    metadata = pd.read_csv(metadata_path)
    return metadata

# Match metadata with image
def get_metadata_for_image(metadata, image_id):
    metadata_row = metadata[metadata['image_id'] == image_id]
    if not metadata_row.empty:
        return metadata_row.to_dict('records')[0]
    else:
        return {}

# Main processing function
def process_images_and_create_csv():
    images, image_files = load_images(RAW_IMAGE_PATH, IMAGE_SIZE)
    metadata = load_metadata(METADATA_PATH)
    patch_metadata = []

    for i, (image, file_name) in enumerate(zip(images, image_files)):
        print(f"Processing image {i+1}/{len(images)}: {file_name}")

        # Extract image ID from the file name
        image_id = os.path.splitext(file_name)[0]

        # Segment the image into patches
        patches = segment_into_patches(image, PATCH_SIZE)

        # Tag each patch with a sequence ID
        tagged_patches = tag_patches(patches)

        # Embed metadata
        image_metadata = get_metadata_for_image(metadata, image_id)
        metadata_entries = embed_metadata(tagged_patches, image_metadata, file_name)

        # Save patches and collect metadata
        for entry in metadata_entries:
            patch_id = entry['patch_id']
            patch_index = int(patch_id.split('_')[-1])  # Extract patch index

            # Correctly unpack three values: index, coords, patch
            _, coords, patch = tagged_patches[patch_index]

            # Save the patch as RGB image
            patch_file = os.path.join(PATCHES_PATH, f"{patch_id}.png")
            cv2.imwrite(patch_file, cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))  # Save as RGB
            patch_metadata.append(entry)

    # Save the patch metadata to CSV
    df = pd.DataFrame(patch_metadata)
    df.to_csv(CSV_OUTPUT_PATH, index=False)

    print(f"All images processed and metadata saved to {CSV_OUTPUT_PATH}.")

# Execute the processing and create CSV
process_images_and_create_csv()
