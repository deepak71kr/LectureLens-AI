import cv2
import os
import glob
import numpy as np
# import tkinter as tk
# from tkinter import filedialog, simpledialog

def create_composite_image(input_folder, output_path):
    """
    Creates a single composite image from a series of transparent PNGs.
    The composite image is a "union" of all images, filling transparent areas
    with valid pixel data from other images.
    """
    if not os.path.exists(input_folder):
        print(f"Error: The folder '{input_folder}' was not found.")
        return

    # Get a list of all PNG files in the folder, sorted to maintain order
    image_files = sorted(glob.glob(os.path.join(input_folder, "*.png")))
    if not image_files:
        print(f"No PNG images found in '{input_folder}'. Exiting.")
        return

    print(f"Found {len(image_files)} images to process...")

    # Read the first image to get dimensions and create the initial composite
    first_image_path = image_files[0]
    print(f"Reading first image: {first_image_path}")
    first_image = cv2.imread(first_image_path, cv2.IMREAD_UNCHANGED)
    if first_image is None or first_image.shape[2] != 4:
        print("Error: First image is not a valid transparent PNG (BGRA format).")
        return

    height, width, _ = first_image.shape
    
    # Initialize the final composite image as a copy of the first one
    composite_image = first_image.copy()

    # Process remaining images
    for i, file_path in enumerate(image_files[1:]):
        print(f"Processing image {i + 2}/{len(image_files)}: {os.path.basename(file_path)}")
        current_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        if current_image is None or current_image.shape[2] != 4:
            print(f"Warning: Skipping file '{os.path.basename(file_path)}' as it's not a valid transparent PNG.")
            continue
        
        # Ensure all images have the same dimensions
        if current_image.shape[0] != height or current_image.shape[1] != width:
            print(f"Warning: Skipping file '{os.path.basename(file_path)}' due to mismatched dimensions.")
            continue

        # Get the alpha channel from the current image
        alpha_channel = current_image[:, :, 3]

        # Find the pixels in the composite image that are transparent (alpha = 0)
        transparent_mask = (composite_image[:, :, 3] == 0)

        # Get the non-transparent pixels from the current image
        valid_pixels_mask = (alpha_channel != 0)

        # Combine the masks to find where the composite is transparent AND the current image has valid data
        fill_mask = transparent_mask & valid_pixels_mask

        # Use the fill_mask to update the composite image with valid pixels from the current image
        composite_image[fill_mask] = current_image[fill_mask]
    
    # Convert BGRA to BGR to save as a standard image format (e.g., JPEG)
    # The remaining transparent parts will become black
    # To save the transparency, you must save it as a PNG
    cv2.imwrite(output_path, composite_image)
    print("\n" + "="*50)
    print(f"✅ Composite image created and saved to '{output_path}'")
    print("="*50)

def main():
    print("🎨 Image Compositor: Create a Union from Transparent Images")
    try:
        root = tk.Tk()
        root.withdraw()

        # Ask user to select the input folder
        input_folder = filedialog.askdirectory(
            title="Select the folder containing transparent PNG frames"
        )
        if not input_folder:
            print("No folder selected. Exiting.")
            return

        # Get the output file path from the user
        output_file_name = simpledialog.askstring(
            "Output File", 
            "Enter the name for the composite image (e.g., composite_image.png):",
            initialvalue="composite_image.png"
        )
        if not output_file_name:
            print("No output file name provided. Exiting.")
            return

        output_path = os.path.join(input_folder, output_file_name)
    finally:
        if 'root' in locals():
            root.destroy()
    
    create_composite_image(input_folder, output_path)

if __name__ == "__main__":
    main()