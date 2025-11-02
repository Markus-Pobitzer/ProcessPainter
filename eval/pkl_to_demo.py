"""Converts a .pckl dataset to a driectory with images."""

import os
import argparse
from PIL import Image

def process_images(input_directory, split, output_directory, controlnet_image_index: int = 7):
    input_path = os.path.join(input_directory, split)
    output_path = os.path.join(output_directory, split)

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Iterate over all subdirectories in the input path
    saved_paths = []
    for source in os.listdir(input_path):
        source_path = os.path.join(input_path, source)
        if not os.path.isdir(source_path):
            continue
        for subdir in os.listdir(source_path):
            subdir_path = os.path.join(source_path, subdir)
            if not os.path.isdir(subdir_path):
                continue

            image_path = os.path.join(subdir_path, "reference_frame.png")
            try:
                img = Image.open(image_path)
                output_image_path = os.path.join(output_path, source, subdir)
                os.makedirs(output_image_path, exist_ok=True)
                output_image_path = os.path.join(output_image_path, f"{controlnet_image_index}.jpg")
                img.save(output_image_path)
                saved_paths.append(output_image_path)
            except Exception as e:
                print(f"Failed to load image from {image_path}: {e}")
    print(f"Saved {len(saved_paths)} number of images to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Process reference_frame.png images from subdirectories.")
    parser.add_argument("input_directory", type=str, help="Path to the input directory")
    parser.add_argument("output_directory", type=str, help="Path to the output directory")
    parser.add_argument("--split", type=str, default="test", help="Split name (default: 'test')")
    parser.add_argument("--controlnet_image_index", type=int, default=7, help="controlnet_image_index for num frame")

    args = parser.parse_args()
    process_images(args.input_directory, args.split, args.output_directory, args.controlnet_image_index)

if __name__ == "__main__":
    main()