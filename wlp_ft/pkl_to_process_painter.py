from PIL import Image
import os
import pickle
import csv
import cv2
import numpy as np
import argparse
from io import BytesIO

def create_video_dataset(dataset_path, output_path, split="train", sample_num_frames=8):
    """
    Converts a dataset of pickled frames into .mp4 videos and a metadata .csv file.
    Samples at most max_frames frames evenly for each video.

    Args:
        dataset_path (str): Path to the root dataset directory.
        output_path (str): Path to the directory where videos and .csv will be saved.
        split (str, optional): The dataset split to process (e.g., "test", "train"). Defaults to "test".
        fps (int, optional): Frames per second for the output video. Defaults to 10.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, split)
    os.makedirs(output_path, exist_ok=True)
    input_path = os.path.join(dataset_path, split)
    
    # Define CSV file path
    csv_path = os.path.join(output_path, f"wlp_{split}.csv")
    
    saved_list = []
    
    # Open the CSV file to write metadata
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        
        # Write the CSV header
        csv_writer.writerow(['videoid', 'name', 'page_dir'])

        # Iterate over all subdirectories in the input path
        print(f"Scanning directories in: {input_path}")
        for source in os.listdir(input_path):
            source_path = os.path.join(input_path, source)
            if not os.path.isdir(source_path):
                continue

            print(f"Processing source: {source}")
            for subdir in os.listdir(source_path):
                subdir_path = os.path.join(source_path, subdir)
                if not os.path.isdir(subdir_path):
                    continue

                video_id_path = os.path.join(output_path, subdir)
                os.makedirs(video_id_path, exist_ok=True)
                try:
                    # --- Load frame_data.pkl ---
                    frame_pkl_path = os.path.join(subdir_path, 'frame_data.pkl')
                    with open(frame_pkl_path, 'rb') as f:
                        frame_list = [Image.open(BytesIO(img_bytes)) for img_bytes in pickle.load(f)]

                    num_frames = len(frame_list)
                    if num_frames > (sample_num_frames - 1):
                        # Calculate evenly spaced indices
                        indices = np.linspace(0, num_frames - 1, num=(sample_num_frames - 1), dtype=int)
                        print(indices)
                        sampled_list = [frame_list[i] for i in indices]
                    else:
                        sampled_list = frame_list
                    # Last frame is equal to reference frame
                    print(len(sampled_list))
                    sampled_list += [sampled_list[-1]]
                    print(len(sampled_list))

                    # --- Load reference_frame_tags.pkl ---
                    prompt_pkl_path = os.path.join(subdir_path, 'reference_frame_tags.pkl')
                    with open(prompt_pkl_path, 'rb') as f:
                        prompt_list = pickle.load(f) # List[str]

                    # Ensure we have data (checking the sampled list)
                    if not sampled_list:
                        print(f"Skipping {subdir}: No frames found.")
                        continue
                    if not prompt_list:
                        print(f"Skipping {subdir}: No prompt (tags) found.")
                        continue

                    # --- Save sampled_list as .jpg ---
                    for idx, img in enumerate(sampled_list):
                        img.save(os.path.join(video_id_path, f"{idx}.jpg"))

                    # --- Write metadata to .csv ---
                    video_id = subdir
                    prompt_name = prompt_list[0] 
                    page_dir = ""                

                    csv_writer.writerow([video_id, prompt_name, page_dir])
                    
                    saved_list.append(subdir)

                except Exception as e:
                    print(f"Failed to load and store video from {subdir_path}: {e}")

    print(f"Saved {len(saved_list)} videos to {output_path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process reference_frame.png images from subdirectories.")
    parser.add_argument("--dataset_directory", type=str, help="Path to the input directory, i.e. dataset", required=True)
    parser.add_argument("--output_directory", type=str, help="Path to the output directory", required=True)

    args = parser.parse_args()
    
    print("Processing train split...")
    create_video_dataset(args.dataset_directory, args.output_directory, split="train")
    
    print("Dataset creation complete.")