"""Converts a .pckl dataset to a driectory with images."""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
import shutil
from PIL import Image
from eval.wd_tagger import Predictor
from omegaconf import OmegaConf
from scripts.animate import main as animate_main
from tqdm import tqdm
from eval.img_utils import pil_resize, undo_pil_resize
from io import BytesIO
import pickle
import glob


def process_images(input_path, output_path, controlnet_image_index: int = 7, width: int = 512, height: int = 512) -> List[Dict[str, Any]]:
    ret_list: List[Dict[str, Any]] = []

    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Iterate over all subdirectories in the input path
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
                img = Image.open(image_path).convert("RGB")
                original_size = img.size
                # Resize to correct size explicetly here
                img = pil_resize(img, target_size=(width, height), pad_input=True)
                output_image_dir = os.path.join(output_path, source, subdir)
                os.makedirs(output_image_dir, exist_ok=True)
                output_image_path = os.path.join(output_image_dir, f"{controlnet_image_index}.jpg")
                img.save(output_image_path)
                ret_list.append(
                    {
                        "reference_frame_path": image_path,
                        "reference_frame_size": original_size,
                        "tmp_frame_dir": output_image_dir,
                        "tmp_frame_path": output_image_path,
                        "source": source,
                        "id": subdir
                    }
                )
            except Exception as e:
                print(f"Failed to load image from {image_path}: {e}")
    print(f"Prepared {len(ret_list)} number of images for ProcessPainter inference, saved to {output_path}")

    return ret_list

def get_latest_sample_subdirs(base_path="samples"):
    # Find all subdirectories under samples/*/
    sample_dirs = sorted(glob.glob(os.path.join(base_path, "*/")))

    if not sample_dirs:
        print("No directories found under 'samples/*/'.")
        return []

    if len(sample_dirs) > 1:
        print("Multiple directories found under 'samples/*/':")
        for d in sample_dirs:
            print(f" - {d}")
        print("Selecting the last one (sorted).")

    # Select the last one
    selected_dir = sample_dirs[-1]

    # List all subdirectories under the selected directory
    subdirs = [
        os.path.join(selected_dir, name)
        for name in os.listdir(selected_dir)
        if os.path.isdir(os.path.join(selected_dir, name))
    ]

    return subdirs

def run_eval(args):
    input_directory=args.input_directory
    output_directory=args.output_directory
    temp_dir=args.temp_dir
    split=args.split
    controlnet_image_index=args.controlnet_image_index
    
    output_path = Path(output_directory, split)
    input_path = Path(input_directory, split)
    tmp_dataset_path = Path(temp_dir, split, "dataset")
    tmp_config_path = Path(temp_dir, split, "config.yaml")
    # Clean it just to make sure
    shutil.rmtree(tmp_dataset_path, ignore_errors=True)
    tmp_dataset_path.mkdir(parents=True, exist_ok=True)
    frame_list_dict = process_images(str(input_path), str(tmp_dataset_path), controlnet_image_index)
    tagger = Predictor()

    for frame_dict in tqdm(frame_list_dict, desc="tagging images"):
        img = Image.open(frame_dict["reference_frame_path"])
        tags, _, _, _ = tagger.predict(image=img)
        frame_dict["prompt"] = tags
        
    # Prepare config
    original_config_path = args.config
    config = OmegaConf.load(original_config_path)

    # Extract new values from frame_list_dict
    new_controlnet_images = [entry["tmp_frame_dir"] for entry in frame_list_dict]
    new_prompts = [entry["prompt"] for entry in frame_list_dict]

    # Update the config (assuming it's a list with one dict as in your YAML)
    config[0]["controlnet_images"] = new_controlnet_images
    config[0]["prompt"] = new_prompts

    # Save the updated config and overwrite args
    OmegaConf.save(config, tmp_config_path)
    print(f"Updated config saved to {tmp_config_path}")
    args.config = str(tmp_config_path)
    
    # Actual inference
    # It saves it under "samples/*/" therefore clean the directory first
    shutil.rmtree("samples", ignore_errors=True)
    animate_main(args=args)
    
    sample_dir_list = get_latest_sample_subdirs()
    processed = 0
    for sample_dir, frame_dict in zip(sample_dir_list, frame_list_dict):
        image_files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_files.sort()
        images: List[Image] = []
        for f in image_files:
            try:
                fp = os.path.join(sample_dir, f)
                gen_frame = Image.open(fp).convert("RGB")
                images.append(gen_frame)
            except Exception as e:
                print(f"Was not able to load frame from {fp}: {e}")
        if len(images) < 1:
            continue
        # Undo resizing and padding
        images = [undo_pil_resize(img, target_size=frame_dict["reference_frame_size"]) for img in images]
        
        # Save as pkl dataset
        source = frame_dict["source"]
        video_id = frame_dict["id"]
        
        image_data_list: List[bytes] = []
        for im in images:
            buffer = BytesIO()
            im.save(buffer, format="PNG")
            image_data_list.append(buffer.getvalue())
            
        n = len(image_data_list)
        # --- Build progress list [0.0 .. 1.0] inclusive ---
        if n == 1:
            progress_steps_list = [1.0]
        else:
            progress_steps_list = [i / (n - 1) for i in range(n)]
            
        video_out = output_path / source / video_id
        video_out.mkdir(parents=True, exist_ok=True)
        with open(video_out / "frame_data.pkl", "wb") as fp:
            pickle.dump(image_data_list, fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open(video_out / "frame_progress.pkl", "wb") as fp:
            pickle.dump(progress_steps_list, fp, protocol=pickle.HIGHEST_PROTOCOL)

        processed += 1
    print(f"Successfuly processed {processed} samples and saved them under {str(video_out)}.")
    

def main():
    parser = argparse.ArgumentParser(description="Process reference_frame.png images from subdirectories.")
    parser.add_argument("--input_directory", type=str, help="Path to the input directory, i.e. dataset", required=True)
    parser.add_argument("--output_directory", type=str, help="Path to the output directory", required=True)
    parser.add_argument("--temp_dir", type=str, default="temp_dir/eval", help="Path to temporary working directory")
    parser.add_argument("--split", type=str, default="test", help="Split name (default: 'test')")
    parser.add_argument("--controlnet_image_index", type=int, default=7, help="controlnet_image_index for num frame")
    # From Process Painter
    parser.add_argument("--pretrained-model-path", type=str, default="models/stable-diffusion-v1-5",)
    parser.add_argument("--inference-config",      type=str, default="configs/inference/inference-v1.yaml")    
    parser.add_argument("--config",                type=str, default="configs/prompts/speedpainting-cn-final.yaml", required=True)
    
    parser.add_argument("--L", type=int, default=8 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    parser.add_argument("--without-xformers", default=True)

    args = parser.parse_args()
    run_eval(args=args)

if __name__ == "__main__":
    main()