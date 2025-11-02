"""Converts a .pckl dataset to a driectory with images."""

import os
import argparse
from pathlib import Path
from typing import List, Dict, Any
import shutil
from PIL import Image
from wd_tagger import Predictor
from omegaconf import OmegaConf
from scripts.animate import main as animate_main

def process_images(input_path, output_path, controlnet_image_index: int = 7) -> List[Dict[str, Any]]:
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
                img = Image.open(image_path)
                output_image_dir = os.path.join(output_path, source, subdir)
                os.makedirs(output_image_dir, exist_ok=True)
                output_image_path = os.path.join(output_image_dir, f"{controlnet_image_index}.jpg")
                img.save(output_image_path)
                ret_list.append(
                    {
                        "reference_frame_path": image_path,
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
    
def run_eval(args):
    input_directory=args.input_directory
    output_directory=args.output_directory
    temp_dir=args.temp_dir
    split=args.split
    controlnet_image_index=args.controlnet_image_index
    
    input_path = Path(input_directory, split)
    tmp_dataset_path = Path(temp_dir, split, "dataset")
    tmp_config_path = Path(temp_dir, split, "config.yaml")
    # Clean it just to make sure
    shutil.rmtree(tmp_dataset_path)
    tmp_dataset_path.mkdir(parents=True, exist_ok=True)
    frame_list_dict = process_images(str(input_path), str(tmp_dataset_path), controlnet_image_index)
    tagger = Predictor()

    # Add tags:
    for frame_dict in frame_list_dict:
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
    config["controlnet_images"] = new_controlnet_images
    config["prompt"] = new_prompts

    # Save the updated config and overwrite args
    OmegaConf.save(config, tmp_config_path)
    print(f"Updated config saved to {tmp_config_path}")
    args.config = str(tmp_config_path)
    
    # Actual inference
    # It saves it under "samples/*/" therefore clean the directory first
    shutil.rmtree("samples")
    animate_main(args=args)
    
    
    

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