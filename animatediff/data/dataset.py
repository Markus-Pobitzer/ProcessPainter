import os, io, csv, math, random
import numpy as np
from einops import rearrange
from decord import VideoReader, cpu

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from animatediff.utils.util import zero_rank_print
from typing import List



class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder, control_indexes: List[int] = [],
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
        ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        with open(csv_path, 'r') as csvfile:
            self.dataset = list(csv.DictReader(csvfile))
        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")

        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        self.control_indexes = control_indexes
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def og_get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
        
        video_dir    = os.path.join(self.video_folder, f"{videoid}.mp4")
        video_reader = VideoReader(video_dir)
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]
        
        return pixel_values, name
    
    def get_batch(self, idx):
        """
        Fetches and processes a video from the dataset.
        
        The video sampling logic (for `if not self.is_image`) has been
        rewritten to evenly sample `self.sample_n_frames` frames,
        always including the first and last frames.
        """
        video_dict = self.dataset[idx]
        videoid, name = video_dict['videoid'], video_dict['name']
        
        video_dir = os.path.join(self.video_folder, f"{videoid}.mp4")
        
        # Check if file exists before opening
        if not os.path.exists(video_dir):
            print(f"Warning: Video file not found {video_dir}")
            # Return empty tensors or handle as appropriate
            # Returning None for simplicity
            return None, None 

        try:
            video_reader = VideoReader(video_dir, ctx=cpu(0))
        except Exception as e:
            print(f"Error opening video {video_dir}: {e}")
            return None, None
            
        video_length = len(video_reader)
        
        if video_length == 0:
            print(f"Warning: Video has 0 length {video_dir}")
            del video_reader
            return None, None

        if not self.is_image:
            indices = np.linspace(
                0,                 # Start frame
                video_length - 1,  # End frame
                self.sample_n_frames, # Number of frames to sample
                dtype=int
            )
            batch_index = list(indices)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        batch_data = video_reader.get_batch(batch_index)
        
        # Check if data is numpy, if not, convert
        if not isinstance(batch_data, np.ndarray):
            try:
                # Try to call asnumpy() if it exists
                batch_data = batch_data.asnumpy()
            except AttributeError:
                raise TypeError("VideoReader.get_batch did not return a numpy array"
                                " and has no .asnumpy() method.")

        # (N_frames, H, W, C) -> (N_frames, C, H, W)
        pixel_values = torch.from_numpy(batch_data).permute(0, 3, 1, 2).contiguous()
        
        # Normalize to [0, 1]
        pixel_values = pixel_values / 255.
        
        del video_reader

        if self.is_image:
            # If it's an image, remove the frames dimension
            pixel_values = pixel_values[0]
        
        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break

            except Exception as e:
                idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name)
        return sample



if __name__ == "__main__":
    from animatediff.utils.util import save_videos_grid

    dataset = WebVid10M(
        csv_path="/mnt/petrelfs/guoyuwei/projects/datasets/webvid/results_2M_val.csv",
        video_folder="/mnt/petrelfs/guoyuwei/projects/datasets/webvid/2M_val",
        sample_size=256,
        sample_stride=4, sample_n_frames=16,
        is_image=True,
    )
    import pdb
    pdb.set_trace()
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)
    for idx, batch in enumerate(dataloader):
        print(batch["pixel_values"].shape, len(batch["text"]))
        # for i in range(batch["pixel_values"].shape[0]):
        #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)