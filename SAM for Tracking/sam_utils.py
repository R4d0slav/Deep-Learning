#!/usr/bin/python3

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset

import os
import glob
import tqdm
from functools import partial

from segment_anything.segment_anything.utils.transforms import ResizeLongestSide
from segment_anything.segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.segment_anything.modeling.prompt_encoder import PromptEncoder
from segment_anything.segment_anything.modeling.image_encoder import ImageEncoderViT
from segment_anything.segment_anything.modeling.transformer import TwoWayTransformer
from segment_anything.segment_anything.modeling.sam import Sam

PATH = "/d/hpc/projects/FRI/DL/ra9902/"

@torch.no_grad()
def preprocess(image_encoder, x: torch.Tensor):
    """Normalize pixel values and pad to a square input."""
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).cuda()
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).cuda()
    # Normalize colors
    x = (x - pixel_mean) / pixel_std

    # Pad
    h, w = x.shape[-2:]
    padh = image_encoder.img_size - h
    padw = image_encoder.img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

@torch.no_grad()
def encode(image_encoder, transform, image):
    input_image = transform.apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device='cuda')
    input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    # Transfer the image tensor to the GPU
    input_image_torch = input_image_torch.cuda()

    # Perform any necessary preprocessing on the GPU
    input_image = preprocess(image_encoder, input_image_torch)

    # Process the tensor using the GPU-accelerated model
    embeddings = image_encoder(input_image)

    torch.cuda.empty_cache()  # Free up GPU memory

    return embeddings, input_image

def postprocess_masks(masks, input_size, original_size, image_encoder):
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (image_encoder.img_size, image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks



class SegmentationDataset(Dataset):

    def __init__(self, train=True):
        dataset_path = PATH+"data/2019/" + ("train" if train else "test")
        subdirectories = sorted(glob.glob(dataset_path + "/JPEGImages/*"))

        self.images, self.templates = [], []
        self.masks, self.template_masks = [], []

        for subdirectory in subdirectories:
            image_paths = sorted(glob.glob(subdirectory + "/*.jpg"))

            for i in range(1, len(image_paths)):
                self.images.append(image_paths[i])
                self.templates.append(image_paths[i - 1])
                self.masks.append(image_paths[i].replace("JPEGImages", "Annotations").replace(".jpg", ".png"))
                self.template_masks.append(image_paths[i - 1].replace("JPEGImages", "Annotations").replace(".jpg", ".png"))

        self.resize_shape = (320, 416)

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path, template_path, template_mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)[:,:,2]
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        template_mask = cv2.imread(template_mask_path, cv2.IMREAD_COLOR)[:,:,2]

        channels=3
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))
        template = cv2.resize(template, dsize=(self.resize_shape[1], self.resize_shape[0]))
        template_mask = cv2.resize(template_mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = np.array(image).reshape((image.shape[0], image.shape[1], channels)).astype(np.float32) / 255.0
        mask = np.array(mask).reshape((image.shape[0], image.shape[1], 1))
        template = np.array(template).reshape((template.shape[0], template.shape[1], channels)).astype(np.float32) / 255.0
        template_mask = np.array(template_mask).reshape((template.shape[0], template.shape[1], 1))
 
        # Choose one instance
        instances, instance_count = np.unique(mask, return_counts = True)
        if len(instances) > 1:
            instances, instance_count = instances[1:], instance_count[1:]
            random_instance = instances[instance_count == max(instance_count)]

            mask = mask == random_instance
            template_mask = template_mask == random_instance

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        template = np.transpose(template, (2, 0, 1))
        template_mask = np.transpose(template_mask, (2, 0, 1))
        return image, mask.astype(np.float32), template, template_mask.astype(np.float32)

    def __getitem__(self, idx):
        image, mask, template, template_mask = self.transform_image(self.images[idx], self.masks[idx], self.templates[idx], self.template_masks[idx])
        sample = {'image': image, "mask": mask, 'idx': idx, 'template': template, 'template_mask': template_mask}

        return sample




