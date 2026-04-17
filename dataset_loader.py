from pathlib import Path

import os
import h5py
import json
import imageio.v3 as io
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch import Tensor
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from utils.utils_3D import rotate_image_and_camera_z_axis
except:
    import sys

    sys.path.append("/home/mattia/Desktop/Repos/sandesc")
    from utils.utils_3D import rotate_image_and_camera_z_axis

# Define primary and fallback dataset paths
DATASET_PATH = Path("data")  # local


def rescale_and_pad(
    img: Tensor, depth: Tensor, K: Tensor, img_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Scale the longest side to fit the image, and pad the other
    Args:
        img: the input image tensor
            C,H,W
        depth: the input depth map
            H,W
        K: the camera intrinsics matrix
            3,3
        img_size: the output image size

    Returns:
        img: the rescaled and padded image
            C,H,W
        depth: the rescaled and padded depth map
            H,W
        K: the adapted intrinsics matrix
            3,3
    """
    H, W = img.shape[-2:]
    scale_factor = min(img_size / H, img_size / W)

    new_shape = (int(np.round(scale_factor * H)), int(np.round(scale_factor * W)))

    #  rescale
    img = F.interpolate(
        img[None],
        size=new_shape,
        mode="bilinear",
        align_corners=False,
    )[0]
    depth = F.interpolate(depth[None, None], size=new_shape, mode="nearest")[0, 0]
    K[:2, :] *= scale_factor

    #  pad
    pad = (img_size - new_shape[1], img_size - new_shape[0])  # x,y
    img = F.pad(img, (0, pad[0], 0, pad[1]), mode="constant", value=0.0)
    depth = F.pad(depth, (0, pad[0], 0, pad[1]), mode="constant", value=float("nan"))
    return img, depth, K


def rescale_and_center_crop(
    img: Tensor, depth: Tensor, K: Tensor, img_size: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Scale such that the shortest side fit the image, and crop the other
    Args:
        img: the input image tensor
            C,H,W
        depth: the input depth map
            H,W
        K: the camera intrinsics matrix
            3,3
        img_size: the output image size

    Returns:
        img: the rescaled and cropped image
            C,H,W
        depth: the rescaled and cropped depth map
            H,W
        K: the adapted intrinsics matrix
            3,3
    """
    H, W = img.shape[-2:]
    scale_factor = max(img_size / H, img_size / W)

    new_shape = (int(np.round(scale_factor * H)), int(np.round(scale_factor * W)))

    #  rescale
    img = F.interpolate(
        img[None],
        size=new_shape,
        mode="bilinear",
        align_corners=False,
    )[0]
    depth = F.interpolate(depth[None, None], size=new_shape, mode="nearest")[0, 0]
    K[:2, :] *= scale_factor

    #  center crop
    # Calculate offsets for center crop
    y_offset = (img.shape[-2] - img_size) // 2
    x_offset = (img.shape[-1] - img_size) // 2
    img = img[:, y_offset : y_offset + img_size, x_offset : x_offset + img_size]
    depth = depth[y_offset : y_offset + img_size, x_offset : x_offset + img_size]
    # Update principal point
    K[0, 2] -= x_offset
    K[1, 2] -= y_offset
    return img, depth, K


class TerraSkyDataset(Dataset):
    def __init__(
        self,
        img_size: int = 512,
        rescale_mode: str = "crop",
        random_rotation_degrees_fn: callable = None,
        transform: transforms.Compose = transforms.Compose([]),
        only_mixed: bool = False,  # not used
        verbose: bool = False,
    ):
        """
        Args:
            img_size: the output image size
            rescale_mode: how to rescale the images, either crop or pad
            random_rotation_degrees_fn: a function that returns a random rotation angle in degrees
            transform: the transformation to apply to the images
        """
        assert rescale_mode in [
            "crop",
            "pad",
        ], "rescale_mode must be either crop or pad"

        self.img_size = img_size
        self.rescale_mode = rescale_mode
        self.verbose = verbose
        self.random_rotation_degrees_fn = random_rotation_degrees_fn
        self.transform = transforms.Compose(
            [t for t in transform.transforms if not isinstance(t, transforms.ToTensor)]
        )
        scenes = sorted(os.listdir(DATASET_PATH))

        with open(DATASET_PATH.parent / "train_data.json") as f:
            self.scenes = json.load(f)

        self.flattened_pairs = []
        bar = tqdm(scenes, desc="Loading scenes and pairs")
        for scene in bar:
            try:
                # this is way to use pairs, but one can use whatever as long as they are meaninguful
                df = pd.read_csv(
                    DATASET_PATH
                    / scene
                    / "cyclic_depth_filtering_results1600_bidirectionally_filtered_square.csv",
                    index_col=None,
                )
                # filter df by num_pixels > 3000 and th 10 > 0.5
                df = df[df["num_pixels"] > 3000]
                df = df[df["10px"] > 0.5]

            except FileNotFoundError:
                # this should not happen, but just in case, we skip the scene if the consistency check results are not found
                if self.verbose:
                    print(
                        f"Consistency check results not found for scene {scene}, skipping..."
                    )
                continue
            # ... here there might some filtering of the pairs based on the consistency check results
            pairs = df[["level_0", "level_1"]].values.tolist()

            if only_mixed:
                pairs = [
                    (img0, img1)
                    for img0, img1 in pairs
                    if ("aerial" in img0 and "aerial" not in img1)
                    or ("aerial" not in img0 and "aerial" in img1)
                ]

            if len(pairs) > 0:
                self.scenes[scene]["pairs"] = pairs

                for img0, img1 in pairs:
                    self.flattened_pairs.append((scene, (img0, img1)))

            else:
                # pop the scene if it has no valid pairs
                self.scenes.pop(scene)

        # count how many pair ahave "aerial" in one of the two images, or in both
        if self.verbose:
            # print total scenes
            print(f"Total scenes: {len(self.scenes):>10,}")
            # print total images
            images = set()
            for pair in self.flattened_pairs:
                scene, (img1, img2) = pair
                images.add(img1)
                images.add(img2)
            print(f"Total images: {len(images):>10,}")
            mixed_count = sum(
                1
                for scene_name, (img0, img1) in self.flattened_pairs
                if ("aerial" in img0 and "aerial" not in img1)
                or ("aerial" not in img0 and "aerial" in img1)
            )
            aerial_count = sum(
                1
                for scene_name, (img0, img1) in self.flattened_pairs
                if "aerial" in img0 and "aerial" in img1
            )
            ground_count = sum(
                1
                for scene_name, (img0, img1) in self.flattened_pairs
                if "aerial" not in img0 and "aerial" not in img1
            )
            total_pairs = len(self.flattened_pairs)
            print(
                f"Mixed images:  {mixed_count:>10,} ({mixed_count/total_pairs:>7.2%})",
                f"Aerial images: {aerial_count:>10,} ({aerial_count/total_pairs:>7.2%})",
                f"Ground images: {ground_count:>10,} ({ground_count/total_pairs:>7.2%})",
                "-" * 40,
                f"Total pairs:   {total_pairs:>10,}",
                sep="\n",
            )

    def __len__(self):
        #  here we should have 10000, but we keep some margin because we might have some invalid images
        return len(self.flattened_pairs)

    def __getitem__(self, idx: int) -> dict:
        """get the pair of images
        Args:
            idx: the index of the pair

        Returns:
            output: a dictionary containing the pair of images, the depth maps, the camera intrinsics and extrinsics
        """

        while True:
            # drawing a random scene
            current_scene_name = list(self.scenes.keys())[idx % len(self.scenes)]
            current_scene = self.scenes[current_scene_name]
            base_path = DATASET_PATH / current_scene_name

            # drawing a pair of images from the current scene
            pairs = current_scene["pairs"]
            if pairs is None or len(pairs) == 0:
                if self.verbose:
                    print(f"No valid pairs in scene {current_scene_name}, skipping...")
                continue
            pair_idx = np.random.randint(len(pairs))
            img0_name, img1_name = pairs[pair_idx]

            img0, depth0, K0, P0 = self.load_data(base_path, img0_name, current_scene)
            img1, depth1, K1, P1 = self.load_data(base_path, img1_name, current_scene)

            if img0 is None or img1 is None:
                if self.verbose:
                    print(
                        f"Skipping invalid pair {img0_name}, {img1_name} in scene {current_scene_name}"
                    )
                continue

            if self.random_rotation_degrees_fn is not None:
                angle1 = self.random_rotation_degrees_fn()
                img1, P1, K1, depth1 = rotate_image_and_camera_z_axis(
                    angle1, img1, P1, K1, depth1
                )

            if self.rescale_mode == "pad":
                img0, depth0, K0 = rescale_and_pad(img0, depth0, K0, self.img_size)
                img1, depth1, K1 = rescale_and_pad(img1, depth1, K1, self.img_size)
            else:
                img0, depth0, K0 = rescale_and_center_crop(
                    img0, depth0, K0, self.img_size
                )
                img1, depth1, K1 = rescale_and_center_crop(
                    img1, depth1, K1, self.img_size
                )

            depth0[depth0 == 0.0] = float("nan")
            depth1[depth1 == 0.0] = float("nan")

            output = {
                # "scene": current_scene_name,
                "img0": self.transform(img0),
                "img1": self.transform(img1),
                "depth0": depth0,
                "depth1": depth1,
                "K0": K0,
                "K1": K1,
                "P0": P0,
                "P1": P1,
            }

            return output

    def load_data(self, base_path, img, current_scene):
        """load the pair of images
        Args:
            idx: the index of the pair

        Returns:
            output: a dictionary containing the pair of images, the depth maps, the camera intrinsics and extrinsics
        """
        img_path = base_path / "frames" / img
        depth_path = (
            str(img_path).replace("frames", "depth/maps").replace(".jpg", ".h5")
        )

        img_rgb = torch.from_numpy(io.imread(img_path) / 255.0).permute(2, 0, 1).float()
        with h5py.File(depth_path, "r") as f:
            depth = torch.from_numpy(f["depth"][()])

        img_entry = current_scene["images"][img]
        P = torch.tensor(img_entry["P"])
        P = torch.cat([P, torch.tensor([[0.0, 0.0, 0.0, 1.0]])], dim=0)

        # forcing the principal point to be in the center of the image, as we will apply random crops and rotations
        # which might not be the best choice but it works for now
        K = torch.tensor(current_scene["cameras"][str(img_entry["K_id"])]["K"])
        K[0, 2] = img_rgb.shape[-1] // 2
        K[1, 2] = img_rgb.shape[-2] // 2

        return img_rgb.float(), depth.float(), K.float(), P.float()

    def reset(self):
        """reset the dataloader"""
        self.__init__(
            img_size=self.img_size,
            rescale_mode=self.rescale_mode,
            random_rotation_degrees_fn=self.random_rotation_degrees_fn,
            transform=self.transform,
        )
