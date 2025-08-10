import os
from glob import glob

import cv2
import torch

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import h5py
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset


def load_ldr_image(image_path, from_srgb=False, clamp=False, normalize=False):
    # Load png or jpg image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image.astype(np.float32) / 255.0)  # (h, w, c)
    image[~torch.isfinite(image)] = 0
    if from_srgb:
        # Convert from sRGB to linear RGB
        image = image**2.2
    if clamp:
        image = torch.clamp(image, min=0.0, max=1.0)
    if normalize:
        # Normalize to [-1, 1]
        image = image * 2.0 - 1.0
        image = torch.nn.functional.normalize(image, dim=-1, eps=1e-6)
    return image.permute(2, 0, 1)  # returns (c, h, w)


def load_hdf5_image(image_path, tonemaping=False, clamp=False, normalize=False,lum_scale=None):
    # Load hdf5 image
    image = h5py.File(image_path, "r")
    image = torch.from_numpy(np.array(image["dataset"]).astype(np.float32))  # (h, w, c)
    image[~torch.isfinite(image)] = 0
    if tonemaping:
        # Exposure adjuestment and tonemapping
        image_Yxy = convert_rgb_2_Yxy(image)
        lum = (
            image[:, :, 0:1] * 0.2125
            + image[:, :, 1:2] * 0.7154
            + image[:, :, 2:3] * 0.0721
        )
        lum = torch.log(torch.clamp(lum, min=1e-4))
        if lum_scale is not None:
            lum_mean = lum_scale
        else:
            lum_mean = torch.exp(torch.mean(lum))
        lp = image_Yxy[:, :, 0:1] * 0.18 / torch.clamp(lum_mean, min=1e-4)
        # image_Yxy[:, :, 0:1] = reinhard(lp)
        # TODO: do not use reinhard curve for now
        image_Yxy[:, :, 0:1] = lp
        image = convert_Yxy_2_rgb(image_Yxy)
    if clamp:
        image = torch.clamp(image, min=0.0, max=1.0)
    if normalize:
        # Already in [-1, 1]
        image = torch.nn.functional.normalize(image, dim=-1, eps=1e-6)
    return image.permute(2, 0, 1)  # returns (c, h, w)


def load_exr_image(image_path, tonemaping=False, clamp=False, normalize=False):
    image = cv2.cvtColor(cv2.imread(image_path, -1), cv2.COLOR_BGR2RGB)
    image = torch.from_numpy(image.astype("float32"))  # (h, w, c)
    image[~torch.isfinite(image)] = 0
    if tonemaping:
        # Exposure adjuestment and tonemapping
        image_Yxy = convert_rgb_2_Yxy(image)
        lum = (
            image[:, :, 0:1] * 0.2125
            + image[:, :, 1:2] * 0.7154
            + image[:, :, 2:3] * 0.0721
        )
        lum = torch.log(torch.clamp(lum, min=1e-6))
        lum_mean = torch.exp(torch.mean(lum))
        lp = image_Yxy[:, :, 0:1] * 0.18 / torch.clamp(lum_mean, min=1e-6)
        # image_Yxy[:, :, 0:1] = reinhard(lp)
        # TODO: do not use reinhard curve for now
        image_Yxy[:, :, 0:1] = lp
        image = convert_Yxy_2_rgb(image_Yxy)
    if clamp:
        image = torch.clamp(image, min=0.0, max=1.0)
    if normalize:
        # Already in [-1, 1]
        image = torch.nn.functional.normalize(image, dim=-1, eps=1e-6)
    return image.permute(2, 0, 1)  # returns (c, h, w)

def luminance(image_path):
    # Load hdf5 image
    image = h5py.File(image_path, "r")
    image = torch.from_numpy(np.array(image["dataset"]).astype(np.float32))  # (h, w, c)
    image[~torch.isfinite(image)] = 0

    # Exposure adjuestment and tonemapping
    image_Yxy = convert_rgb_2_Yxy(image)
    lum = (
        image[:, :, 0:1] * 0.2125
        + image[:, :, 1:2] * 0.7154
        + image[:, :, 2:3] * 0.0721
    )
    lum = torch.log(torch.clamp(lum, min=1e-4))

    lum_mean = torch.exp(torch.mean(lum))
    return lum_mean


def convert_rgb_2_XYZ(rgb):
    # Reference: https://web.archive.org/web/20191027010220/http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
    # rgb: (h, w, 3)
    # XYZ: (h, w, 3)
    XYZ = torch.ones_like(rgb)
    XYZ[:, :, 0] = (
        0.4124564 * rgb[:, :, 0] + 0.3575761 * rgb[:, :, 1] + 0.1804375 * rgb[:, :, 2]
    )
    XYZ[:, :, 1] = (
        0.2126729 * rgb[:, :, 0] + 0.7151522 * rgb[:, :, 1] + 0.0721750 * rgb[:, :, 2]
    )
    XYZ[:, :, 2] = (
        0.0193339 * rgb[:, :, 0] + 0.1191920 * rgb[:, :, 1] + 0.9503041 * rgb[:, :, 2]
    )
    return XYZ


def convert_XYZ_2_Yxy(XYZ):
    # XYZ: (h, w, 3)
    # Yxy: (h, w, 3)
    Yxy = torch.ones_like(XYZ)
    Yxy[:, :, 0] = XYZ[:, :, 1]
    sum = torch.sum(XYZ, dim=2)
    inv_sum = 1.0 / torch.clamp(sum, min=1e-4)
    Yxy[:, :, 1] = XYZ[:, :, 0] * inv_sum
    Yxy[:, :, 2] = XYZ[:, :, 1] * inv_sum
    return Yxy


def convert_rgb_2_Yxy(rgb):
    # rgb: (h, w, 3)
    # Yxy: (h, w, 3)
    return convert_XYZ_2_Yxy(convert_rgb_2_XYZ(rgb))


def convert_XYZ_2_rgb(XYZ):
    # XYZ: (h, w, 3)
    # rgb: (h, w, 3)
    rgb = torch.ones_like(XYZ)
    rgb[:, :, 0] = (
        3.2404542 * XYZ[:, :, 0] - 1.5371385 * XYZ[:, :, 1] - 0.4985314 * XYZ[:, :, 2]
    )
    rgb[:, :, 1] = (
        -0.9692660 * XYZ[:, :, 0] + 1.8760108 * XYZ[:, :, 1] + 0.0415560 * XYZ[:, :, 2]
    )
    rgb[:, :, 2] = (
        0.0556434 * XYZ[:, :, 0] - 0.2040259 * XYZ[:, :, 1] + 1.0572252 * XYZ[:, :, 2]
    )
    return rgb


def convert_Yxy_2_XYZ(Yxy):
    # Yxy: (h, w, 3)
    # XYZ: (h, w, 3)
    XYZ = torch.ones_like(Yxy)
    XYZ[:, :, 0] = Yxy[:, :, 1] / torch.clamp(Yxy[:, :, 2], min=1e-6) * Yxy[:, :, 0]
    XYZ[:, :, 1] = Yxy[:, :, 0]
    XYZ[:, :, 2] = (
        (1.0 - Yxy[:, :, 1] - Yxy[:, :, 2])
        / torch.clamp(Yxy[:, :, 2], min=1e-4)
        * Yxy[:, :, 0]
    )
    return XYZ


def convert_Yxy_2_rgb(Yxy):
    # Yxy: (h, w, 3)
    # rgb: (h, w, 3)
    return convert_XYZ_2_rgb(convert_Yxy_2_XYZ(Yxy))


def reinhard(x):
    return x / (1.0 + x)


def tonemap_aces(x):
    # Reference: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    a = 2.51
    b = 0.03
    c = 2.43
    d = 0.59
    e = 0.14
    return (x * (a * x + b)) / (x * (c * x + d) + e)


class EverMotionDataset(Dataset):
    """
    Evermotion dataset
    """

    def __init__(
        self,
        dataset_dir,
        required_aovs=["albedo"],
        required_resolution=512,
        resize=False,
        center_crop=False,
        crop=True,
    ):
        super().__init__()
        self.aov_num_of_c = {
            "albedo": 3,
            "normal": 3,
            "roughness": 3,
            "metallic": 3,
        }
        self.required_aovs = required_aovs.copy()

        self.dataset_dir = dataset_dir

        # Always read normal
        if "normal" not in self.required_aovs:
            self.required_aovs.append("normal")

        # Read scene list
        self.scene_list = sorted(glob(os.path.join(self.dataset_dir, "*")))
        # Read photo list
        self.image_list = []
        for scene in self.scene_list:
            scene_image_list = sorted(glob(os.path.join(scene, "*[!a-z].exr")))
            self.image_list += scene_image_list

        # Read other AOVs list
        self.aov_list = {}
        for aov_name in self.required_aovs:
            if aov_name not in self.aov_num_of_c.keys():
                continue
            self.aov_list[aov_name] = [
                x.replace(".exr", f"_{aov_name}.exr") for x in self.image_list
            ]

        # Read prompts
        self.prompts_path = "/sensei-fs/users/zhengz/datasets/evermotion/prompts.txt"

        if self.prompts_path is not None:
            self.prompts = []
            with open(self.prompts_path, "r") as f:
                for line in f.readlines():
                    self.prompts.append(line.strip())

        # Meta info
        self.total_num_of_c = 0
        for aov_name in self.required_aovs:
            if aov_name in self.aov_list.keys():
                self.total_num_of_c += self.aov_num_of_c[aov_name]

        print(
            f"Loaded {len(self.image_list)} images from {self.dataset_dir}. Dataset will supply {self.total_num_of_c}: {self.aov_list.keys()}"
        )

        # Check if the files are valid
        for aov_name in self.required_aovs:
            if aov_name in self.aov_list.keys():
                for aov_path in self.aov_list[aov_name]:
                    assert os.path.exists(aov_path), f"{aov_path} does not exist."

        for image_path in self.image_list:
            assert os.path.exists(image_path), f"{image_path} does not exist."

        if self.prompts_path is not None:
            assert len(self.prompts) == len(
                self.image_list
            ), f"Number of prompts {len(self.prompts)} does not match number of images {len(self.image_list)}"

        # Augmentation
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(required_resolution, antialias=True)
                if resize
                else transforms.Lambda(lambda x: x),
                (
                    transforms.CenterCrop(required_resolution)
                    if center_crop
                    else transforms.RandomCrop(required_resolution)
                )
                if crop
                else transforms.Lambda(lambda x: x),
            ]
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        target_path = self.image_list[idx]
        target = load_exr_image(target_path, tonemaping=True, clamp=True)

        aovs = {}
        for aov_name in self.required_aovs:
            if aov_name == "albedo":
                aovs[aov_name] = load_exr_image(
                    self.aov_list[aov_name][idx], clamp=True
                )
            elif aov_name == "normal":
                aovs[aov_name] = load_exr_image(
                    self.aov_list[aov_name][idx], normalize=True
                )
            elif aov_name == "roughness":
                aovs[aov_name] = load_exr_image(
                    self.aov_list[aov_name][idx], clamp=True
                )
            elif aov_name == "metallic":
                aovs[aov_name] = load_exr_image(
                    self.aov_list[aov_name][idx], clamp=True
                )

        # Read normal anyway to get the mask
        normal_norm = torch.norm(aovs["normal"], dim=0, keepdim=True)
        mask = normal_norm > 1e-6
        for aov in aovs:
            aovs[aov] = aovs[aov] * mask
        target = target * mask

        # Transform
        transformed = self.train_transforms(
            torch.cat(list(aovs.values()) + [target], dim=0)
        )
        target = transformed[-3:]
        # TODO: after transform, the value sometimes exceeds 1.0
        target = torch.clamp(target, min=0.0, max=1.0)

        i = 0
        for aov_name in self.required_aovs:
            if aov_name in self.aov_num_of_c.keys():
                aovs[aov_name] = transformed[i : i + self.aov_num_of_c[aov_name]]
                # TODO: after transform, the value sometimes exceeds 1.0
                if aov_name == "normal":
                    aovs["normal"] = torch.nn.functional.normalize(
                        aovs["normal"], dim=0, eps=1e-6
                    )
                else:
                    aovs[aov_name] = torch.clamp(aovs[aov_name], min=0.0, max=1.0)
                i += self.aov_num_of_c[aov_name]

        # Fill the unsupported aovs with zero
        for aov_name in self.required_aovs:
            if aov_name not in self.aov_list.keys():
                aovs[aov_name] = torch.zeros_like(target)

        # Provide the valid or not information
        for aov_name in self.required_aovs:
            if aov_name in self.aov_list.keys():
                aovs[f"{aov_name}_valid"] = torch.tensor([True], dtype=torch.bool)
            else:
                aovs[f"{aov_name}_valid"] = torch.tensor([False], dtype=torch.bool)

        if self.prompts_path is not None:
            # Read prompt from file
            prompt = self.prompts[idx]
        else:
            prompt = ""

        return {
            "prompt": prompt,
            "target": target,
        } | aovs


class InteriorVerseDataset(Dataset):
    """
    InteriorVerse dataset
    See https://interiorverse.github.io/
    """

    def __init__(
        self,
        dataset_dir,
        sub_folder="dataset_120",
        phase="train",
        required_aovs=["albedo"],
        required_resolution=448,
        resize=False,
        center_crop=False,
        crop=True,
        denoised=False,
        return_raw=False,
    ):
        super().__init__()
        self.aov_num_of_c = {
            "albedo": 3,
            "normal": 3,
            "roughness": 3,
            "metallic": 3,
        }
        self.required_aovs = required_aovs.copy()
        self.return_raw = return_raw

        # Always read normal
        if "normal" not in self.required_aovs:
            self.required_aovs.append("normal")

        # Locate scene list file
        self.dataset_dir = os.path.join(dataset_dir, sub_folder)
        self.phase = phase
        if self.phase.upper() == "TRAIN":
            scene_file = os.path.join(self.dataset_dir, "train.txt")
        elif self.phase.upper() == "TEST":
            scene_file = os.path.join(self.dataset_dir, "test.txt")
        elif self.phase.upper() == "VAL":
            scene_file = os.path.join(self.dataset_dir, "val.txt")
        else:
            assert False, f"Invalid phase: {self.phase}"

        # Read scene list
        with open(scene_file, "r") as f:
            self.scene_list = f.readlines()
        self.scene_list = [x.strip() for x in self.scene_list]

        # Read image list
        self.image_list = []
        for scene in self.scene_list:
            self.image_list += sorted(
                glob(os.path.join(self.dataset_dir, scene, "*_im.exr"))
            )

        # Read other AOVs list
        self.aov_list = {}
        for aov_name in self.required_aovs:
            if aov_name == "roughness":
                self.aov_list[aov_name] = [
                    x.replace("_im", "_material") for x in self.image_list
                ]
            elif aov_name == "metallic":
                self.aov_list[aov_name] = [
                    x.replace("_im", "_material") for x in self.image_list
                ]
            if aov_name == "albedo" or aov_name == "normal":
                self.aov_list[aov_name] = [
                    x.replace("_im", f"_{aov_name}") for x in self.image_list
                ]
        self.aov_list["mask"] = [x.replace("_im", "_mask") for x in self.image_list]

        # If request the denoised image, replace the subfolder for image_list
        if denoised:
            self.image_list = [
                image_path.replace(sub_folder, f"{sub_folder}_denoised")
                for image_path in self.image_list
            ]

        # Read prompts
        if self.phase.upper() == "TRAIN":
            self.prompts_path = (
                "/sensei-fs/users/zhengz/datasets/interiorverse/prompts.txt"
            )
        else:
            self.prompts_path = (
                "/sensei-fs/users/zhengz/datasets/interiorverse/prompts_test.txt"
            )

        if self.prompts_path is not None:
            self.prompts = []
            with open(self.prompts_path, "r") as f:
                for line in f.readlines():
                    self.prompts.append(line.strip())

        # Meta info
        self.total_num_of_c = 0
        for aov_name in self.required_aovs:
            if aov_name in self.aov_list.keys():
                self.total_num_of_c += self.aov_num_of_c[aov_name]

        print(
            f"Loaded {len(self.image_list)} images from {self.dataset_dir}. Dataset will supply {self.total_num_of_c}: {self.aov_list.keys()}"
        )

        # Check if the files are valid
        for aov_name in self.required_aovs:
            if aov_name in self.aov_list.keys():
                for aov_path in self.aov_list[aov_name]:
                    assert os.path.exists(aov_path), f"{aov_path} does not exist."

        for image_path in self.image_list:
            assert os.path.exists(image_path), f"{image_path} does not exist."

        if self.prompts_path is not None:
            assert len(self.prompts) == len(
                self.image_list
            ), f"Number of prompts {len(self.prompts)} does not match number of images {len(self.image_list)}"

        # Augmentation
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(required_resolution, antialias=True)
                if resize
                else transforms.Lambda(lambda x: x),
                (
                    transforms.CenterCrop(required_resolution)
                    if center_crop
                    else transforms.RandomCrop(required_resolution)
                )
                if crop
                else transforms.Lambda(lambda x: x),
            ]
        )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        if self.return_raw:  # for denoising purpose
            target_path = self.image_list[idx]
            target = load_exr_image(target_path)

            aovs = {}
            for aov_name in self.required_aovs:
                if aov_name == "albedo":
                    aov = load_exr_image(self.aov_list[aov_name][idx])
                if aov_name == "normal":
                    aov = load_exr_image(self.aov_list[aov_name][idx])
                aovs[aov_name] = aov

            return {
                "target": target,
                "target_path": target_path,
            } | aovs

        target_path = self.image_list[idx]
        target = load_exr_image(target_path, tonemaping=True, clamp=True)

        aovs = {}
        for aov_name in self.required_aovs:
            if aov_name == "albedo":
                aovs[aov_name] = load_exr_image(
                    self.aov_list[aov_name][idx], clamp=True
                )
            elif aov_name == "normal":
                aovs[aov_name] = load_exr_image(
                    self.aov_list[aov_name][idx], normalize=True
                )
            elif aov_name == "roughness":
                aov = load_exr_image(self.aov_list[aov_name][idx], clamp=True)
                aovs[aov_name] = torch.cat([aov[0:1]] * 3, dim=0)
            elif aov_name == "metallic":
                aov = load_exr_image(self.aov_list[aov_name][idx], clamp=True)
                aovs[aov_name] = torch.cat([aov[1:2]] * 3, dim=0)

        # Mask out
        mask = load_exr_image(self.aov_list["mask"][idx], clamp=True)
        for aov in aovs:
            aovs[aov] = aovs[aov] * mask
        target = target * mask

        # Transform
        transformed = self.train_transforms(
            torch.cat(list(aovs.values()) + [target], dim=0)
        )
        target = transformed[-3:]
        # TODO: after transform, the value sometimes exceeds 1.0
        target = torch.clamp(target, min=0.0, max=1.0)

        i = 0
        for aov_name in self.required_aovs:
            if aov_name in self.aov_num_of_c.keys():
                aovs[aov_name] = transformed[i : i + self.aov_num_of_c[aov_name]]
                # TODO: after transform, the value sometimes exceeds 1.0
                if aov_name == "normal":
                    aovs["normal"] = torch.nn.functional.normalize(
                        aovs["normal"], dim=0, eps=1e-6
                    )
                else:
                    aovs[aov_name] = torch.clamp(aovs[aov_name], min=0.0, max=1.0)
                i += self.aov_num_of_c[aov_name]

        # Fill the unsupported aovs with zero
        for aov_name in self.required_aovs:
            if aov_name not in self.aov_list.keys():
                aovs[aov_name] = torch.zeros_like(target)

        # Provide the valid or not information
        for aov_name in self.required_aovs:
            if aov_name in self.aov_list.keys():
                aovs[f"{aov_name}_valid"] = torch.tensor([True], dtype=torch.bool)
            else:
                aovs[f"{aov_name}_valid"] = torch.tensor([False], dtype=torch.bool)

        # The roughness and metallic are not reliable, so we mark them invalid here
        aovs["roughness_valid"] = torch.tensor([False], dtype=torch.bool)
        aovs["metallic_valid"] = torch.tensor([False], dtype=torch.bool)

        if self.prompts_path is not None:
            # Read prompt from file
            prompt = self.prompts[idx]
        else:
            prompt = ""

        return {
            "prompt": prompt,
            "target": target,
        } | aovs


class StockImagesDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        phase="train",
        required_aovs=["albedo", "normal"],
        required_resolution=512,
        resize=False,
        crop=True,
        center_crop=False,
    ):
        super().__init__()
        self.aov_num_of_c = {
            "albedo": 3,
            "normal": 3,
            "roughness": 3,
            "metallic": 3,
        }
        self.required_aovs = required_aovs.copy()

        # Always read normal
        if "normal" not in self.required_aovs:
            self.required_aovs.append("normal")

        self.dataset_dir = dataset_dir

        # Read photo list
        self.photo_list = sorted(
            glob(os.path.join(self.dataset_dir, "photo", "*-photo.png"))
        )

        # Read other AOVs list
        self.aov_list = {}
        for aov_name in self.required_aovs:
            if aov_name == "normal":
                self.aov_list[aov_name] = [
                    x.replace("/photo/", "/normal/").replace("-photo", "-normal")
                    for x in self.photo_list
                ]
            elif aov_name == "albedo":
                self.aov_list[aov_name] = [
                    x.replace("/photo/", "/mat/").replace("-photo", "-albedo")
                    for x in self.photo_list
                ]
            elif aov_name == "roughness":
                self.aov_list[aov_name] = [
                    x.replace("/photo/", "/mat/").replace("-photo", "-roughness")
                    for x in self.photo_list
                ]
            elif aov_name == "metallic":
                self.aov_list[aov_name] = [
                    x.replace("/photo/", "/mat/").replace("-photo", "-metallic")
                    for x in self.photo_list
                ]
        # Read prompts list
        self.prompt_file_list = [
            x.replace("/photo/", "/text/").replace("-photo.png", "-text.txt")
            for x in self.photo_list
        ]

        # Read prompts
        self.prompts = []
        for prompt_path in self.prompt_file_list:
            with open(prompt_path, "r") as f:
                self.prompts.append(f.read().strip())

        # Check if the files are valid
        for aov_name in self.required_aovs:
            if aov_name in self.aov_list.keys():
                for aov_path in self.aov_list[aov_name]:
                    assert os.path.exists(aov_path), f"{aov_path} does not exist."

        for prompt_path in self.prompt_file_list:
            assert os.path.exists(prompt_path), f"{prompt_path} does not exist."

        for photo_path in self.photo_list:
            assert os.path.exists(photo_path), f"{photo_path} does not exist."

        # Meta info
        self.total_num_of_c = 0
        for aov_name in self.required_aovs:
            if aov_name in self.aov_list.keys():
                self.total_num_of_c += self.aov_num_of_c[aov_name]

        print(
            f"Loaded {len(self.photo_list)} images from {self.dataset_dir}. Dataset will supply photo, prompt, and {self.total_num_of_c} channels of aovs: {self.aov_list.keys()}"
        )

        # Augmentation
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(required_resolution, antialias=True)
                if resize
                else transforms.Lambda(lambda x: x),
                (
                    transforms.CenterCrop(required_resolution)
                    if center_crop
                    else transforms.RandomCrop(required_resolution)
                )
                if crop
                else transforms.Lambda(lambda x: x),
            ]
        )

    def __len__(self):
        return len(self.photo_list)

    def __getitem__(self, idx):
        target_path = self.photo_list[idx]
        target = load_ldr_image(target_path, from_srgb=True)

        aovs = {}
        for aov_name in self.required_aovs:
            if aov_name == "albedo":
                aovs[aov_name] = load_ldr_image(
                    self.aov_list[aov_name][idx], from_srgb=True
                )
            elif aov_name == "normal":
                aovs[aov_name] = load_ldr_image(
                    self.aov_list[aov_name][idx], normalize=True
                )
            elif aov_name == "roughness":
                aovs[aov_name] = load_ldr_image(self.aov_list[aov_name][idx])
            elif aov_name == "metallic":
                aovs[aov_name] = load_ldr_image(self.aov_list[aov_name][idx])

        # Read normal anyway to get the mask
        normal_norm = torch.norm(aovs["normal"], dim=0, keepdim=True)
        mask = normal_norm > 1e-6
        for aov in aovs:
            aovs[aov] = aovs[aov] * mask
        target = target * mask

        # Transform
        transformed = self.train_transforms(
            torch.cat(list(aovs.values()) + [target], dim=0)
        )
        target = transformed[-3:]
        i = 0
        for aov_name in self.required_aovs:
            if aov_name in self.aov_num_of_c.keys():
                aovs[aov_name] = transformed[i : i + self.aov_num_of_c[aov_name]]
                # TODO: after transform, the value sometimes exceeds 1.0
                if aov_name == "normal":
                    aovs["normal"] = torch.nn.functional.normalize(
                        aovs["normal"], dim=0, eps=1e-6
                    )
                else:
                    aovs[aov_name] = torch.clamp(aovs[aov_name], min=0.0, max=1.0)
                i += self.aov_num_of_c[aov_name]

        # Fill the unsupported aovs with zero
        for aov_name in self.required_aovs:
            if aov_name not in self.aov_list.keys():
                aovs[aov_name] = torch.zeros_like(target)

        # Provide the valid or not information
        for aov_name in self.required_aovs:
            if aov_name in self.aov_list.keys():
                aovs[f"{aov_name}_valid"] = torch.tensor([True], dtype=torch.bool)
            else:
                aovs[f"{aov_name}_valid"] = torch.tensor([False], dtype=torch.bool)

        # The roughness and metallic are not reliable, so we mark them invalid here
        aovs["roughness_valid"] = torch.tensor([False], dtype=torch.bool)
        aovs["metallic_valid"] = torch.tensor([False], dtype=torch.bool)

        # Read prompt from file
        prompt = self.prompts[idx]

        return {
            "prompt": prompt,
            "target": target,
        } | aovs


class HypersimDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        phase="train",
        required_aovs=["albedo", "normal"],
        required_resolution=512,
        resize=True,
        crop=True,
        center_crop=False,
    ):
        super().__init__()
        self.aov_num_of_c = {
            "albedo": 3,
            "normal": 3,
            "irradiance": 3,
        }
        self.required_aovs = required_aovs.copy()

        # Always read normal
        if "normal" not in self.required_aovs:
            self.required_aovs.append("normal")

        self.dataset_dir = dataset_dir

        self.test_scene_list = ["ai_001_006", "ai_001_010", "ai_002_005", "ai_019_001"]
        # Read scene list
        self.scene_list = sorted(glob(os.path.join(self.dataset_dir, "*")))

        # Read photo list
        photo_list = [
            sorted(glob(os.path.join(scene, "images", "*_final_hdf5", "*.color.hdf5")))
            for scene in self.scene_list
        ]
        photo_list = [item for sublist in photo_list for item in sublist]

        # Read prompts
        self.prompts_path = "/sensei-fs/users/zhengz/datasets/hypersim/prompts.txt"
        prompts = []
        with open(self.prompts_path, "r") as f:
            for line in f.readlines():
                prompts.append(line.strip())
        self.photo_list = []
        self.prompts = []
        if phase == "train":
            # Remove all file containing name in test_scene_list
            # also remove the corresponding prompts, according to the index
            # Loop over the list
            for i, x in enumerate(photo_list):
                is_test = False
                for scene in self.test_scene_list:
                    if scene in x:
                        is_test = True
                        break
                if not is_test:
                    self.photo_list.append(x)
                    self.prompts.append(prompts[i])
        else:
            for i, x in enumerate(photo_list):
                is_test = False
                for scene in self.test_scene_list:
                    if scene in x:
                        is_test = True
                        break
                if is_test:
                    self.photo_list.append(x)
                    self.prompts.append(prompts[i])

        # Read other AOVs list
        self.aov_list = {}
        for aov_name in self.required_aovs:
            if aov_name == "normal":
                self.aov_list[aov_name] = [
                    x.replace("final", "geometry").replace("color", "normal_bump_cam")
                    for x in self.photo_list
                ]
            elif aov_name == "albedo":
                self.aov_list[aov_name] = [
                    x.replace("color", "diffuse_reflectance") for x in self.photo_list
                ]
            elif aov_name == "irradiance":
                self.aov_list[aov_name] = [
                    x.replace("color", "diffuse_illumination") for x in self.photo_list
                ]

        # Check if the files are valid
        for aov_name in self.required_aovs:
            if aov_name in self.aov_list.keys():
                for aov_path in self.aov_list[aov_name]:
                    assert os.path.exists(aov_path), f"{aov_path} does not exist."

        for photo_path in self.photo_list:
            assert os.path.exists(photo_path), f"{photo_path} does not exist."

        assert len(self.prompts) == len(
            self.photo_list
        ), f"Number of prompts {len(self.prompts)} does not match number of images {len(self.photo_list)}"

        # Meta info
        self.total_num_of_c = 0
        for aov_name in self.required_aovs:
            if aov_name in self.aov_list.keys():
                self.total_num_of_c += self.aov_num_of_c[aov_name]

        print(
            f"Loaded {len(self.photo_list)} images from {self.dataset_dir}. Dataset will supply photo, prompt, and {self.total_num_of_c} channels of aovs: {self.aov_list.keys()}"
        )

        # Augmentation
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(required_resolution, antialias=True)
                if resize
                else transforms.Lambda(lambda x: x),
                (
                    transforms.CenterCrop(required_resolution)
                    if center_crop
                    else transforms.RandomCrop(required_resolution)
                )
                if crop
                else transforms.Lambda(lambda x: x),
            ]
        )

    def __len__(self):
        return len(self.photo_list)

    def __getitem__(self, idx):
        target_path = self.photo_list[idx]
        target = load_hdf5_image(target_path, tonemaping=True, clamp=True)

        aovs = {}
        for aov_name in self.required_aovs:
            if aov_name == "albedo":
                aovs[aov_name] = load_hdf5_image(
                    self.aov_list[aov_name][idx], clamp=True
                )
            elif aov_name == "normal":
                aovs[aov_name] = load_hdf5_image(
                    self.aov_list[aov_name][idx], normalize=True
                )
            elif aov_name == "irradiance":
                aovs[aov_name] = load_hdf5_image(
                    self.aov_list[aov_name][idx], tonemaping=True, clamp=True,lum_scale=luminance(target_path) # scale the irradiance to match the target
                )

        # Read normal anyway to get the mask
        normal_norm = torch.norm(aovs["normal"], dim=0, keepdim=True)
        mask = normal_norm > 1e-6
        for aov in aovs:
            aovs[aov] = aovs[aov] * mask
        target = target * mask

        # Transform
        transformed = self.train_transforms(
            torch.cat(list(aovs.values()) + [target], dim=0)
        )
        target = transformed[-3:]

        # TODO: after transform, the value sometimes exceeds 1.0
        target = torch.clamp(target, min=0.0, max=1.0)

        i = 0
        for aov_name in self.required_aovs:
            if aov_name in self.aov_num_of_c.keys():
                aovs[aov_name] = transformed[i : i + self.aov_num_of_c[aov_name]]
                # TODO: after transform, the value sometimes exceeds 1.0
                if aov_name == "normal":
                    aovs["normal"] = torch.nn.functional.normalize(
                        aovs["normal"], dim=0, eps=1e-6
                    )
                else:
                    aovs[aov_name] = torch.clamp(aovs[aov_name], min=0.0, max=1.0)
                i += self.aov_num_of_c[aov_name]

        # Fill the unsupported aovs with zero
        for aov_name in self.required_aovs:
            if aov_name not in self.aov_list.keys():
                aovs[aov_name] = torch.zeros_like(target)

        # Provide the valid or not information
        for aov_name in self.required_aovs:
            if aov_name in self.aov_list.keys():
                aovs[f"{aov_name}_valid"] = torch.tensor([True], dtype=torch.bool)
            else:
                aovs[f"{aov_name}_valid"] = torch.tensor([False], dtype=torch.bool)

        # Read prompt from file
        prompt = self.prompts[idx]

        return {
            "prompt": prompt,
            "target": target,
        } | aovs


# Test code
if __name__ == "__main__":
    # iv_dataset = InteriorVerseDataset(
    #     dataset_dir="/mnt/localssd/interiorverse",
    #     sub_folder="dataset_120",
    #     phase="train",
    #     required_aovs=["albedo", "normal", "roughness", "metallic"],
    #     required_resolution=512,
    #     resize=True,
    #     crop=True,
    #     center_crop=False,
    #     denoised=True,
    # )

    # si_dataset = StockImagesDataset(
    #     dataset_dir="/sensei-fs/users/zhengz/datasets/testset/stock_images",
    #     # dataset_dir="/mnt/localssd/stock_images",
    #     phase="train",
    #     required_aovs=["albedo", "normal", "roughness"],
    #     required_resolution=512,
    #     resize=True,
    #     center_crop=False,
    # )

    hs_dataset = HypersimDataset(
        dataset_dir="/mnt/localssd/hypersim",
        phase="train",
        required_aovs=["albedo", "normal", "irradiance"],
        required_resolution=512,
        resize=True,
        center_crop=False,
    )

    # combined_dataset = ConcatDataset([iv_dataset, hs_dataset])

    # em_dataset = EverMotionDataset(
    #     dataset_dir="/mnt/localssd/synthworld_aov_render/evermotion",
    #     required_aovs=["albedo", "normal", "roughness", "metallic"],
    #     required_resolution=512,
    #     resize=True,
    #     center_crop=False,
    # )

    dataloader = torch.utils.data.DataLoader(
        hs_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    for i, batch in enumerate(dataloader):
        print(torch.max(batch["albedo"]))
        print(torch.min(batch["target"]),torch.max(batch["target"]))
        print(torch.min(batch["irradiance"]),torch.max(batch["target"]))
