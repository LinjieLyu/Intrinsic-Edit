<h1 align="center"> IntrinsicEdit: Precise Generative Image Manipulation in Intrinsic Space </h1>

<p align="center">ACM Transactions on Graphics (Proceedings of SIGGRAPH), 2025</p>

<p align="center"><img src="images/teaser.png"></p>


This is the official PyTorch implementation of the SIGGRAPH 2025 journal paper: [**IntrinsicEdit**](https://intrinsic-edit.github.io/)


## Environment

```
conda create -n intrinsic python=3.9 -y
conda activate intrinsic

pip install diffusers["torch"]==0.20
pip install transformers imageio torchvision wandb lpips opencv-python h5py
imageio_download_bin freeimage
pip install gradio==3.48.0
pip install markupsafe==2.0.1
```

## Trained Models

Pretrained models are download automatically to your huggingface cache folder during inference.

## Pipeline

1. Start with the original image.
2. Use `rgb2x` from [RGBX](https://github.com/zheng95z/rgbx) to decompose the image into intrinsic channels (e.g., albedo, normal, irradiance). Try different seeds for potentially better decompositions.
3. Edit one or more channels (e.g., modify the albedo channel for object removal or appearance editing).
4. Run the demo to view the edited results.

## Running the Demo

To launch the demo, simply run:

```
python gradio_demo_intrinsic_edit.py
```

## Example: Albedo-Based Editing
This example demonstrates object removal or appearance editing by modifying the albedo channel. Note that the irradiance channel is required by default.

Input:
- Original image
- Original irradiance (used for prompt optimization)
- Original albedo
- New (edited) albedo

For more examples, please refer to [results](https://github.com/LinjieLyu/IntrisinEdit/tree/main/images/results)
## Note: 
- If you want to edit other intrinsic channels (e.g. normal, roughness, irradiance), you must provide the original and new versions of that channel as a pair:

- For channels that are not being edited, you have two options:

1. Leave them blank — we will automatically fill in the missing channels for you.

2. Provide the same image as both the original and new version to indicate no changes:


