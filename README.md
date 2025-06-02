# Intrinsic Decomposition-Based Image Editing

This project demonstrates an image editing method based on **intrinsic image decomposition**.

## Environment

Refer to the [RGBX repository](https://github.com/zheng95z/rgbx) for environment setup and dependencies.

## Trained Models

Pretrained models are loaded directly from the [RGBX repository](https://github.com/zheng95z/rgbx).

## Pipeline

1. Start with the original image.
2. Use `rgb2x` from [RGBX](https://github.com/zheng95z/rgbx) to decompose the image into intrinsic channels (e.g., albedo, shading, irradiance). Try different seeds for potentially better decompositions.
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

## Note: 
- If you want to edit other intrinsic channels (e.g. normal, roughness, irradiance), you must provide the original and new versions of that channel as a pair:

- For channels that are not being edited, you have two options:

1. Leave them blank — we will automatically fill in the missing channels for you.

2. Provide the same image as both the original and new version to indicate no changes:


