import argparse
import os
import sys
import safetensors

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch import autocast, inference_mode
import torchvision.transforms as transforms
import torchvision

import uvicorn
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel,DPMSolverMultistepScheduler

import gradio as gr
from fastapi import FastAPI

# Local imports
from dataset import load_exr_image, load_ldr_image
from pipeline_stable_diffusion_intrinsic_edit import StableDiffusionAOVDropoutPipeline_Inversion


def create_intrinsicedit_demo():
    # Load pipeline
    pipe = StableDiffusionAOVDropoutPipeline_Inversion.from_pretrained("zheng95z/x-to-rgb", torch_dtype=torch.float32).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        rescale_betas_zero_snr=True,
        solver_order=1,
        steps_offset=1,
        algorithm_type='dpmsolver++',
        prediction_type='v_prediction')
    pipe.to("cuda")

    def callback(
        photo, albedo, albedo_new, normal, normal_new, roughness, roughness_new, metallic, metallic_new,
        irradiance, irradiance_new, irradiance_text, mask, prompt, seed, inference_step, optimization_step,
        num_samples, guidance_scale, image_guidance_scale, traintext, saveprompt, loadprompt, loadnoise,
        skipinverse, augtext, decoderinv, transferweight, originweight, text_lr,
    ):
        # Set the number of inference steps
        pipe.scheduler.set_timesteps(inference_step)

        def process_image(file_obj, clamp=False, normalize=False, tonemaping=False, from_srgb=False):
            if file_obj is None:
                return None
            if isinstance(file_obj, str):
                name = file_obj
            else:
                name = file_obj.name
            # print(f"Processing image: {name}", flush=True)
            if name.endswith(".exr"):
                return load_exr_image(name, clamp=clamp, normalize=normalize, tonemaping=tonemaping).to("cuda")
            elif name.endswith((".png", ".jpg", ".jpeg")):
                return load_ldr_image(name, from_srgb=from_srgb, clamp=clamp, normalize=normalize).to("cuda")
            return None

        # Load condition images
        albedo_image = process_image(albedo, clamp=True, from_srgb=True)
        albedo_new_image = process_image(albedo_new, clamp=True, from_srgb=True)
        normal_image = process_image(normal, normalize=True)
        normal_new_image = process_image(normal_new, normalize=True)
        roughness_image = process_image(roughness, clamp=True)
        roughness_new_image = process_image(roughness_new, clamp=True)
        metallic_image = process_image(metallic, clamp=True)
        metallic_new_image = process_image(metallic_new, clamp=True)
        irradiance_image = process_image(irradiance, clamp=True, tonemaping=True, from_srgb=True)
        irradiance_new_image = process_image(irradiance_new, clamp=True, tonemaping=True, from_srgb=True)
        irradiance_text_image = process_image(irradiance_text, clamp=True, tonemaping=True, from_srgb=True)

        # Load input image, crop if width and height are not multiples of 8
        photo = process_image(photo, clamp=True, tonemaping=True, from_srgb=True)
        if photo.shape[1] % 8 != 0 or photo.shape[2] % 8 != 0:
            photo = torchvision.transforms.CenterCrop(
                (photo.shape[1] // 8 * 8, photo.shape[2] // 8 * 8)
            )(photo)

        height = photo.shape[1]
        width = photo.shape[2]

        required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
        encoder_prompts = {
            "albedo": "Albedo (diffuse basecolor)",
            "normal": "Camera-space Normal",
            "roughness": "Roughness",
            "metallic": "Metallicness",
            "irradiance": "Irradiance (diffuse lighting)",
        }

        # Check if any of the given images are not None
        images = [photo, albedo_image, normal_image, roughness_image, metallic_image, irradiance_image]
        for img in images:
            if img is not None:
                height = img.shape[1]
                width = img.shape[2]
                break

        # required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
        return_list = []

        generator = torch.Generator(device="cuda").manual_seed(seed)

        mask = torch.zeros_like(photo)

        recon_aovs = {
            'albedo': albedo_new_image,
            'normal': normal_new_image,
            'roughness': roughness_new_image,
            'metallic': metallic_new_image,
            'irradiance': irradiance_new_image
        }
        old_aovs = {
            'albedo': albedo_image,
            'normal': normal_image,
            'roughness': roughness_image,
            'metallic': metallic_image,
            'irradiance': irradiance_image
        }

        for i in range(num_samples):
            # Clean memory
            torch.cuda.empty_cache()
            torch.set_grad_enabled(True)

            # Prompt optimization and noise inversion
            if not skipinverse:
                xT,prompt_embeds = pipe.forward_diffusion(
                    mask=mask,
                    prompt=prompt,
                    photo=photo,
                    albedo=albedo_image,
                    albedo_old=albedo_image,
                    normal=normal_image,
                    normal_old=normal_image,
                    roughness=roughness_image,
                    roughness_old=roughness_image,
                    metallic=metallic_image,
                    metallic_old=metallic_image,
                    irradiance=irradiance_image,
                    irradiance_old=irradiance_image,
                    irradiance_text=irradiance_text_image,
                    num_inference_steps=inference_step,
                    num_optimization_steps=optimization_step,
                    height=height,
                    width=width,
                    generator=generator,
                    required_aovs=required_aovs,
                    # guidance_scale=guidance_scale,
                    # image_guidance_scale=image_guidance_scale,
                    guidance_rescale=0.7,
                    inverse_opt=True,
                    inv_order=1 ,
                    traintext=traintext,
                    augtext=augtext,
                    decoderinv=decoderinv,
                    transferweight=float(transferweight),
                    originweight=float(originweight),
                    text_lr=float(text_lr),
                )
                if saveprompt:
                    torch.save([xT,prompt_embeds], f"./output/optimized_albedo_latents.pt")

            if loadprompt:
                prompt_embeds = torch.load(f"./output/optimized_albedo_latents.pt")[1]
            if loadnoise:
                xT = torch.load(f"./output/optimized_albedo_latents.pt")[0]

            # Input reconstruction
            generated_image = pipe(
                mask=mask,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=prompt_embeds,
                photo=photo,
                albedo=albedo_image,
                albedo_old=albedo_image,
                normal=normal_image,
                normal_old=normal_image,
                roughness=roughness_image,
                roughness_old=roughness_image,
                metallic=metallic_image,
                metallic_old=metallic_image,
                irradiance=irradiance_image,
                irradiance_old=irradiance_image,
                num_inference_steps=inference_step,
                height=height,
                width=width,
                generator=generator,
                latents=xT,
                required_aovs=required_aovs,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                guidance_rescale=0.7,
                output_type="np",
                traintext=traintext,
            ).images[0]

            if num_samples > 1:
                return_list.append((generated_image, f"Input reconstruction [{i+1}]"))
            else:
                return_list.append((generated_image, f"Input reconstruction"))

            # Edited result
            generated_image = pipe(
                mask=mask,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=prompt_embeds,
                photo=photo,
                albedo=recon_aovs['albedo'],
                albedo_old=old_aovs['albedo'],
                normal=recon_aovs['normal'],
                normal_old=old_aovs['normal'],
                roughness=recon_aovs['roughness'],
                roughness_old=old_aovs['roughness'],
                metallic=recon_aovs['metallic'],
                metallic_old=old_aovs['metallic'],
                irradiance=recon_aovs['irradiance'],
                irradiance_old=old_aovs['irradiance'],
                num_inference_steps=inference_step,
                height=height,
                width=width,
                generator=generator,
                latents=xT,
                required_aovs=required_aovs,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                guidance_rescale=0.7,
                output_type="np",
                traintext=traintext,
            ).images[0]

            if num_samples > 1:
                return_list.append((generated_image, f"Edited result [{i+1}]"))
            else:
                return_list.append((generated_image, f"Edited result"))

        return return_list

    def create_ui():
        block = gr.Blocks()
        with block:
            with gr.Row():
                gr.Markdown("# IntrinsicEdit demo")
            with gr.Row():
                # Left main column, 50% wide, with 25%-wide subcolumns
                with gr.Column(scale=2):
                    gr.Markdown("## Inputs")
                    with gr.Row():
                        with gr.Column(min_width=100):
                            image_label = "Original image"
                            image = gr.Image(label=image_label, type="filepath")
                            # image = gr.File(label=image_label, file_types=[".exr", ".png", ".jpg"])
                        with gr.Column(min_width=100):
                            irradiance_text_label = "Original irradiance (for better identity preservation)"
                            irradiance_text = gr.Image(label=irradiance_text_label, type="filepath")
                            # irradiance_text = gr.File(label=irradiance_text_label, file_types=[".exr", ".png", ".jpg"])
                    with gr.Row():
                        with gr.Column(min_width=100):
                            albedo_label="Original albedo"
                            albedo = gr.Image(label=albedo_label, type="filepath")
                            # albedo = gr.File(label=albedo_label, file_types=[".exr", ".png", ".jpg"])
                        with gr.Column(min_width=100):
                            albedo_edited_label="Edited albedo"
                            albedo_edited = gr.Image(label=albedo_edited_label, type="filepath")
                            # albedo_edited = gr.File(label=albedo_edited_label, file_types=[".exr", ".png", ".jpg"])
                    with gr.Row():
                        with gr.Column(min_width=100):
                            normal_label="Original normal"
                            normal = gr.Image(label=normal_label, type="filepath")
                            # normal = gr.File(label=normal_label, file_types=[".exr", ".png", ".jpg"])
                        with gr.Column(min_width=100):
                            normal_edited_label="Edited normal"
                            normal_edited = gr.Image(label=normal_edited_label, type="filepath")
                            # normal_edited = gr.File(label=normal_edited_label, file_types=[".exr", ".png", ".jpg"])
                    with gr.Row():
                        with gr.Column(min_width=100):
                            roughness_label="Original roughness"
                            roughness = gr.Image(label=roughness_label, type="filepath")
                            # roughness = gr.File(label=roughness_label, file_types=[".exr", ".png", ".jpg"])
                        with gr.Column(min_width=100):
                            roughness_edited_label="Edited roughness"
                            roughness_edited = gr.Image(label=roughness_edited_label, type="filepath")
                            # roughness_edited = gr.File(label=roughness_edited_label, file_types=[".exr", ".png", ".jpg"])
                    with gr.Row():
                        with gr.Column(min_width=100):
                            metallic_label="Original metallic"
                            metallic = gr.Image(label=metallic_label, type="filepath")
                            # metallic = gr.File(label=metallic_label, file_types=[".exr", ".png", ".jpg"])
                        with gr.Column(min_width=100):
                            metallic_edited_label="Edited metallic"
                            metallic_edited = gr.Image(label=metallic_edited_label, type="filepath")
                            # metallic_edited = gr.File(label=metallic_edited_label, file_types=[".exr", ".png", ".jpg"])
                    with gr.Row():
                        with gr.Column(min_width=100):
                            irradiance_label="Original irradiance"
                            irradiance = gr.Image(label=irradiance_label, type="filepath")
                            # irradiance = gr.File(label=irradiance_label, file_types=[".exr", ".png", ".jpg"])
                        with gr.Column(min_width=100):
                            irradiance_edited_label="Edited irradiance"
                            irradiance_edited = gr.Image(label=irradiance_edited_label, type="filepath")
                            # irradiance_edited = gr.File(label=irradiance_edited_label, file_types=[".exr", ".png", ".jpg"])
                # Right main column, 50% wide
                with gr.Column(scale=2, min_width=200):
                    gr.Markdown("## Outputs")
                    result_gallery = gr.Gallery(label="Output", show_label=False, elem_id="gallery", columns=2)

                    run_button = gr.Button()

                    gr.Markdown("## Other parameters")

                    prompt = gr.Textbox(label="Prompt")

                    with gr.Accordion("Advanced options", open=False):
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        inference_step = gr.Slider(label="Number of inference steps", minimum=1, maximum=1000, step=1, value=50)
                        optimization_step = gr.Slider(label="Number of optimization steps", minimum=1, maximum=1000, step=1, value=200)
                        num_samples = gr.Slider(label="Number of samples", minimum=1, maximum=100, step=1, value=1)
                        guidance_scale = gr.Slider(label="Guidance scale", minimum=0.0, maximum=10.0, step=0.1, value=7.5)
                        image_guidance_scale = gr.Slider(label="Image guidance scale", minimum=0.0, maximum=10.0, step=0.1, value=1.5)
                        traintext = gr.Slider(label="traintext", minimum=0.0, maximum=1.0, step=1.0, value=1.0)
                        saveprompt = gr.Slider(label="saveprompt", minimum=0.0, maximum=1.0, step=1.0, value=0.0)
                        loadprompt = gr.Slider(label="loadprompt", minimum=0.0, maximum=1.0, step=1.0, value=0.0)
                        loadnoise = gr.Slider(label="loadnoise", minimum=0.0, maximum=1.0, step=1.0, value=0.0)
                        skipinverse = gr.Slider(label="skipinverse", minimum=0.0, maximum=1.0, step=1.0, value=0.0)
                        augtext = gr.Slider(label="augtext", minimum=0.0, maximum=1.0, step=1.0, value=1.0)
                        decoderinv = gr.Slider(label="decoderinv", minimum=0.0, maximum=1.0, step=1.0, value=0.0)
                        transferweight = gr.Slider(label="transferweight", minimum=-1.0, maximum=100.0, step=0.1, value=1.0)
                        originweight = gr.Slider(label="originweight", minimum=-1.0, maximum=1000.0, step=0.1, value=10.0)
                        text_lr = gr.Slider(label="text_lr", minimum=0.0, maximum=1.0, step=0.01, value=0.1)

            mask = albedo
            inputs = [
                image, albedo, albedo_edited, normal, normal_edited, roughness, roughness_edited,
                metallic, metallic_edited, irradiance, irradiance_edited, irradiance_text, mask, prompt,
                seed, inference_step, optimization_step, num_samples, guidance_scale, image_guidance_scale,
                traintext, saveprompt, loadprompt, loadnoise, skipinverse, augtext, decoderinv,
                transferweight, originweight, text_lr,
            ]
            run_button.click(fn=callback, inputs=inputs, outputs=result_gallery, queue=True)

        return block

    return create_ui()


if __name__ == "__main__":
    print("Starting IntrinsicEdit demo...")

    demo = create_intrinsicedit_demo()
    demo.queue(max_size=1)
    demo.launch(
        share=True,
        # server_name="0.0.0.0",
        # server_port=8888,
    )
