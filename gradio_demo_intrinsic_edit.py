import argparse
import os
import sys
import safetensors
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import uvicorn
from diffusers import DDIMScheduler, DDPMScheduler, UNet2DConditionModel,DPMSolverMultistepScheduler
from fastapi import FastAPI
import torchvision

from inversion_utils import inversion_forward_process, inversion_reverse_process,inversion_optimization_process
from torch import autocast, inference_mode

from dataset import load_exr_image, load_ldr_image
from pipeline_stable_diffusion_x2rgb_lightinglatent import (
    StableDiffusionAOVDropoutPipeline,
)
from pipeline_stable_diffusion_rgb2x import StableDiffusionAOVMatEstPipeline
# from pipeline_stable_diffusion_x2rgb_inversion import StableDiffusionAOVDropoutPipeline_Inversion
from pipeline_stable_diffusion_intrinsic_edit import StableDiffusionAOVDropoutPipeline_Inversion
# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp


PORT = 8888


def get_x2rgb_inversion_demo():
    
    # Load pipeline
    pipe = StableDiffusionAOVDropoutPipeline_Inversion.from_pretrained(
        # "/sensei-fs/users/mihasan/rgbx/unet-x2rgb-anrm-lightinglatent-iv-hs-em-fixn-fixmask-dropout-exIVrm-0115",
        "zheng95z/x-to-rgb",
        # scheduler=scheduler,
        torch_dtype=torch.float32,
    ).to("cuda")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, rescale_betas_zero_snr=True,solver_order=1,steps_offset=1
    ,algorithm_type='dpmsolver++',prediction_type='v_prediction') #timestep_spacing="trailing"

    
  
    pipe.to("cuda")

   
    # Augmentation
    def callback(
        photo,
        albedo,
        albedo_new,
        normal,
        normal_new,
        roughness,
        roughness_new,
        metallic,
        metallic_new,
        irradiance,
        irradiance_new,
        irradiance_text,
        mask,
        prompt,
        seed,
        inference_step,
        optimization_step,
        num_samples,
        guidance_scale,
        image_guidance_scale,

        traintext,
    
        saveprompt,
        loadprompt,
        loadnoise,
        skipinverse,
        augtext,
        decoderinv,
        transferweight,
        originweight,
        text_lr,
    ):
        # Set the number of inference steps
        pipe.scheduler.set_timesteps(inference_step)


         # Load condition images
        if True:
            if albedo is None:
                albedo_image = None
            elif albedo.name.endswith(".exr"):
                albedo_image = load_exr_image(albedo.name, clamp=True).to("cuda")
            elif (
                albedo.name.endswith(".png")
                or albedo.name.endswith(".jpg")
                or albedo.name.endswith(".jpeg")
            ):
                albedo_image = load_ldr_image(albedo.name, from_srgb=True).to("cuda")
            
            if albedo_new is None:
                albedo_new_image = None
            elif albedo_new.name.endswith(".exr"):
                albedo_new_image = load_exr_image(albedo_new.name, clamp=True).to("cuda")
            elif (
                albedo_new.name.endswith(".png")
                or albedo_new.name.endswith(".jpg")
                or albedo_new.name.endswith(".jpeg")
            ):
                albedo_new_image = load_ldr_image(albedo_new.name, from_srgb=True).to("cuda")
            

            if normal is None:
                normal_image = None
            elif normal.name.endswith(".exr"):
                normal_image = load_exr_image(normal.name, normalize=True).to("cuda")
            elif (
                normal.name.endswith(".png")
                or normal.name.endswith(".jpg")
                or normal.name.endswith(".jpeg")
            ):
                normal_image = load_ldr_image(normal.name, normalize=True).to("cuda")

            if normal_new is None:
                normal_new_image = None
            elif normal_new.name.endswith(".exr"):
                normal_new_image = load_exr_image(normal_new.name, normalize=True).to("cuda")
            elif (
                normal_new.name.endswith(".png")
                or normal_new.name.endswith(".jpg")
                or normal_new.name.endswith(".jpeg")
            ):
                normal_new_image = load_ldr_image(normal_new.name, normalize=True).to("cuda")

            if roughness is None:
                roughness_image = None
            elif roughness.name.endswith(".exr"):
                roughness_image = load_exr_image(roughness.name, clamp=True).to("cuda")
            elif (
                roughness.name.endswith(".png")
                or roughness.name.endswith(".jpg")
                or roughness.name.endswith(".jpeg")
            ):
                roughness_image = load_ldr_image(roughness.name, clamp=True).to("cuda")

            if roughness_new is None:
                roughness_new_image = None
            elif roughness.name.endswith(".exr"):
                roughness_new_image = load_exr_image(roughness_new.name, clamp=True).to("cuda")
            elif (
                roughness_new.name.endswith(".png")
                or roughness_new.name.endswith(".jpg")
                or roughness_new.name.endswith(".jpeg")
            ):
                roughness_new_image = load_ldr_image(roughness_new.name, clamp=True).to("cuda")

            if metallic is None:
                metallic_image = None
            elif metallic.name.endswith(".exr"):
                metallic_image = load_exr_image(metallic.name, clamp=True).to("cuda")
            elif (
                metallic.name.endswith(".png")
                or metallic.name.endswith(".jpg")
                or metallic.name.endswith(".jpeg")
            ):
                metallic_image = load_ldr_image(metallic.name, clamp=True).to("cuda")

            if metallic_new is None:
                metallic_new_image = None
            elif metallic_new.name.endswith(".exr"):
                metallic_new_image = load_exr_image(metallic_new.name, clamp=True).to("cuda")
            elif (
                metallic_new.name.endswith(".png")
                or metallic_new.name.endswith(".jpg")
                or metallic_new.name.endswith(".jpeg")
            ):
                metallic_new_image = load_ldr_image(metallic_new.name, clamp=True).to("cuda")


            if irradiance is None:
                irradiance_image = None
            elif irradiance.name.endswith(".exr"):
                irradiance_image = load_exr_image(
                    irradiance.name, tonemaping=True, clamp=True
                ).to("cuda")
            elif (
                irradiance.name.endswith(".png")
                or irradiance.name.endswith(".jpg")
                or irradiance.name.endswith(".jpeg")
            ):
                irradiance_image = load_ldr_image(
                    irradiance.name, from_srgb=True, clamp=True
                ).to("cuda")

            if irradiance_new is None:
                irradiance_new_image = None
            elif irradiance_new.name.endswith(".exr"):
                irradiance_new_image = load_exr_image(
                    irradiance_new.name, tonemaping=True, clamp=True
                ).to("cuda")
            elif (
                irradiance_new.name.endswith(".png")
                or irradiance_new.name.endswith(".jpg")
                or irradiance_new.name.endswith(".jpeg")
            ):
                irradiance_new_image = load_ldr_image(
                    irradiance_new.name, from_srgb=True, clamp=True
                ).to("cuda")

            if irradiance_text is None:
                irradiance_text_image = None
            elif irradiance_text.name.endswith(".exr"):
                irradiance_text_image = load_exr_image(
                    irradiance_text.name, tonemaping=True, clamp=True
                ).to("cuda")
            elif (
                irradiance_text.name.endswith(".png")
                or irradiance_text.name.endswith(".jpg")
                or irradiance_text.name.endswith(".jpeg")
            ):
                irradiance_text_image = load_ldr_image(
                    irradiance_text.name, from_srgb=True, clamp=True
                ).to("cuda")

            generator = torch.Generator(device="cuda").manual_seed(seed)

            # Load input photo
            if photo.name.endswith(".exr"):
                photo = load_exr_image(photo.name, tonemaping=True, clamp=True).to("cuda")
            elif (
                photo.name.endswith(".png")
                or photo.name.endswith(".jpg")
                or photo.name.endswith(".jpeg")
            ):
                photo = load_ldr_image(photo.name, from_srgb=True).to("cuda")

        
        # find Zs and wts - forward process
  
        

        # Check if the width and height are multiples of 8. If not, crop it using torchvision.transforms.CenterCrop
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
        images = [
            photo,
            albedo_image,
            normal_image,
            roughness_image,
            metallic_image,
            irradiance_image,
        ]
        for img in images:
            if img is not None:
                height = img.shape[1]
                width = img.shape[2]
                break

        required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
        return_list = []




        mask = torch.zeros_like(photo)

        recon_aovs = {'albedo': albedo_new_image, 'normal': normal_new_image, 'roughness': roughness_new_image, 'metallic': metallic_new_image, 'irradiance': irradiance_new_image}
        old_aovs = {'albedo': albedo_image, 'normal': normal_image, 'roughness': roughness_image, 'metallic': metallic_image, 'irradiance': irradiance_image}

        for i in range(num_samples):
            #clean memory
            torch.cuda.empty_cache()
            torch.set_grad_enabled(True)




            

            ######## prompt optimization and noise inversion ########
            if not skipinverse:
                xT,prompt_embeds = pipe.forward_diffusion(
                                mask=mask,
                                prompt=prompt,
                                photo=photo,
                                ########
                                albedo=albedo_image,
                                albedo_old=albedo_image,
                                ########
                                normal=normal_image,
                                normal_old=normal_image,
                                roughness=roughness_image,
                                roughness_old=roughness_image,
                                metallic=metallic_image,
                                metallic_old=metallic_image,
                                ######
                                irradiance=irradiance_image,
                                irradiance_old=irradiance_image,
                                irradiance_text=irradiance_text_image,
                                ########
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

            ######## image editing ########
            generated_image = pipe(
                mask=mask,
                # prompt=prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=prompt_embeds,
                photo=photo,
                ########
                albedo=albedo_image,
                albedo_old=albedo_image,
                ########
                normal=normal_image,
                normal_old=normal_image,
                roughness=roughness_image,
                roughness_old=roughness_image,
                metallic=metallic_image,
                metallic_old=metallic_image,
                ########
                irradiance=irradiance_image,
                irradiance_old=irradiance_image,
                ########
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
                ########
              
                traintext=traintext,

            ).images[0]

            generated_image = (generated_image, f"Generated Recon {i}")
            return_list.append(generated_image)

            generated_image = pipe(
                mask=mask,
                # prompt=prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=prompt_embeds,
                photo=photo,
                ########
                albedo=recon_aovs['albedo'],
                albedo_old=old_aovs['albedo'],
                ########
                normal=recon_aovs['normal'],
                normal_old=old_aovs['normal'],
                roughness=recon_aovs['roughness'],
                roughness_old=old_aovs['roughness'],
                metallic=recon_aovs['metallic'],
                metallic_old=old_aovs['metallic'],
                ########
                irradiance=recon_aovs['irradiance'],
                irradiance_old=old_aovs['irradiance'],
                ########
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
                ########
               
                traintext=traintext,
  
            ).images[0]
          

            generated_image = (generated_image, f"Generated Image {i}")
            return_list.append(generated_image)

            # generated_image = pipe(
            #     mask=mask,
            #     # prompt=prompt,
            #     prompt_embeds=prompt_embeds,
            #     negative_prompt_embeds=prompt_embeds,
            #     photo=photo,
            #     ########
            #     albedo=recon_aovs['albedo'],
            #     albedo_old=old_aovs['albedo'],
            #     ########
            #     normal=recon_aovs['normal'],
            #     normal_old=old_aovs['normal'],
            #     roughness=recon_aovs['roughness'],
            #     roughness_old=old_aovs['roughness'],
            #     metallic=recon_aovs['metallic'],
            #     metallic_old=old_aovs['metallic'],
            #     ########
            #     irradiance=recon_aovs['irradiance'],
            #     irradiance_old=old_aovs['irradiance'],
            #     ########
            #     num_inference_steps=inference_step,
            #     height=height,
            #     width=width,
            #     generator=generator,
            #     # latents=xT,
            #     required_aovs=required_aovs,
            #     guidance_scale=guidance_scale,
            #     image_guidance_scale=image_guidance_scale,
            #     guidance_rescale=0.7,
            #     output_type="np",
            #     ########
        
            #     traintext=traintext,

            # ).images[0]

            # generated_image = (generated_image, f"Generated Image {i}")
            # return_list.append(generated_image)


           
            
        return return_list

    block = gr.Blocks()
    with block:
        with gr.Row():
            gr.Markdown("## Model rgb2rgb inversion (photo -> photo)")
        with gr.Row():
            # Input side
            with gr.Column():
                gr.Markdown("### Given Image")
                photo = gr.File(label="Photo", file_types=[".exr", ".png", ".jpg"])
                irradiance_text = gr.File(label="Irradiance_text", file_types=[".exr", ".png", ".jpg"])
                albedo = gr.File(label="Albedo", file_types=[".exr", ".png", ".jpg"])
                albedo_new = gr.File(label="Albedo_new", file_types=[".exr", ".png", ".jpg"])

                normal = gr.File(label="Normal", file_types=[".exr", ".png", ".jpg"])
                normal_new = gr.File(label="Normal_new", file_types=[".exr", ".png", ".jpg"])
                roughness = gr.File(
                    label="Roughness", file_types=[".exr", ".png", ".jpg"]
                )
                roughness_new = gr.File(label="Roughness_new", file_types=[".exr", ".png", ".jpg"])
                metallic = gr.File(
                    label="Metallic", file_types=[".exr", ".png", ".jpg"]
                )
                metallic_new = gr.File(label="Metallic_new", file_types=[".exr", ".png", ".jpg"])
                irradiance = gr.File(
                    label="Irradiance", file_types=[".exr", ".png", ".jpg"]
                )
                irradiance_new = gr.File(label="Irradiance_new", file_types=[".exr", ".png", ".jpg"])

                gr.Markdown("### Parameters")
                prompt = gr.Textbox(label="Prompt")
                run_button = gr.Button()
                with gr.Accordion("Advanced options", open=False):
                    seed = gr.Slider(
                        label="Seed",
                        minimum=-1,
                        maximum=2147483647,
                        step=1,
                        randomize=True,
                    )
                    inference_step = gr.Slider(
                        label="Inference Step",
                        minimum=1,
                        maximum=1000,
                        step=1,
                        value=50,
                    )
                    optimization_step = gr.Slider(
                        label="Optimization Step",
                        minimum=1,
                        maximum=1000,
                        step=1,
                        value=200,
                    )
                    num_samples = gr.Slider(
                        label="Samples",
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=1,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=7.5,
                    )
                    image_guidance_scale = gr.Slider(
                        label="Image Guidance Scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=1.5,
                    )
                    
                    traintext = gr.Slider(
                        label="traintext",
                        minimum=0.0,
                        maximum=1.0,
                        step=1.0,
                        value=1.0,
                    )
                   
                    saveprompt = gr.Slider(
                        label="saveprompt",
                        minimum=0.0,
                        maximum=1.0,
                        step=1.0,
                        value=0.0,
                    )
                    loadprompt = gr.Slider(
                        label="loadprompt",
                        minimum=0.0,
                        maximum=1.0,
                        step=1.0,
                        value=0.0,
                    )
                    loadnoise = gr.Slider(
                        label="loadnoise",
                        minimum=0.0,
                        maximum=1.0,
                        step=1.0,
                        value=0.0,
                    )
                    skipinverse = gr.Slider(
                        label="skipinverse",
                        minimum=0.0,
                        maximum=1.0,
                        step=1.0,
                        value=0.0,
                    )
                    augtext = gr.Slider(
                        label="augtext",
                        minimum=0.0,
                        maximum=1.0,
                        step=1.0,
                        value=1.0,
                    )
                    decoderinv = gr.Slider(
                        label="decoderinv",
                        minimum=0.0,
                        maximum=1.0,
                        step=1.0,
                        value=0.0,
                    )
                    transferweight = gr.Slider(
                        label="transferweight",
                        minimum=-1.0,
                        maximum=100.0,
                        step=0.1,
                        value=1.0,
                    )
                    originweight = gr.Slider(
                        label="originweight",
                        minimum=-1.0,
                        maximum=1000.0,
                        step=0.1,
                        value=10.0,
                    )
                    text_lr = gr.Slider(
                        label="text_lr",
                        minimum=0.0,
                        maximum=1.0,
                        step=0.01,
                        value=0.1,
                    )

            # Output side
            with gr.Column():
                gr.Markdown("### Output Gallery")
                result_gallery = gr.Gallery(
                    label="Output",
                    show_label=False,
                    elem_id="gallery",
                    columns=2,
                )

        mask = albedo
        inputs = [
            photo,
            albedo,
            albedo_new,
            normal,
            normal_new,
            roughness,
            roughness_new,
            metallic,
            metallic_new,
            irradiance,
            irradiance_new,
            irradiance_text,
            mask,
            prompt,
            seed,
            inference_step,
            optimization_step,
            num_samples,
            guidance_scale,
            image_guidance_scale,
            traintext,
            saveprompt,
            loadprompt,
            loadnoise,
            skipinverse,
            augtext,
            decoderinv,
            transferweight,
            originweight,
            text_lr,
        ]
        run_button.click(fn=callback, inputs=inputs, outputs=result_gallery, queue=True)

    return block


if __name__ == "__main__":
    # for debug
    print("Starting the demo")
    demo = get_x2rgb_inversion_demo()

    demo.queue(max_size=1)
    demo.launch(
        share=True,
        # server_name="0.0.0.0",
        # server_port=PORT,
    )
