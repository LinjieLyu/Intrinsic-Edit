import cv2
import numpy as np
import torch
from torchvision.transforms import CenterCrop
from diffusers import DPMSolverMultistepScheduler
import gradio as gr
from pipeline import IntrinsicEditPipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")


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


def load_image(file, width=None, height=None, clamp=False, normalize=False, tonemaping=False, from_srgb=False):
    image = None
    if file is not None:
        file_path = file if isinstance(file, str) else file.name
        # if file_path.endswith(".exr"):
        #     image = load_exr_image(file_path, clamp=clamp, normalize=normalize, tonemaping=tonemaping).to("cuda")
        if file_path.endswith((".png", ".jpg", ".jpeg")):
            image = load_ldr_image(file_path, from_srgb=from_srgb, clamp=clamp, normalize=normalize).to("cuda")
        if image is not None and width is not None and height is not None:
            image = CenterCrop((height, width))(image)
    return image


def create_intrinsicedit_demo():
    # Load pipeline
    pipeline = IntrinsicEditPipeline.from_pretrained("zheng95z/x-to-rgb", torch_dtype=torch.float32).to("cuda")
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config,
        rescale_betas_zero_snr=True,
        solver_order=1,
        steps_offset=1,
        algorithm_type='dpmsolver++',
        prediction_type='v_prediction')
    pipeline.to("cuda")

    def run_pipeline(
        image, albedo, albedo_edited, normal, normal_edited, roughness, roughness_edited, metallic, metallic_edited,
        irradiance, irradiance_edited, irradiance_text, prompt, seed, inference_step, optimization_step,
        num_samples, guidance_scale, image_guidance_scale, rgb2x_steps, traintext, saveprompt, loadprompt, loadnoise,
        skipinverse, augtext, decoderinv, transferweight, originweight, text_lr,
    ):
        # Set the number of inference steps
        pipeline.scheduler.set_timesteps(inference_step)

        image_temp = load_image(image, clamp=True, tonemaping=True, from_srgb=True)
        height = image_temp.shape[1] // 8 * 8
        width = image_temp.shape[2] // 8 * 8

        # Load input images, ensuring have the same dimensions that are also multiple of 8
        image = load_image(image, width=width, height=height, clamp=True, tonemaping=True, from_srgb=True)
        albedo = load_image(albedo, width=width, height=height, clamp=True, from_srgb=True)
        albedo_edited = load_image(albedo_edited, width=width, height=height, clamp=True, from_srgb=True)
        normal = load_image(normal, width=width, height=height, normalize=True)
        normal_edited = load_image(normal_edited, width=width, height=height, normalize=True)
        roughness = load_image(roughness, width=width, height=height, clamp=True)
        roughness_edited = load_image(roughness_edited, width=width, height=height, clamp=True)
        metallic = load_image(metallic, width=width, height=height, clamp=True)
        metallic_edited = load_image(metallic_edited, width=width, height=height, clamp=True)
        irradiance = load_image(irradiance, width=width, height=height, clamp=True, tonemaping=True, from_srgb=True)
        irradiance_edited = load_image(irradiance_edited, width=width, height=height, clamp=True, tonemaping=True, from_srgb=True)
        irradiance_text = load_image(irradiance_text, width=width, height=height, clamp=True, tonemaping=True, from_srgb=True)

        required_aovs = ["albedo", "normal", "roughness", "metallic", "irradiance"]
        return_list = []
        rng = torch.Generator(device="cuda").manual_seed(seed)
        mask = torch.zeros_like(image)

        print("--------")

        for i in range(num_samples):
            # Clean memory
            torch.cuda.empty_cache()
            torch.set_grad_enabled(True)

            # Prompt optimization and noise inversion
            if not skipinverse:
                xT, prompt_embeds = pipeline.forward_diffusion(
                    mask=mask,
                    prompt=prompt,
                    photo=image,
                    albedo=albedo,
                    albedo_old=albedo,
                    normal=normal,
                    normal_old=normal,
                    roughness=roughness,
                    roughness_old=roughness,
                    metallic=metallic,
                    metallic_old=metallic,
                    irradiance=irradiance,
                    irradiance_old=irradiance,
                    irradiance_text=irradiance_text,
                    num_inference_steps=inference_step,
                    num_optimization_steps=optimization_step,
                    height=height,
                    width=width,
                    generator=rng,
                    required_aovs=required_aovs,
                    # guidance_scale=guidance_scale,
                    # image_guidance_scale=image_guidance_scale,
                    guidance_rescale=0.7,
                    inverse_opt=True,
                    inv_order=1,
                    rgb2x_steps=rgb2x_steps,
                    traintext=traintext,
                    augtext=augtext,
                    decoderinv=decoderinv,
                    transferweight=float(transferweight),
                    originweight=float(originweight),
                    text_lr=float(text_lr),
                )
                if saveprompt:
                    torch.save([xT, prompt_embeds], f"./output/optimized_albedo_latents.pt")

            if loadprompt:
                prompt_embeds = torch.load(f"./output/optimized_albedo_latents.pt")[1]
            if loadnoise:
                xT = torch.load(f"./output/optimized_albedo_latents.pt")[0]

            # Input reconstruction
            generated_image = pipeline(
                mask=mask,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=prompt_embeds,
                photo=image,
                albedo=albedo,
                albedo_old=albedo,
                normal=normal,
                normal_old=normal,
                roughness=roughness,
                roughness_old=roughness,
                metallic=metallic,
                metallic_old=metallic,
                irradiance=irradiance,
                irradiance_old=irradiance,
                num_inference_steps=inference_step,
                height=height,
                width=width,
                generator=rng,
                latents=xT,
                required_aovs=required_aovs,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                guidance_rescale=0.7,
                output_type="np",
                traintext=traintext,
                task_name="Computing input reconstruction",
            ).images[0]

            if num_samples > 1:
                return_list.append((generated_image, f"Input reconstruction [{i+1}]"))
            else:
                return_list.append((generated_image, f"Input reconstruction"))

            # Edited result
            generated_image = pipeline(
                mask=mask,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=prompt_embeds,
                photo=image,
                albedo=albedo_edited,
                albedo_old=albedo,
                normal=normal_edited,
                normal_old=normal,
                roughness=roughness_edited,
                roughness_old=roughness,
                metallic=metallic_edited,
                metallic_old=metallic,
                irradiance=irradiance_edited,
                irradiance_old=irradiance,
                num_inference_steps=inference_step,
                height=height,
                width=width,
                generator=rng,
                latents=xT,
                required_aovs=required_aovs,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                guidance_rescale=0.7,
                output_type="np",
                traintext=traintext,
                task_name="Computing edited result",
            ).images[0]

            if num_samples > 1:
                return_list.append((generated_image, f"Edited result [{i+1}]"))
            else:
                return_list.append((generated_image, f"Edited result"))

        return return_list

    def create_ui():
        block = gr.Blocks(title="IntrinsicEdit demo", theme=gr.themes.Default())
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
                    result_gallery = gr.Gallery(
                        label="Output", show_label=False, elem_id="gallery",
                        columns=2, height="auto", object_fit="contain"
                    )

                    run_button = gr.Button()

                    # gr.Markdown("## Other parameters")

                    with gr.Accordion("Options", open=False):
                        prompt = gr.Textbox(label="Prompt")
                        num_samples = gr.Slider(label="Number of samples", minimum=1, maximum=100, step=1, value=1)
                        seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                        inference_step = gr.Slider(label="Number of inference steps", minimum=1, maximum=1000, step=1, value=50)
                        optimization_step = gr.Slider(label="Number of optimization steps", minimum=1, maximum=1000, step=1, value=200)
                        guidance_scale = gr.Slider(label="Guidance scale", minimum=0.0, maximum=10.0, step=0.1, value=7.5)
                        image_guidance_scale = gr.Slider(label="Image guidance scale", minimum=0.0, maximum=10.0, step=0.1, value=1.5)
                        with gr.Accordion("Advanced", open=False):
                            rgb2x_steps = gr.Slider(label="RGB→X inference steps", minimum=1, maximum=1000, step=1, value=2)
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

            inputs = [
                image, albedo, albedo_edited, normal, normal_edited, roughness, roughness_edited,
                metallic, metallic_edited, irradiance, irradiance_edited, irradiance_text, prompt,
                seed, inference_step, optimization_step, num_samples, guidance_scale, image_guidance_scale,
                rgb2x_steps, traintext, saveprompt, loadprompt, loadnoise, skipinverse, augtext, decoderinv,
                transferweight, originweight, text_lr,
            ]
            run_button.click(fn=run_pipeline, inputs=inputs, outputs=result_gallery, queue=True)

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
