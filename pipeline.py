import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, FrozenDict, register_to_config
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import (
    FromSingleFileMixin,
    LoraLoaderMixin,
    TextualInversionLoaderMixin,
)
from diffusers.models import AutoencoderKL, UNet2DConditionModel,AutoencoderTiny,VQModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    rescale_noise_cfg,
)
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    CONFIG_NAME,
    BaseOutput,
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,

    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer
from torch import autocast, inference_mode
logger = logging.get_logger(__name__)

import os
import safetensors
from torch import nn
from torchvision.transforms import GaussianBlur

from accelerate import Accelerator
from diffusers.loaders import AttnProcsLayers, LoraLoaderMixin
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import (
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    LoRAAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    SlicedAttnAddedKVProcessor,
)
from diffusers.optimization import get_scheduler
import tqdm
from accelerate.utils import set_seed
from transformers import AutoConfig

from torch.optim.lr_scheduler import ReduceLROnPlateau,LambdaLR
import math


class VaeImageProcrssorAOV(VaeImageProcessor):
    """
    Image processor for VAE AOV.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to downscale the image's (height, width) dimensions to multiples of `vae_scale_factor`.
        vae_scale_factor (`int`, *optional*, defaults to `8`):
            VAE scale factor. If `do_resize` is `True`, the image is automatically resized to multiples of this factor.
        resample (`str`, *optional*, defaults to `lanczos`):
            Resampling filter to use when resizing the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image to [-1,1].
    """

    config_name = CONFIG_NAME

    @register_to_config
    def __init__(
        self,
        do_resize: bool = True,
        vae_scale_factor: int = 8,
        resample: str = "lanczos",
        do_normalize: bool = True,
    ):
        super().__init__()

    def postprocess(
        self,
        image: torch.FloatTensor,
        output_type: str = "pil",
        do_denormalize: Optional[List[bool]] = None,
        do_gamma_correction: bool = True,
    ):
        if not isinstance(image, torch.Tensor):
            raise ValueError(
                f"Input for postprocessing is in incorrect format: {type(image)}. We only support pytorch tensor"
            )
        if output_type not in ["latent", "pt", "np", "pil"]:
            deprecation_message = (
                f"the output_type {output_type} is outdated and has been set to `np`. Please make sure to set it to one of these instead: "
                "`pil`, `np`, `pt`, `latent`"
            )
            deprecate(
                "Unsupported output_type",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            output_type = "np"

        if output_type == "latent":
            return image

        if do_denormalize is None:
            do_denormalize = [self.config.do_normalize] * image.shape[0]

        image = torch.stack(
            [
                self.denormalize(image[i]) if do_denormalize[i] else image[i]
                for i in range(image.shape[0])
            ]
        )

        # Gamma correction
        if do_gamma_correction:
            image = torch.pow(image, 1.0 / 2.2)

        if output_type == "pt":
            return image

        image = self.pt_to_numpy(image)

        if output_type == "np":
            return image

        if output_type == "pil":
            return self.numpy_to_pil(image)

    def preprocess_normal(
        self,
        image: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray],
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        image = torch.stack([image], axis=0)
        return image


@dataclass
class StableDiffusionAOVPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion AOV pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or NumPy array of shape `(batch_size, height, width,
            num_channels)`.
        nsfw_content_detected (`List[bool]`)
            List indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content or
            `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    predicted_x0_images: Optional[Union[List[PIL.Image.Image], np.ndarray]] = None


class IntrinsicEditPipeline(DiffusionPipeline, TextualInversionLoaderMixin, LoraLoaderMixin):
    r"""
    Pipeline for AOVs.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    The pipeline also inherits the following loading methods:
        - [`~loaders.TextualInversionLoaderMixin.load_textual_inversion`] for loading textual inversion embeddings
        - [`~loaders.LoraLoaderMixin.load_lora_weights`] for loading LoRA weights
        - [`~loaders.LoraLoaderMixin.save_lora_weights`] for saving LoRA weights

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`~transformers.CLIPTextModel`]):
            Frozen text-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
        tokenizer ([`~transformers.CLIPTokenizer`]):
            A `CLIPTokenizer` to tokenize text.
        unet ([`UNet2DConditionModel`]):
            A `UNet2DConditionModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcrssorAOV(
            vae_scale_factor=self.vae_scale_factor
        )
        self.register_to_config()

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_ prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(
                prompt, padding="longest", return_tensors="pt"
            ).input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1
            ] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            print(text_input_ids.shape)
            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: procecss multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            # pix2pix has two  negative embeddings, and unlike in other pipelines latents are ordered [prompt_embeds, negative_prompt_embeds, negative_prompt_embeds]
            prompt_embeds = torch.cat(
                [prompt_embeds, negative_prompt_embeds, negative_prompt_embeds]
            )

        return prompt_embeds

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        callback_steps,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if (callback_steps is None) or (
            callback_steps is not None
            and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_image_latents(
        self,
        image,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        do_classifier_free_guidance,
        image_old=None,
        generator=None,
    ):
        if not isinstance(image, (torch.Tensor, PIL.Image.Image, list)):
            raise ValueError(f"`image` has to be of type `torch.Tensor`, `PIL.Image.Image` or list but is {type(image)}")

        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == 4:
            image_latents = image
        else:
            if isinstance(generator, list) and len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            if isinstance(generator, list):
                image_latents = [
                    self.vae.encode(image[i : i + 1]).latent_dist.mode()
                    for i in range(batch_size)
                ]
                image_latents = torch.cat(image_latents, dim=0)
            else:
                image_latents = self.vae.encode(image).latent_dist.mode()

        if (
            batch_size > image_latents.shape[0]
            and batch_size % image_latents.shape[0] == 0
        ):
            # expand image_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {image_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate(
                "len(prompt) != len(image)",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            additional_image_per_prompt = batch_size // image_latents.shape[0]
            image_latents = torch.cat(
                [image_latents] * additional_image_per_prompt, dim=0
            )
        elif (
            batch_size > image_latents.shape[0]
            and batch_size % image_latents.shape[0] != 0
        ):
            raise ValueError(
                f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            image_latents = torch.cat([image_latents], dim=0)

        if do_classifier_free_guidance:
            if image_old is not None:
                uncond_image_latents=self.vae.encode(image_old).latent_dist.mode()
            else:
                uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    def latent_optimization(self, measurement, z_init,mask, eps=1e-3, max_iters=50, lr=None):
        """
        Function to compute argmin_z ||y - A( D(z) )||_2^2

        Arguments:
            measurement:           Measurement vector y in y=Ax+n.
            z_init:                Starting point for optimization
            operator_fn:           Operator to perform forward operation A(.)
            eps:                   Tolerance error
            max_iters:             Maximum number of GD iterations

        Optimal parameters seem to be at around 500 steps, 200 steps for inpainting.
        """
        with torch.enable_grad():
            # Base case
            if not z_init.requires_grad:
                z_init = z_init.requires_grad_()

            if lr is None:
                lr_val = 1e-2
            else:
                lr_val = lr.item()

            loss = torch.nn.MSELoss()  # MSE loss
            optimizer = torch.optim.AdamW([z_init], lr=lr_val)  # Initializing optimizer ###change the learning rate
            measurement = measurement.detach()  # Need to detach for weird PyTorch reasons

            # Training loop
            init_loss = 0
            losses = []

            for itr in range(max_iters):
                optimizer.zero_grad()
                output = loss(measurement*mask, mask*self.vae.decode(z_init/ self.vae.config.scaling_factor, return_dict=False)[0])

                if itr == 0:
                    init_loss = output.detach().clone()

                output.backward()  # Take GD step
                optimizer.step()
                cur_loss = output.detach().cpu().numpy()

                # Convergence criteria

                if itr < 200:  # may need tuning for early stopping
                    losses.append(cur_loss)
                else:
                    losses.append(cur_loss)
                    if losses[0] < cur_loss:
                        break
                    else:
                        losses.pop(0)

                if cur_loss < eps ** 2:  # needs tuning according to noise level for early stopping
                    break

        return z_init, init_loss

    def stochastic_resample(self, pseudo_x0, x_t, a_t, sigma):
        """
        Function to resample x_t based on ReSample paper.
        """
        device = pseudo_x0.device
        noise = torch.randn_like(pseudo_x0, device=device)
        return (sigma * a_t.sqrt() * pseudo_x0 + (1 - a_t) * x_t) / (sigma + 1 - a_t) + noise * torch.sqrt(1 / (1 / sigma + 1 / (1 - a_t)))

    @torch.no_grad()
    def __call__(
        self,
        height: int,
        width: int,
        prompt: Union[str, List[str]] = None,
        photo: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]] = None,
        albedo_old: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        normal_old: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        roughness_old: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        metallic_old: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        irradiance_old: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        albedo: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        normal: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        roughness: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        metallic: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        irradiance: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        mask=None,
        guidance_scale: float = 0.0,
        image_guidance_scale: float = 0.0,
        guidance_rescale: float = 0.0,
        num_inference_steps: int = 100,
        required_aovs: List[str] = ["albedo"],
        return_predicted_x0s: bool = False,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        traintext=False,
    ):
        # 0. Check inputs
        self.check_inputs(
            prompt,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = (
            guidance_scale >= 1.0 and image_guidance_scale >= 1.0
        )
        # check if scheduler is in sigmas space
        scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")

        # 2. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance,
            negative_prompt, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
        )

        # 3. Preprocess image
        # For normal, the preprocessing does nothing
        # For others, the preprocessing remap the values to [-1, 1]
        preprocessed_aovs = {}
        preprocessed_aovs_old = {aov_name: None for aov_name in required_aovs}

        for aov_name in required_aovs:
            if aov_name == "albedo":
                if albedo is not None:
                    preprocessed_aovs[aov_name] = self.image_processor.preprocess(albedo)
                else:
                    preprocessed_aovs[aov_name] = None
            if aov_name == "normal":
                if normal is not None:
                    preprocessed_aovs[aov_name] = self.image_processor.preprocess_normal(normal)
                else:
                    preprocessed_aovs[aov_name] = None
            if aov_name == "roughness":
                if roughness is not None:
                    preprocessed_aovs[aov_name] = self.image_processor.preprocess(roughness)
                else:
                    preprocessed_aovs[aov_name] = None
            if aov_name == "metallic":
                if metallic is not None:
                    preprocessed_aovs[aov_name] = self.image_processor.preprocess(metallic)
                else:
                    preprocessed_aovs[aov_name] = None
            if aov_name == "irradiance":
                if irradiance is not None:
                    preprocessed_aovs[aov_name] = self.image_processor.preprocess(irradiance)
                else:
                    preprocessed_aovs[aov_name] = None

        if albedo_old is not None:
            preprocessed_aovs_old["albedo"] = self.image_processor.preprocess(albedo_old)
        if normal_old is not None:
            preprocessed_aovs_old["normal"] = self.image_processor.preprocess_normal(normal_old)
        if roughness_old is not None:
            preprocessed_aovs_old["roughness"] = self.image_processor.preprocess(roughness_old)
        if metallic_old is not None:
            preprocessed_aovs_old["metallic"] = self.image_processor.preprocess(metallic_old)
        if mask is not None:
            albedo_mask = mask
            if albedo_mask.dim() < 4:
                albedo_mask = albedo_mask[None, :, :, :]

            do_denormalize = [True] * albedo_mask.shape[0]

            generated_image = self.image_processor.postprocess(albedo_mask, output_type='pil', do_denormalize=do_denormalize)[0]

        if irradiance_old is not None:
            preprocessed_aovs_old["irradiance"] = self.image_processor.preprocess(irradiance_old)

        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        height_latent, width_latent = latents.shape[-2:]

        # 6. Prepare Image latents
        image_latents = []
        scaling_factors = {
            "albedo": 0.17301377137652138,
            "normal": 0.17483895473058078,
            "roughness": 0.1680724853626448,
            "metallic": 0.13135013390855135,
        }
        for aov_name, aov in preprocessed_aovs.items():
            if aov is None:
                image_latent = torch.zeros(
                    batch_size,
                    num_channels_latents,
                    height_latent,
                    width_latent,
                    dtype=prompt_embeds.dtype,
                    device=device,
                )
                if aov_name == "irradiance":
                    image_latent = image_latent[:, 0:3]

                if do_classifier_free_guidance:
                    image_latents.append(torch.cat([image_latent, image_latent, image_latent], dim=0))
                else:
                    image_latents.append(image_latent)
            else:
                if aov_name == "irradiance":
                    image_latent = F.interpolate(
                        aov.to(device=device, dtype=prompt_embeds.dtype),
                        size=(height_latent, width_latent),
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )
                    if do_classifier_free_guidance:
                        if irradiance_old is not None:
                            uncond_image_latent = F.interpolate(
                                preprocessed_aovs_old["irradiance"].to(device=device, dtype=prompt_embeds.dtype),
                                size=(height_latent, width_latent),
                                mode="bilinear",
                                align_corners=False,
                                antialias=True,
                            )
                        else:
                            uncond_image_latent = torch.zeros_like(image_latent)
                        image_latent = torch.cat([image_latent, image_latent, uncond_image_latent], dim=0)
                else:
                    scaling_factor = scaling_factors[aov_name]
                    image_latent = (
                        self.prepare_image_latents(
                            aov,
                            batch_size,
                            num_images_per_prompt,
                            prompt_embeds.dtype,
                            device,
                            do_classifier_free_guidance,
                            preprocessed_aovs_old[aov_name],
                            generator,
                        )
                        * scaling_factor
                    )
                image_latents.append(image_latent)
        image_latents = torch.cat(image_latents, dim=1)

        if photo is not None:
            preprocessed_photo = self.image_processor.preprocess(photo)

        photo_latents = self.prepare_image_latents(
            preprocessed_photo, batch_size, num_images_per_prompt,
            prompt_embeds.dtype, device, False, generator,
        )
        photo_latents = photo_latents * self.vae.config.scaling_factor

        # 7. Check that shapes of latents and image match the UNet channels
        num_channels_image = image_latents.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        predicted_x0s = []

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # Expand the latents if we are doing classifier free guidance.
                # The latents are expanded 3 times because for pix2pix the guidance\
                # is applied for both the text and the input image.
                latent_model_input = (torch.cat([latents] * 3) if do_classifier_free_guidance else latents)

                # concat latents, image_latents in the channel dimension
                scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

                # predict the noise residual
                noise_pred = self.unet( scaled_latent_model_input, t, encoder_hidden_states=prompt_embeds, return_dict=False)[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_image)
                        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                    )

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1

                latents_prev = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                # if i%10==0:
                #
                #     s = t - self.scheduler.num_train_timesteps // self.scheduler.num_inference_steps
                #     a_prev = self.scheduler.alphas_cumprod[s]
                #     # sigma = self.scheduler.sigma_t[s]
                #     pseudo_x0=self.scheduler.convert_model_output(noise_pred, t, latents).detach()
                #     a_t = self.scheduler.alphas_cumprod[t]
                #     sigma = 20*(1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)
                #     pseudo_x0,_= self.latent_optimization(preprocessed_photo,pseudo_x0 ,albedo_mask, eps=1e-3, max_iters=100, lr=None)
                #     latents_prev= self.stochastic_resample(pseudo_x0=pseudo_x0, x_t=latents_prev, a_t=a_prev, sigma=sigma)

                latents=latents_prev
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]

            if return_predicted_x0s:
                predicted_x0_images = [
                    self.vae.decode(predicted_x0 / self.vae.config.scaling_factor, return_dict=False)[0]
                    for predicted_x0 in predicted_x0s
                ]
        else:
            image = latents
            predicted_x0_images = predicted_x0s

        do_denormalize = [True] * image.shape[0]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        if return_predicted_x0s:
            predicted_x0_images = [
                self.image_processor.postprocess(predicted_x0_image, output_type=output_type, do_denormalize=do_denormalize)
                for predicted_x0_image in predicted_x0_images
            ]

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        if not return_dict:
            return image

        if return_predicted_x0s:
            return StableDiffusionAOVPipelineOutput(images=image, predicted_x0_images=predicted_x0_images)
        else:
            return StableDiffusionAOVPipelineOutput(images=image)


    #@torch.no_grad()
    def forward_diffusion(
        self,
        height: int,
        width: int,
        prompt: Union[str, List[str]] = None,
        photo: Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]] = None,
        albedo_old: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        normal_old: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        roughness_old: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        metallic_old: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        irradiance_old: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        irradiance_text: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        albedo: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        normal: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        roughness: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        metallic: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        irradiance: Optional[Union[torch.FloatTensor, PIL.Image.Image, np.ndarray, List[torch.FloatTensor], List[PIL.Image.Image], List[np.ndarray]]] = None,
        mask=None,
        guidance_scale: float = 0.0,
        image_guidance_scale: float = 0.0,
        guidance_rescale: float = 0.0,
        num_inference_steps: int = 100,
        num_optimization_steps: int = 200,
        required_aovs: List[str] = ["albedo"],
        return_predicted_x0s: bool = False,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        use_old_emb_i=25,
        inverse_opt=True,
        inv_order=None,
        traintext=False,
        augtext=True,
        decoderinv=False,
        transferweight=1.0,
        originweight=1.0,
        text_lr=1e-1,
    ):
        # 0. Check inputs
        self.check_inputs(prompt, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds)

        # 1. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = (
            guidance_scale >= 1.0 and image_guidance_scale >= 1.0
        )
        # check if scheduler is in sigmas space
        scheduler_is_in_sigma_space = hasattr(self.scheduler, "sigmas")

        # 2. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        )

        # 3. Preprocess image
        # For normal, the preprocessing does nothing
        # For others, the preprocessing remap the values to [-1, 1]

        preprocessed_aovs = {}
        preprocessed_aovs_old={aov_name: None for aov_name in required_aovs}
        for aov_name in required_aovs:
            if aov_name == "albedo":
                if albedo is not None:
                    preprocessed_aovs[aov_name] = self.image_processor.preprocess(albedo)
                else:
                    preprocessed_aovs[aov_name] = None

            if aov_name == "normal":
                if normal is not None:
                    preprocessed_aovs[aov_name] = self.image_processor.preprocess_normal(normal)
                else:
                    preprocessed_aovs[aov_name] = None

            if aov_name == "roughness":
                if roughness is not None:
                    preprocessed_aovs[aov_name] = self.image_processor.preprocess(roughness)
                else:
                    preprocessed_aovs[aov_name] = None
            if aov_name == "metallic":
                if metallic is not None:
                    preprocessed_aovs[aov_name] = self.image_processor.preprocess(metallic)
                else:
                    preprocessed_aovs[aov_name] = None
            if aov_name == "irradiance":
                if irradiance is not None:
                    preprocessed_aovs[aov_name] = self.image_processor.preprocess(irradiance)
                else:
                    preprocessed_aovs[aov_name] = None

        if albedo_old is not None:
            preprocessed_aovs_old["albedo"] = self.image_processor.preprocess(albedo_old)
        if normal_old is not None:
            preprocessed_aovs_old["normal"] = self.image_processor.preprocess_normal(normal_old)
        if roughness_old is not None:
            preprocessed_aovs_old["roughness"] = self.image_processor.preprocess(roughness_old)
        if metallic_old is not None:
            preprocessed_aovs_old["metallic"] = self.image_processor.preprocess(metallic_old)
        if mask is not None:
            albedo_mask = mask
            if albedo_mask.dim()<4:
                albedo_mask = albedo_mask[None,:,:,:]

            do_denormalize = [True] * albedo_mask.shape[0]

            generated_image = self.image_processor.postprocess(albedo_mask, output_type='pil', do_denormalize=do_denormalize)[0]

        if irradiance_old is not None:
            preprocessed_aovs_old["irradiance"] = self.image_processor.preprocess(irradiance_old)


        # 4. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        timesteps = reversed(timesteps) # inversion process

        # 5. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        # latents = self.prepare_latents(
        #     batch_size * num_images_per_prompt,
        #     num_channels_latents,
        #     height,
        #     width,
        #     prompt_embeds.dtype,
        #     device,
        #     generator,
        #     latents,
        # )
        # height_latent, width_latent = latents.shape[-2:]
        if photo is not None:
            preprocessed_photo = self.image_processor.preprocess(photo)
        photo_latents = self.prepare_image_latents(
            preprocessed_photo, batch_size, num_images_per_prompt,
            prompt_embeds.dtype,device, False, generator,
        )
        photo_latents = photo_latents * self.vae.config.scaling_factor
        height_latent, width_latent = photo_latents.shape[-2:]
        latents=photo_latents
        latents=latents * self.scheduler.init_noise_sigma


        # decoder inverse optimization
        if decoderinv:
            print('decoder inverse optimization')
            latents = self.decoder_inv(preprocessed_photo, latents, mask=1.)


        # 6. Prepare Image latents
        image_latents = []
        scaling_factors = {
            "albedo": 0.17301377137652138,
            "normal": 0.17483895473058078,
            "roughness": 0.1680724853626448,
            "metallic": 0.13135013390855135,
        }
        for aov_name, aov in preprocessed_aovs.items():
            if aov is None:
                image_latent = torch.zeros(
                    batch_size,
                    num_channels_latents,
                    height_latent,
                    width_latent,
                    dtype=prompt_embeds.dtype,
                    device=device,
                )
                if aov_name == "irradiance":
                    image_latent = image_latent[:, 0:3]
                if do_classifier_free_guidance:
                    image_latents.append(torch.cat([image_latent, image_latent, image_latent], dim=0))
                else:
                    image_latents.append(image_latent)
            else:
                if aov_name == "irradiance":
                    image_latent = F.interpolate(
                        aov.to(device=device, dtype=prompt_embeds.dtype),
                        size=(height_latent, width_latent),
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )
                    if do_classifier_free_guidance:
                        if irradiance_old is not None:
                            uncond_image_latent = F.interpolate(
                                preprocessed_aovs_old["irradiance"].to(device=device, dtype=prompt_embeds.dtype),
                                size=(height_latent, width_latent),
                                mode="bilinear",
                                align_corners=False,
                                antialias=True,
                            )
                        else:
                            uncond_image_latent = torch.zeros_like(image_latent)
                        image_latent = torch.cat([image_latent, image_latent, uncond_image_latent], dim=0)
                else:
                    scaling_factor = scaling_factors[aov_name]
                    image_latent = (
                        self.prepare_image_latents(
                            aov,
                            batch_size,
                            num_images_per_prompt,
                            prompt_embeds.dtype,
                            device,
                            do_classifier_free_guidance,
                            preprocessed_aovs_old[aov_name],
                            generator,
                        )
                        * scaling_factor
                    )
                image_latents.append(image_latent)
        image_latents = torch.cat(image_latents, dim=1)

        if irradiance_text is not None:
            preprocessed_irradiance_text = self.image_processor.preprocess(irradiance_text)
            irradiance_text_latent=F.interpolate(
                preprocessed_irradiance_text.to(device=device, dtype=prompt_embeds.dtype),
                size=(height_latent, width_latent),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        else:
            irradiance_text_latent=None

        # 7. Check that shapes of latents and image match the UNet channels
        num_channels_image = image_latents.shape[1]
        if num_channels_latents + num_channels_image != self.unet.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents+num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        # 8. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        predicted_x0s = []

        # 8.5 train text prompt
        prompt_embeds_nocfg=prompt_embeds[:1]
        rgb2x_prompt_embeds=[]
        rgb2x_prompt_embeds.append(self._encode_prompt(
            "Albedo (diffuse basecolor)",#
            device,
            num_images_per_prompt,
            False,
        ).detach())
        rgb2x_prompt_embeds.append(self._encode_prompt(
            "Camera-space Normal",  #
            device,
            num_images_per_prompt,
            False,
        ).detach())
        rgb2x_prompt_embeds.append(self._encode_prompt(
            "Roughness",  #
            device,
            num_images_per_prompt,
            False,
        ).detach())
        rgb2x_prompt_embeds.append(self._encode_prompt(
            "Metallicness",  #
            device,
            num_images_per_prompt,
            False,
        ).detach())


        #### prompt optimization
        if traintext:
            prompt_embeds=train_text(latents, image_latents,irradiance_text_latent,prompt_embeds_nocfg,rgb2x_prompt_embeds,preprocessed_aovs,unet=self.unet
                ,augtext=augtext,transferweight=transferweight,originweight=originweight,traintext=traintext, text_lr=text_lr, steps=num_optimization_steps)

        self.unet.requires_grad_(False)

        # 9. Noise inversion loop
        # with torch.no_grad():
        latents_0 = latents.detach().clone()
        image_latents= image_latents.detach().clone()
        prompt_embeds = prompt_embeds.detach().clone()
        noise = torch.randn_like(latents)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        # Fixed point inversion
        # Note during the inversion there is no cfg guidance, so the guidance_scale and image_guidance_scale are set to 0.0
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            T=timesteps[-1]
            lambda_T = self.scheduler.lambda_t[-1]
            sigma_T = self.scheduler.sigma_t[-1]
            alpha_T = self.scheduler.alpha_t[-1]
            for i, t in enumerate(timesteps):
                prev_timestep = (
                    t
                    - self.scheduler.config.num_train_timesteps
                    // self.scheduler.num_inference_steps
                )

                s = t
                t = prev_timestep

                x_t = latents.detach().clone()
                print(f"t: {t}, s: {s}, T: {T}")

                latents, noise_s = self.fixedpoint_correction(
                    i,latents, s, t, x_t, order=1,latents_0=latents_0,image_latents=image_latents,
                    prompt_embeds=prompt_embeds, guidance_scale=guidance_scale,image_guidance_scale=image_guidance_scale,
                    guidance_rescale=guidance_rescale,step_size=1.0, scheduler=True,**extra_step_kwargs)

        return latents, prompt_embeds

    # Find the previous latents from the current latents
    # From https://github.com/smhongok/inv-dpm
    # @torch.inference_mode()
    def fixedpoint_correction(self, index,x, s, t, x_t, r=None, order=1, n_iter=40, step_size=0.1, th=2e-3,photo=None,mask=None,
                                model_s_output=None, model_r_output=None, latents_0=None,image_latents=None, prompt_embeds=None, guidance_scale=0.0, image_guidance_scale=0.0,guidance_rescale=0.0,
                                scheduler=False, factor=0.5, patience=20, anchor=False, warmup=True, warmup_time=20,**extra_step_kwargs):
        do_classifier_free_guidance = (guidance_scale >= 1.0 and image_guidance_scale >= 1.0)
        input_mean = self.scheduler.add_noise(latents_0, torch.zeros_like(latents_0), s)
        input=x.detach().clone()
        input.requires_grad_(True)
        noise_optimizer=torch.optim.Adam([input], lr=step_size)

        lambda_s, lambda_t = self.scheduler.lambda_t[s], self.scheduler.lambda_t[t]
        sigma_s, sigma_t = self.scheduler.sigma_t[s], self.scheduler.sigma_t[t]
        h = lambda_t - lambda_s
        alpha_s, alpha_t = self.scheduler.alpha_t[s], self.scheduler.alpha_t[t]
        phi_1 = torch.expm1(-h)

        original_step_size = step_size

        # step size scheduler, reduce when not improved
        if scheduler:
            step_scheduler = StepScheduler(current_lr=step_size, factor=factor, patience=patience)

        for i in range(n_iter):
            # step size warmup
            if warmup:
                if i < warmup_time:
                    step_size = original_step_size * (i+1)/(warmup_time)

            # Expand the latents if we are doing classifier free guidance.
            # The latents are expanded 3 times because for pix2pix the guidance\
            # is applied for both the text and the input image.

            latent_model_input = (torch.cat([input] * 3) if do_classifier_free_guidance else input)

            # concat latents, image_latents in the channel dimension
            scaled_latent_model_input = self.scheduler.scale_model_input(latent_model_input, s)
            scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

            # predict the noise residual
            noise_pred = self.unet(
                scaled_latent_model_input,
                s,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                (
                    noise_pred_text,
                    noise_pred_image,
                    noise_pred_uncond,
                ) = noise_pred.chunk(3)
                noise_pred = (
                    noise_pred_uncond
                    + guidance_scale * (noise_pred_text - noise_pred_image)
                    + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                )

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            x_t_pred=self.scheduler.step(noise_pred, s, input, **extra_step_kwargs).prev_sample
            # forward step method
            input.data = input.data - step_size * (x_t_pred - x_t)

            # model_output = self.scheduler.convert_model_output(noise_pred, s, input).detach()
            # input.data = (x_t+(alpha_t * phi_1 ) * model_output)/(sigma_t / sigma_s)

            n_loss = torch.nn.functional.mse_loss(x_t_pred.detach(), x_t.detach(), reduction='sum') #
            residual=input-input_mean
            # if index==0:
            #     loss=torch.nn.functional.mse_loss(self.vae.decode(x_t_pred/ self.vae.config.scaling_factor, return_dict=False)[0], photo.detach(), reduction='sum') #

            # loss.backward()
            # noise_optimizer.step()
            # noise_optimizer.zero_grad()
            # noise_normalize_(residual)
            # input.data=input_mean+residual

            if i % 10 == 0:
                # free, total = torch.cuda.mem_get_info()
                # mem_used_MB = (total - free) / 1024 ** 2
                print(f"Loss: {n_loss.item()}") #, Memory used: {mem_used_MB} MB
            if n_loss.item() < th:
                print(f"Loss: {n_loss.item()}")
                break

            # residual_sqr = ((input-input_mean)**2).detach()
            # residual_sqr/=residual_sqr.max()
            # input.data += torch.randn_like(input.data) * step_size * 0.005*(1-residual_sqr)

            if scheduler:
                step_size = step_scheduler.step(n_loss)
                noise_optimizer.param_groups[0]['lr'] = step_size

        # model_output = self.scheduler.convert_model_output(noise_pred, s, input).detach()
        # x_t_pred = (sigma_t / sigma_s) * input - (alpha_t * phi_1 ) * model_output

        return input.detach(),input-input_mean


    def decoder_inv(self, x,z_int,mask=1.):
        with torch.enable_grad():
            """
            decoder_inv calculates latents z of the image x by solving optimization problem ||E(x)-z||,
            not by directly encoding with VAE encoder. "Decoder inversion"

            INPUT
            x : image data (1, 3, 512, 512)
            OUTPUT
            z : modified latent data (1, 4, 64, 64)

            Goal : minimize norm(e(x)-z)
            """
            input = x.clone().float()

            z = z_int.detach().clone().float()
            z.requires_grad_(True)

            loss_function = torch.nn.MSELoss(reduction='sum')
            optimizer = torch.optim.Adam([z], lr=1e-1)
            lr_scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=10, num_training_steps=200)

            for i in self.progress_bar(range(100)):
                x_pred = self.vae.decode(z/ self.vae.config.scaling_factor, return_dict=False)[0]

                loss = loss_function(x_pred*mask, input*mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            return z

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
) -> LambdaLR:
    """
    Create a learning rate schedule that linearly increases the learning rate from
    0.0 to lr over num_warmup_steps, then decreases to 0.0 on a cosine schedule over
    the remaining num_training_steps-num_warmup_steps (assuming num_cycles = 0.5).

    This is based on the Hugging Face implementation
    https://github.com/huggingface/transformers/blob/v4.23.1/src/transformers/optimization.py#L104.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to
            schedule the learning rate.
        num_warmup_steps (int): The number of steps for the warmup phase.
        num_training_steps (int): The total number of training steps.
        num_cycles (float): The number of waves in the cosine schedule. Defaults to 0.5
            (decrease from the max value to 0 following a half-cosine).
        last_epoch (int): The index of the last epoch when resuming training. Defaults to -1

    Returns:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule.
    """

    def lr_lambda(current_step: int) -> float:
        # linear warmup phase
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)

        # cosine
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )

        cosine_lr_multiple = 0.5 * (
            1.0 + math.cos(math.pi * num_cycles * 2.0 * progress)
        )
        return max(0.0, cosine_lr_multiple)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class StepScheduler(ReduceLROnPlateau):
    def __init__(self, mode='min', current_lr=0, factor=0.1, patience=10,
                threshold=1e-4, threshold_mode='rel', cooldown=0,
                min_lr=0, eps=1e-8, verbose=False):
        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor
        if current_lr == 0:
            raise ValueError('Step size cannot be 0')

        self.min_lr = min_lr
        self.current_lr = current_lr
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self._init_is_better(mode=mode, threshold=threshold,
                            threshold_mode=threshold_mode)
        self._reset()

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            import warnings
            warnings.warn("EPOCH_DEPRECATION_WARNING", UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        return self.current_lr

    def _reduce_lr(self, epoch):
        old_lr = self.current_lr
        new_lr = max(self.current_lr * self.factor, self.min_lr)
        if old_lr - new_lr > self.eps:
            self.current_lr = new_lr
            if self.verbose:
                epoch_str = ("%.2f" if isinstance(epoch, float) else
                            "%.5d") % epoch
                print('Epoch {}: reducing learning rate'
                        ' to {:.4e}.'.format(epoch_str,new_lr))




def train_text(photo_latents, image_latents,irradiance_text_latent, encoder_hidden_states,rgb2x_prompt_embeds,preprocessed_aovs,
               unet=None, noise_scheduler=None, augtext=True,transferweight=1.0,originweight=1.0,steps=200, text_lr=1e-1, progress=tqdm,traintext=False):
    # initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        # mixed_precision='fp16'
    )
    set_seed(0)

    # initialize the model
    if noise_scheduler is None:
        noise_scheduler = DDIMScheduler.from_pretrained("zheng95z/x-to-rgb", subfolder="scheduler")
        noise_scheduler = DDIMScheduler.from_config(noise_scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing")
        noise_scheduler.set_timesteps(2)
        rgb2x_noise_scheduler = DDIMScheduler.from_pretrained("zheng95z/rgb-to-x", subfolder="scheduler")
        rgb2x_noise_scheduler = DDIMScheduler.from_config(rgb2x_noise_scheduler.config, rescale_betas_zero_snr=True, timestep_spacing="trailing")
        rgb2x_noise_scheduler.set_timesteps(2)

    timesteps = noise_scheduler.timesteps
    rgb2x_timesteps = rgb2x_noise_scheduler.timesteps

    unet = UNet2DConditionModel.from_pretrained("zheng95z/x-to-rgb", subfolder="unet", revision=None)
    rgb2x_unet=UNet2DConditionModel.from_pretrained("zheng95z/rgb-to-x", subfolder="unet", revision=None)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    unet.requires_grad_(False)
    unet.to(device)
    rgb2x_unet.requires_grad_(False)
    rgb2x_unet.to(device)

    description = f"Training "

    # Optimizer creation
    model_input = photo_latents.detach().clone()
    bsz, channels, height, width = model_input.shape
    # if image_latents.shape[0] != photo_latents.shape[0]:
    cond_image_latents = image_latents[:1].detach().clone()
    uncond_image_latents = image_latents[-1:].detach().clone()

    image_latents=cond_image_latents
    encoder_hidden_states = encoder_hidden_states.detach().clone()
    encoder_hidden_states_init = encoder_hidden_states.detach().clone()

    if traintext:
        text_lr=text_lr
        description += f"prompt, lr={text_lr}"
        encoder_hidden_states.requires_grad = True
        text_optimizer = torch.optim.AdamW(
            [encoder_hidden_states],
            lr=text_lr,
            betas=(0.9, 0.999),
            # weight_decay=1e-2,
            eps=1e-08,
        )

    noise_init = torch.randn_like(model_input)
    noise= noise_init.detach().clone()
    rgb2x_noise=noise_init.detach().clone()

    l1_loss = nn.L1Loss()
    step_size = 0.01
    step_scheduler = StepScheduler(current_lr=step_size, factor=0.5, patience=20)

    required_aovs = ["albedo", "normal", "roughness", "metallic"]
    image_latents_aug = image_latents.detach().clone()

    # If the AOVs are not provided, we will estimate them
    if augtext:
        for c in range(4):
            aov_name = required_aovs[c]
            if preprocessed_aovs[aov_name] is None:
                print(aov_name)
                aov_input = torch.randn_like(rgb2x_noise)
                for i, t in enumerate(rgb2x_timesteps):
                    scaled_aov_input = rgb2x_noise_scheduler.scale_model_input(aov_input, t)
                    aov_concatenated_noisy_latents = torch.cat([scaled_aov_input, model_input], dim=1)
                    aov_pred = rgb2x_unet(aov_concatenated_noisy_latents, t, rgb2x_prompt_embeds[c].detach(), return_dict=False)[0]
                    aov_input = rgb2x_noise_scheduler.step(aov_pred, t, aov_input, return_dict=False)[0]

                image_latents_aug[:, 4 * c:4 * c + 4] = aov_input

    image_latents_text_aug = torch.cat([image_latents_aug[:, :-3], irradiance_text_latent], dim=1)

    # Prompt optimization loop
    for _ in progress.tqdm(range(steps), desc=description):
        loss = 0
        x0_preds = []

        # Sample a random timestep for each image
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=noise.device)
        t = t.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        noise_=torch.randn_like(noise)
        noisy_model_input = noise_scheduler.add_noise(model_input, noise_, t)

        concatenated_noisy_latents = torch.cat([noisy_model_input, image_latents], dim=1)
        model_pred = unet(concatenated_noisy_latents, t, encoder_hidden_states).sample

        # Get the image for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(model_input, noise_, t)
        else:
            raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

        # Tuning loss in Eq. 6
        loss += F.mse_loss(model_pred.float(),target.float(),reduction="mean")*float(originweight)
        # Transfer loss in Eq. 7
        if traintext:
            concatenated_noisy_latents_text = torch.cat([noisy_model_input, image_latents_text_aug], dim=1)
            # Input with null text embeddings+ all the aov channels
            model_pred_text = unet(concatenated_noisy_latents_text, t, encoder_hidden_states_init).sample
            loss += F.mse_loss(model_pred_text, model_pred, reduction="mean")*float(transferweight)

        accelerator.backward(loss)

        if traintext:
            text_optimizer.step()
            text_optimizer.zero_grad()

    return encoder_hidden_states
