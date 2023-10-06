from pytorch_lightning import seed_everything

from torch import autocast
import math
import os
from typing import List, Union

import numpy as np
import torch
from einops import rearrange, repeat
from imwatermark import WatermarkEncoder
from omegaconf import ListConfig, OmegaConf
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torch import autocast
from torchvision import transforms
from torchvision.utils import make_grid

from scripts.util.detection.nsfw_and_watermark_dectection import DeepFloydDataFiltering
from sgm.modules.diffusionmodules.sampling import (
    DPMPP2MSampler,
    DPMPP2SAncestralSampler,
    EulerAncestralSampler,
    EulerEDMSampler,
    HeunEDMSampler,
    LinearMultistepSampler,
)
from sgm.util import append_dims, instantiate_from_config






SAVE_PATH = "outputs/demo/txt2img/"
SEED = 1234
WORKER_JOB_TYPE = "stable_diffusion_xl_txt2img"
WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d7"

# A fixed 48-bit message that was choosen at random
# WATERMARK_MESSAGE = 0xB3EC907BB19E
WATERMARK_MESSAGE = 0b101100111110110010010000011110111011000110011110
# bin(x)[2:] gives bits of x as str, use int to convert them to 0/1
WATERMARK_BITS = [int(bit) for bit in bin(WATERMARK_MESSAGE)[2:]]
EMBED_WATERMARK = WatermarkEmbedder(WATERMARK_BITS)

SD_XL_BASE_RATIOS = {
    "0.5": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.6": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.0": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "1.91": (1344, 704),
    "2.0": (1408, 704),
    "2.09": (1472, 704),
    "2.4": (1536, 640),
    "2.5": (1600, 640),
    "2.89": (1664, 576),
    "3.0": (1728, 576),
}

VERSION2SPECS = {
    "SDXL-base-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_1.0.safetensors",
    },
    "SDXL-base-0.9": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_base_0.9.safetensors",
    },
    "SD-2.1": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_2_1.yaml",
        "ckpt": "checkpoints/v2-1_512-ema-pruned.safetensors",
    },
    "SD-2.1-768": {
        "H": 768,
        "W": 768,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_2_1_768.yaml",
        "ckpt": "checkpoints/v2-1_768-ema-pruned.safetensors",
    },
    "SDXL-refiner-0.9": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_xl_refiner.yaml",
        "ckpt": "checkpoints/sd_xl_refiner_0.9.safetensors",
    },
    "SDXL-refiner-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": True,
        "config": "configs/inference/sd_xl_refiner.yaml",
        "ckpt": "checkpoints/sd_xl_refiner_1.0.safetensors",
    },
}


def load_img(display=True, key=None, device="cuda"):
    image = get_interactive_image(key=key)
    if image is None:
        return None
    if display:
        st.image(image)
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    width, height = map(
        lambda x: x - x % 64, (w, h)
    )  # resize to integer multiple of 64
    image = image.resize((width, height))
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    return image.to(device)


def run_txt2img(
    model,
    version,
    version_dict,
    params,
    is_legacy=False,
    filter=None,

):

    C = version_dict["C"]
    F = version_dict["f"]

    init_dict = {
        "orig_width": params['width'],
        "orig_height": params['height'],
        "target_width": params['width'],
        "target_height": params['height'],
    }
    value_dict = init_embedder_options(
        get_unique_embedder_keys_from_conditioner(model.conditioner),
        init_dict,
        params
    )
    sampler, num_rows, num_cols = init_sampling(params)

    out = do_sample(
        model,
        sampler,
        value_dict,
        params,
        C,
        F,
        force_uc_zero_embeddings=["txt"] if not is_legacy else [],
        return_latents=params['return_latents'],
        filter=filter,
    )
        return out


def run_img2img(
    state,
    version_dict,
    is_legacy=False,
    return_latents=False,
    filter=None,
    stage2strength=None,
    params
):
    img = load_img()
    if img is None:
        return None
    H, W = img.shape[2], img.shape[3]

    init_dict = {
        "orig_width": W,
        "orig_height": H,
        "target_width": W,
        "target_height": H,
    }
    value_dict = init_embedder_options(
        get_unique_embedder_keys_from_conditioner(state["model"].conditioner),
        init_dict,
        params
    )
    strength = st.number_input(
        "**Img2Img Strength**", value=0.75, min_value=0.0, max_value=1.0
    )
    sampler, num_rows, num_cols = init_sampling(
        img2img_strength=strength,
        stage2strength=stage2strength,
    )
    num_samples = num_rows * num_cols

    if st.button("Sample"):
        out = do_img2img(
            repeat(img, "1 ... -> n ...", n=num_samples),
            state["model"],
            sampler,
            value_dict,
            num_samples,
            force_uc_zero_embeddings=["txt"] if not is_legacy else [],
            return_latents=return_latents,
            filter=filter,
        )
        return out


def apply_refiner(
    input,
    model,
    sampler,
    num_samples,
    prompt,
    negative_prompt,
    filter=None,
    finish_denoising=False,

):
    init_dict = {
        "orig_width": input.shape[3] * 8,
        "orig_height": input.shape[2] * 8,
        "target_width": input.shape[3] * 8,
        "target_height": input.shape[2] * 8,
    }

    value_dict = init_dict
    value_dict["prompt"] = prompt
    value_dict["negative_prompt"] = negative_prompt

    value_dict["crop_coords_top"] = 0
    value_dict["crop_coords_left"] = 0

    value_dict["aesthetic_score"] = 6.0
    value_dict["negative_aesthetic_score"] = 2.5

    samples = do_img2img(
        input,
        model,
        sampler,
        value_dict,
        skip_encode=True,
        add_noise=not finish_denoising,
    )

    return samples




def get_sampler(discretization_config, guider_config, params):
    sampler_name = params['sampler_name']
    steps = params['steps']
    s_churn = params['s_churn']
    s_tmin = params['s_tmin']
    s_tmax = params['s_tmax']
    s_noise = params['s_noise']
    eta = params['eta']
    order = params['order']
    if sampler_name == "EulerEDMSampler" or sampler_name == "HeunEDMSampler":

        if sampler_name == "EulerEDMSampler":
            sampler = EulerEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
        elif sampler_name == "HeunEDMSampler":
            sampler = HeunEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
    elif (
        sampler_name == "EulerAncestralSampler"
        or sampler_name == "DPMPP2SAncestralSampler"
    ):

        if sampler_name == "EulerAncestralSampler":
            sampler = EulerAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=True,
            )
        elif sampler_name == "DPMPP2SAncestralSampler":
            sampler = DPMPP2SAncestralSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                eta=eta,
                s_noise=s_noise,
                verbose=True,
            )
    elif sampler_name == "DPMPP2MSampler":
        sampler = DPMPP2MSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    elif sampler_name == "LinearMultistepSampler":

        sampler = LinearMultistepSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            order=order,
            verbose=True,
        )
    else:
        raise ValueError(f"unknown sampler {sampler_name}!")

    return sampler

def init_sampling(params):
    img2img_strength = params['img2img_strength']
    specify_num_samples = params['specify_num_samples']
    stage2strength = params['stage2strength']
    steps = params['steps']
    sampler_name = params['sampler_name']
    discretization = params['discretization']
    sigma_min = params['sigma_min']
    sigma_max = params['sigma_max']
    rho = params['rho']
    guider = params['guider']
    scale = params['scale']
    thresholder = params['thresholder']
    num_cols = params['num_cols']


    num_rows = 1
    if specify_num_samples:
        num_cols = 1

    discretization_config = get_discretization(params)

    guider_config = get_guider(params)

    sampler = get_sampler(discretization_config, guider_config, params)
    if img2img_strength < 1.0:
        sampler.discretization = Img2ImgDiscretizationWrapper(
            sampler.discretization, strength=img2img_strength
        )
    if stage2strength is not None:
        sampler.discretization = Txt2NoisyDiscretizationWrapper(
            sampler.discretization, strength=stage2strength, original_steps=steps
        )
    return sampler, num_rows, num_cols

def get_discretization(params):

    if params['discretization'] == "LegacyDDPMDiscretization":
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
        }
    elif params['discretization'] == "EDMDiscretization":
        discretization_config = {
            "target": "sgm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": params['sigma_min'],
                "sigma_max": params['sigma_max'],
                "rho": params['rho'],
            },
        }

    return discretization_config

def get_guider(params):

    if params['guider'] == "IdentityGuider":
        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"
        }
    elif params['guider'] == "VanillaCFG":


        if params['thresholder'] == "None":
            dyn_thresh_config = {
                "target": "sgm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"
            }
        else:
            raise NotImplementedError

        guider_config = {
            "target": "sgm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {"scale": params['scale'], "dyn_thresh_config": dyn_thresh_config},
        }
    else:
        raise NotImplementedError
    return guider_config


def do_sample(
    model,
    sampler,
    value_dict,
    params,
    C,
    F,
    force_uc_zero_embeddings: List = None,
    batch2model_input: List = None,
    return_latents=False,
    filter=None,
):
    num_samples = params['num_cols']
    if force_uc_zero_embeddings is None:
        force_uc_zero_embeddings = []
    if batch2model_input is None:
        batch2model_input = []

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                num_samples = [num_samples]
                load_model(model.conditioner)
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    num_samples,
                )
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        print(key, batch[key].shape)
                    elif isinstance(batch[key], list):
                        print(key, [len(l) for l in batch[key]])
                    else:
                        print(key, batch[key])
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )
                unload_model(model.conditioner)

                for k in c:
                    if not k == "crossattn":
                        c[k], uc[k] = map(
                            lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                        )

                additional_model_inputs = {}
                for k in batch2model_input:
                    additional_model_inputs[k] = batch[k]

                shape = (math.prod(num_samples), C, H // F, W // F)
                randn = torch.randn(shape).to("cuda")

                def denoiser(input, sigma, c):
                    return model.denoiser(
                        model.model, input, sigma, c, **additional_model_inputs
                    )

                load_model(model.denoiser)
                load_model(model.model)
                samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                unload_model(model.model)
                unload_model(model.denoiser)

                load_model(model.first_stage_model)
                samples_x = model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                unload_model(model.first_stage_model)

                if filter is not None:
                    samples = filter(samples)


                if return_latents:
                    return samples, samples_z
                return samples

def get_batch(keys, value_dict, N: Union[List, ListConfig], device="cuda"):
    # Hardcoded demo setups; might undergo some changes in the future

    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = (
                np.repeat([value_dict["prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
            batch_uc["txt"] = (
                np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(*N, 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        else:
            batch[key] = value_dict[key]

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc


@torch.no_grad()
def do_img2img(
    img,
    model,
    sampler,
    value_dict,
    force_uc_zero_embeddings=[],
    additional_kwargs={},
    offset_noise_level: int = 0.0,
    return_latents=False,
    skip_encode=False,
    filter=None,
    add_noise=True,
):

    precision_scope = autocast
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                load_model(model.conditioner)
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model.conditioner),
                    value_dict,
                    [num_samples],
                )
                c, uc = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=force_uc_zero_embeddings,
                )
                unload_model(model.conditioner)
                for k in c:
                    c[k], uc[k] = map(lambda y: y[k][:num_samples].to("cuda"), (c, uc))

                for k in additional_kwargs:
                    c[k] = uc[k] = additional_kwargs[k]
                if skip_encode:
                    z = img
                else:
                    load_model(model.first_stage_model)
                    z = model.encode_first_stage(img)
                    unload_model(model.first_stage_model)

                noise = torch.randn_like(z)

                sigmas = sampler.discretization(sampler.num_steps).cuda()
                sigma = sigmas[0]

                if offset_noise_level > 0.0:
                    noise = noise + offset_noise_level * append_dims(
                        torch.randn(z.shape[0], device=z.device), z.ndim
                    )
                if add_noise:
                    noised_z = z + noise * append_dims(sigma, z.ndim).cuda()
                    noised_z = noised_z / torch.sqrt(
                        1.0 + sigmas[0] ** 2.0
                    )  # Note: hardcoded to DDPM-like scaling. need to generalize later.
                else:
                    noised_z = z / torch.sqrt(1.0 + sigmas[0] ** 2.0)

                def denoiser(x, sigma, c):
                    return model.denoiser(model.model, x, sigma, c)

                load_model(model.denoiser)
                load_model(model.model)
                samples_z = sampler(denoiser, noised_z, cond=c, uc=uc)
                unload_model(model.model)
                unload_model(model.denoiser)

                load_model(model.first_stage_model)
                samples_x = model.decode_first_stage(samples_z)
                unload_model(model.first_stage_model)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

                if filter is not None:
                    samples = filter(samples)

                if return_latents:
                    return samples, samples_z
                return samples


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def unload_model(model):
    global lowvram_mode
    if lowvram_mode:
        model.cpu()
        torch.cuda.empty_cache()


def init_embedder_options(keys, init_dict, params):
    # Hardcoded demo settings; might undergo some changes in the future

    value_dict = {}
    for key in keys:
        if key == "txt":
            value_dict["prompt"] = params['prompt']
            value_dict["negative_prompt"] = params['negative_prompt']

        if key == "original_size_as_tuple":
            value_dict["orig_width"] = params['orig_width']
            value_dict["orig_height"] = params['orig_height']

        if key == "crop_coords_top_left":

            value_dict["crop_coords_top"] = params['crop_coord_top']
            value_dict["crop_coords_left"] = params['crop_coord_left']

        if key == "aesthetic_score":
            value_dict["aesthetic_score"] = 6.0
            value_dict["negative_aesthetic_score"] = 2.5

        if key == "target_size_as_tuple":
            value_dict["target_width"] = init_dict["target_width"]
            value_dict["target_height"] = init_dict["target_height"]

    return value_dict


class ProcessOutputCallback():
    def __init__(self, local_rank, api_worker):
        self.local_rank = local_rank
        self.api_worker = api_worker
        self.job_data = None

    def process_output(self, output, info):
        if self.local_rank == 0:
            results = {'image': output, 'info':info}
            return self.api_worker.send_job_results(self.job_data, results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_server",
        type=str,
        default="http://0.0.0.0:7777",
        help="Address of the API server",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0, required=False,
        help="ID of the GPU to be used"
    )
    args = parser.parse_args()
    

    seed_everything(SEED)
    version_dict_base = VERSION2SPECS['SDXL-base-1.0']
    version_dict_refiner = VERSION2SPECS['SDXL-refiner-1.0']
    config_base = OmegaConf.load(version_dict_base['config'])
    config_refiner = OmegaConf.load(version_dict_base['config'])
    ckpt_base = version_dict_base['ckpt']
    ckpt_refiner = version_dict_refiner['ckpt']
    model_base, msg_base = load_model_from_config(config_base, ckpt_base)
    model_refiner, msg_refiner = load_model_from_config(config_refiner, ckpt_refiner)

    is_legacy = version_dict_base["is_legacy"]

    params_base = {
        'stage2strength': 0.15,
        'finish_denoising_base': False,
        'img2img_strength': 1.0,
        'specify_num_samples': True,
        'stage2strength': 0.15,
        'steps': 40,
        'sampler_name': "EulerEDMSampler",
        'discretization': "LegacyDDPMDiscretization",
        'sigma_min': 0.03,
        'sigma_max': 14.61,
        'rho': 3.0, 
        'guider': 'VanillaCFG',
        'scale': 5.0,
        'thresholder': 'None',
        'n_samples': 2,
        'width': 1024,
        'height': 1024,
        'return_latents': True
        }
    params_refiner = {
        'stage2strength': None,
        'img2img_strength': 0.15,
        'specify_num_samples': True,
        'steps': 40,
        'sampler_name': "EulerEDMSampler",
        'discretization': "LegacyDDPMDiscretization"
        'sigma_min': 0.03,
        'sigma_max': 14.61,
        'rho': 3.0, 
        'guider': 'VanillaCFG',
        'scale': 5.0,
        'thresholder': 'None',
        'n_samples': 2,
        's_churn': 0.0, 
        's_tmin': 0.0, 
        's_tmax': 999.0, 
        's_noise': 1.0, 
        'eta': 1.0, 
        'order': 4,
        'finish_denoising': False,
        'negative_prompt': ''
    }


    """
    st.number_input(
            "**Refinement strength**", value=0.15, min_value=0.0, max_value=1.0
        )
    """
    """
    strength = st.number_input(
        "**Img2Img Strength**", value=0.75, min_value=0.0, max_value=1.0
    )
    sampler = st.sidebar.selectbox(
        f"Sampler #{key}",
        [
            "EulerEDMSampler",
            "HeunEDMSampler",
            "EulerAncestralSampler",
            "DPMPP2SAncestralSampler",
            "DPMPP2MSampler",
            "LinearMultistepSampler",
        ],
        0,
    )
    steps = st.sidebar.number_input(
        f"steps #{key}", value=40, min_value=1, max_value=1000
    )
    discretization = st.sidebar.selectbox(
        f"Discretization #{key}",
        [
            "LegacyDDPMDiscretization",
            "EDMDiscretization",
        ],
    )


    sigma_min=0.03
    sigma_max=14.61
    rho=3.0
        guider = st.sidebar.selectbox(
        f"Discretization #{key}",
        [
            "VanillaCFG",
            "IdentityGuider",
        ],
    )
        scale = st.number_input(
            f"cfg-scale #{key}", value=5.0, min_value=0.0, max_value=100.0
        )

        thresholder = st.sidebar.selectbox(
            f"Thresholder #{key}",
            [
                "None",
            ],
        )
    s_churn = st.sidebar.number_input(f"s_churn #{key}", value=0.0, min_value=0.0)
    s_tmin = st.sidebar.number_input(f"s_tmin #{key}", value=0.0, min_value=0.0)
    s_tmax = st.sidebar.number_input(f"s_tmax #{key}", value=999.0, min_value=0.0)
    s_noise = st.sidebar.number_input(f"s_noise #{key}", value=1.0, min_value=0.0)
    eta = st.sidebar.number_input("eta", value=1.0, min_value=0.0)
    order = st.sidebar.number_input("order", value=4, min_value=1)
    f"num cols #{key}", value=2, min_value=1, max_value=10
    if version.startswith("SDXL-base"):
        W, H = st.selectbox("Resolution:", list(SD_XL_BASE_RATIOS.values()), 10)
    else:
        H = st.number_input("H", value=version_dict["H"], min_value=64, max_value=2048)
        W = st.number_input("W", value=version_dict["W"], min_value=64, max_value=2048)
    
    """
    
    if not params_refiner['finish_denoising']:
        params_base['stage2strength'] = None


    sampler_refiner, *_ = init_sampling(params_refiner)
    if load_filter:
        filter = DeepFloydDataFiltering(verbose=False)
    else:
        filter = None

    local_rank, world_size = 0,1
    api_worker = APIWorkerInterface(args.api_server, WORKER_JOB_TYPE, WORKER_AUTH_KEY, args.gpu_id, world_size=world_size, rank=local_rank)
    callback = ProcessOutputCallback(local_rank, api_worker)
    progress_callback = ProgressCallback(api_worker)
    while True:
        prompt = []
        job_data = api_worker.job_request()
        callback.job_data = job_data
        progress_callback.job_data = job_data
        prompt = job_data['text']
        data = [params_base['n_samples'] * [prompt]]
        out = run_txt2img(
            model_base,
            version_base,
            version_dict_base,
            params_base,
            is_legacy=is_legacy,
            filter=filter
            
        )

        if isinstance(out, (tuple, list)):
            samples, samples_z = out
        else:
            samples = out
            samples_z = None

        if samples_z is not None:
            samples = apply_refiner(
                samples_z,
                model_refiner,
                sampler_refiner,
                samples_z.shape[0],
                prompt=prompt,
                negative_prompt=params_refiner['negative_prompt'] if is_legacy else "",
                filter=filter,
                finish_denoising=params_refiner['finish_denoising'],
            )
        samples = EMBED_WATERMARK(samples)
        for sample in samples:
            sample = 255.0 * rearrange(sample.cpu().numpy(), "c h w -> h w c")
            image = Image.fromarray(sample.astype(np.uint8))
            callback.process_output(image, f'Prompt: {data[0][0]}')

