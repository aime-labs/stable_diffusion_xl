from sgm.inference.api import SamplingParams, SamplingPipeline, ModelArchitecture
from sgm.inference.helpers import embed_watermark
import torch
import random
from einops import rearrange
import argparse
from PIL import Image, ExifTags
import numpy as np
import math
import sys
import datetime

from aime_api_worker_interface import APIWorkerInterface

WORKER_JOB_TYPE = "stable_diffusion_xl_txt2img"
DEFAULT_WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d7"
VERSION = 0
SEND_LESS_PREVIEWS = False

class ProcessOutputCallback():
    def __init__(self, api_worker, decode_first_stage):
        self.api_worker = api_worker
        self.decode_first_stage = decode_first_stage
        self.job_data = None
        self.total_steps = None
        self.current_step = None


    def process_output(self, output, finished):
        list_images = list()
        if output.get('images') is None:
            output['images'] = [Image.fromarray((np.random.rand(1024,1024,3) * 255).astype(np.uint8))]
            return self.api_worker.send_job_results(output)
        else:
            images = output.pop('images')
            if not finished:
                if self.api_worker.progress_data_received:
                    progress = self.calculate_progress(output)
                    preview_steps = self.get_preview_steps()
                    if self.job_data.get('provide_progress_images') == 'None' or self.current_step not in preview_steps:
                        return self.api_worker.send_progress(progress)

                    elif self.job_data.get('provide_progress_images') == 'decoded':
                        images = self.decode_first_stage(images)

                    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)

                    image_list = self.get_image_list(images)
                    output['progress_images'] = image_list
                    
                    return self.api_worker.send_progress(progress, output)

            else:
                images = self.decode_first_stage(images)
                images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                images = embed_watermark(images)
                image_list = self.get_image_list(images)

                output['images'] = image_list
                self.api_worker.send_progress(100, None)
                return self.api_worker.send_job_results(output)


    def get_image_list(self, images):
        image_list = list()
        for image in images:
            image = 255. * rearrange(image.cpu().numpy(), 'c h w -> h w c')
            image = Image.fromarray(image.astype(np.uint8))
            image_list.append(image)
        return image_list



    def calculate_progress(self, output):
        self.total_steps = self.job_data['base_steps'] + max(int(self.job_data['img2img_strength'] * self.job_data['refine_steps']), 1) + 1
        if output['stage'] == 'base':
            self.current_step = output.pop('progress')
        else:
            self.current_step = output.pop('progress') + self.job_data['base_steps']
        output.pop('stage')

        return round(self.current_step*100/ self.total_steps)


    def get_preview_steps(self):
        if SEND_LESS_PREVIEWS:
            def alpha(x):
                return math.log(0.5*x)
            norm_factor = self.total_steps/alpha(self.total_steps)
            preview_steps = set([round(norm_factor*alpha(step+1)) for step in range(self.total_steps)])
            preview_steps.add(1)
        else:
            preview_steps = [step for step in range(self.total_steps)]

        return preview_steps


def load_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_server", type=str, default="http://0.0.0.0:7777", help="Address of the AIME API server"
                        )
    parser.add_argument(
        "--gpu_id", type=int, default=0, required=False, help="ID of the GPU to be used"
                        )
    parser.add_argument(
        "--use_fp16", action='store_true', help="Use model in half precision"
                        )
    parser.add_argument(
        "--compile", action='store_true', help="Use torch.compile(model) from Pytorch 2"
                        )
    parser.add_argument(
        "--api_auth_key", type=str , default=DEFAULT_WORKER_AUTH_KEY, required=False, 
        help="API server worker auth key",
    )

    return parser.parse_args()


def set_seed(job_data):
    
    seed = job_data.get('seed', -1)
    if seed == -1:
        random.seed(datetime.datetime.now().timestamp())
        seed = random.randint(1, 99999999)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    job_data['seed'] = seed
    return job_data


def get_sampling_parameters(job_data, stage):
    params = {key:value for key, value in job_data.items() if hasattr(SamplingParams(), key)}
    for param_key in ['sampler', 'discretization', 'steps']:
        params[param_key] = job_data[f'{stage}_{param_key}']

    return SamplingParams(**params)


def main():
    args = load_flags()
    torch.cuda.set_device(args.gpu_id)
    api_worker = APIWorkerInterface(args.api_server, WORKER_JOB_TYPE, args.api_auth_key, args.gpu_id, world_size=1, rank=0, gpu_name=torch.cuda.get_device_name())
    pipeline_base = SamplingPipeline(ModelArchitecture.SDXL_V1_BASE, use_fp16=args.use_fp16, compile=args.compile)
    pipeline_refiner = SamplingPipeline(ModelArchitecture.SDXL_V1_REFINER, use_fp16=args.use_fp16, compile=args.compile)
    
    callback = ProcessOutputCallback(api_worker, pipeline_base.model.decode_first_stage)

    while True:
        try:
            job_data = api_worker.job_request()
            job_data = set_seed(job_data)
            
            callback.job_data = job_data           
            
            samples = pipeline_base.text_to_image(
                        params = get_sampling_parameters(job_data, 'base'),
                        prompt = job_data['prompt'],
                        negative_prompt = job_data['negative_prompt'],
                        samples = job_data['num_samples'],
                        progress_callback = callback.process_output
                    )          
            samples = pipeline_refiner.refiner(
                        params = get_sampling_parameters(job_data, 'refine'),
                        image = samples,
                        prompt = job_data['prompt'],
                        negative_prompt = job_data['negative_prompt'],
                        samples = job_data['num_samples'],
                        progress_callback = callback.process_output
                    )
            callback.process_output({'images': samples}, True)

        except ValueError as exc:
            callback.process_output({'error': f'{exc}\nChange parameters and try again'}, True)
            continue
        except torch.cuda.OutOfMemoryError as exc:
            callback.process_output({'error': f'{exc}\nReduce num_samples and try again'}, True)
            continue


if __name__ == "__main__":
    main()
