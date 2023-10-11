from sgm.inference.api import model_specs, SamplingParams, SamplingPipeline, Sampler, ModelArchitecture, Discretization, Guider
from sgm.inference.helpers import embed_watermark
import torch
import random
from einops import rearrange
import argparse
from PIL import Image
import numpy as np
import sys
from api_worker_interface import APIWorkerInterface, ProgressCallback

WORKER_JOB_TYPE = "stable_diffusion_xl_txt2img"
WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d7"


class ProcessOutputCallback():
    def __init__(self, api_worker, decode_first_stage):
        self.api_worker = api_worker
        self.job_data = None
        self.decode_first_stage = decode_first_stage


    def process_output(self, output, finished):
        list_images = list()
        if output.get('images') is None:
            output['images'] = [Image.fromarray((np.random.rand(1024,1024,3) * 255).astype(np.uint8))]
            return self.api_worker.send_job_results(self.job_data, output)
        else:
            images = output.pop('images')
            if not finished:
                progress = self.calculate_progress(output)
                if self.job_data.get('provide_progress_images') == 'None':
                    return self.api_worker.send_progress(self.job_data, progress, None)

                elif self.job_data.get('provide_progress_images') == 'decoded':
                    images = self.decode_first_stage(images)

                images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                for image in images:
                    image = 255. * rearrange(image.cpu().numpy(), 'c h w -> h w c')
                    list_images.append(Image.fromarray(image.astype(np.uint8)))

                output['progress_images'] = list_images
                return self.api_worker.send_progress(self.job_data, progress, output)

            else:
                images = self.decode_first_stage(images)
                images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                images = embed_watermark(images)
                for image in images:
                    image = 255. * rearrange(image.cpu().numpy(), 'c h w -> h w c')
                    list_images.append(Image.fromarray(image.astype(np.uint8)))

                output['images'] = list_images
                self.api_worker.send_progress(self.job_data, 100, None)
                return self.api_worker.send_job_results(self.job_data, output)


    def calculate_progress(self, output):
        total_steps = self.job_data['base_steps'] + max(int(self.job_data['img2img_strength'] * self.job_data['refine_steps']), 1) + 1
        if output['stage'] == 'base':
            return round(output['progress']*100/ total_steps, 1)
        else:
            return round((output['progress'] + self.job_data['base_steps'])*100/ total_steps, 1)


def load_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_server", type=str, default="http://0.0.0.0:7777", help="Address of the API server"
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
    return parser.parse_args()

def set_seed(job_data):
    seed = job_data.get('seed', 1234)
    if seed == -1:
        seed = random.randint(1, 99999999)
        job_data['seed'] = seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    return job_data


def get_sampling_parameters(job_data, stage):
    params = {key:value for key, value in job_data.items() if hasattr(SamplingParams(), key)}
    for param_key in ['sampler', 'discretization', 'steps']:
        params[param_key] = job_data[f'{stage}_{param_key}']

    return SamplingParams(**params)


def main():
    args = load_flags()
    pipeline_base = SamplingPipeline(ModelArchitecture.SDXL_V1_BASE, use_fp16=args.use_fp16, compile=args.compile)
    pipeline_refiner = SamplingPipeline(ModelArchitecture.SDXL_V1_REFINER, use_fp16=args.use_fp16, compile=args.compile)
    api_worker = APIWorkerInterface(args.api_server, WORKER_JOB_TYPE, WORKER_AUTH_KEY, args.gpu_id, world_size=1, rank=0)
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
            print('OOM')
            callback.process_output({'error': f'{exc}\nReduce num_samples and try again'}, True)
            continue
        


if __name__ == "__main__":
    main()