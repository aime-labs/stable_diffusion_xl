from sgm.inference.api import model_specs, SamplingParams, SamplingPipeline, Sampler, ModelArchitecture, Discretization, Guider
from sgm.inference.helpers import embed_watermark
import torch
import random
from einops import rearrange
import argparse
from PIL import Image
import numpy as np
from api_worker_interface import APIWorkerInterface, ProgressCallback

WORKER_JOB_TYPE = "stable_diffusion_xl_txt2img"
WORKER_AUTH_KEY = "5b07e305b50505ca2b3284b4ae5f65d7"

class ProcessOutputCallback():
    def __init__(self, local_rank, api_worker):
        self.local_rank = local_rank
        self.api_worker = api_worker
        self.job_data = None
        self.current_step = 0

    def process_output(self, output, finished):
        if self.local_rank == 0:
            images = torch.clamp((output.pop('image') + 1.0) / 2.0, min=0.0, max=1.0)
            if finished:
                images = embed_watermark(images)

            list_images = list()
            for sample in images:
                sample = 255. * rearrange(sample.cpu().numpy(), 'c h w -> h w c')
                list_images.append(Image.fromarray(sample.astype(np.uint8)))

            if finished:
                output['image'] = list_images[0]
                self.api_worker.send_progress(self.job_data, 100, None)
                return self.api_worker.send_job_results(self.job_data, output)
            else:
                output['progress_data'] = list_images[0]
                total_steps = self.job_data['base_steps'] + max(int(self.job_data['img2img_strength'] * self.job_data['refine_steps']), 1) + 1
                if output['stage'] == 'base':
                    progress = output['progress']*100/ total_steps
                else:
                    progress = (output['progress']+self.job_data['base_steps'])*100/ total_steps
                
                return self.api_worker.send_progress(self.job_data, progress, output)

def load_flags():
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
    parser.add_argument(
        "--use_fp16",
        action='store_true',
        help="Use model in half precision"
    )
    parser.add_argument(
        "--compile",
        action='store_true',
        help="Use torch.compile(model) from Pytorch 2"
    )


    
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_sampling_parameters(job_data, stage):
    params = {key:value for key, value in job_data.items() if hasattr(SamplingParams(), key)}
    params['sampler'] = job_data[f'{stage}_sampler']
    params['discretization'] =  job_data[f'{stage}_discretization']
    params['steps'] =  job_data[f'{stage}_steps']

    return SamplingParams(**params)


def main():
    args = load_flags()
    pipeline_base = SamplingPipeline(ModelArchitecture.SDXL_V1_BASE, use_fp16=args.use_fp16, compile=args.compile)
    pipeline_refiner = SamplingPipeline(ModelArchitecture.SDXL_V1_REFINER, use_fp16=args.use_fp16, compile=args.compile)
    world_size, local_rank = 1,0

    api_worker = APIWorkerInterface(args.api_server, WORKER_JOB_TYPE, WORKER_AUTH_KEY, args.gpu_id, world_size=world_size, rank=local_rank)
    callback = ProcessOutputCallback(local_rank, api_worker)

    while True:
        try:
            job_data = api_worker.job_request()
            total_steps = job_data['base_steps'] + job_data['refine_steps']
            set_seed(job_data.get('seed', 1234))
            callback.job_data = job_data           
            print('job_data: ', job_data)
            params_base = get_sampling_parameters(job_data, 'base')
            print('params_base: ', params_base)
            params_refine = get_sampling_parameters(job_data, 'refine')
            

            output = pipeline_base.text_to_image(
                        params = params_base,
                        prompt = job_data['text'],
                        negative_prompt = job_data.get('negative_prompt',''),
                        samples = 1,
                        return_latents = True,
                        progress_callback = callback.process_output
                    )
            
            _, samples_z = output
            samples = pipeline_refiner.refiner(
                        params = params_refine,
                        image = samples_z,
                        prompt = job_data['text'],
                        negative_prompt = job_data.get('negative_prompt',''),
                        samples = 1,
                        progress_callback = callback.process_output
                    )
            
            callback.process_output({'image': samples, 'info': f'Prompt: {job_data["text"]}'}, True)

        except (RuntimeError, ValueError) as exc:
            callback.process_output({'image': np.random.rand(1024,1024,3) * 255, 'info': exc}, True)
            continue
        


if __name__ == "__main__":
    main()