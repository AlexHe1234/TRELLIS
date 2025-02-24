"""Generate texture for a given mesh by running slat model only
"""

import os
import hashlib
from typing import List

import cv2
import torch
import imageio
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from diffusers import UniPCMultistepScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

import utils3d
from mesh_preproc import process_mesh
from trellis.utils import render_utils, postprocessing_utils
from trellis.pipelines.trellis_image_to_3d import TrellisImageTo3DPipeline


class ControlNet:
    def __init__(self, variant: str = 'canny'):
        self.type = type
        
        self.controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-" + variant, torch_dtype=torch.float16)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=self.controlnet, torch_dtype=torch.float16
        )
        
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_xformers_memory_efficient_attention()

    def run_controlnet_depth(
        self,
        image: np.ndarray, 
        prompt: List[str],
        seed: int = 42, 
    ):
        # image: HxW depth image
        image = (255 * (image - image.min()) / (image.max() - image.min())).astype(np.uint8)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)
        
        prompt = [p + ", best quality, realistic, extremely detailed" for p in prompt]

        generator = [torch.Generator(device="cpu").manual_seed(seed) for i in range(len(prompt))]

        output = self.pipe(
            prompt,
            image,
            negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(prompt),
            generator=generator,
            num_inference_steps=20,
        )
        
        image_gen = output['images']

        return np.array(image_gen)

    def run_controlnet_canny(
        self,
        image: np.ndarray, 
        prompt: List[str],
        seed: int = 42, 
        low_thresh: float = 100, 
        high_thresh: float = 200,
    ):
        # image: H, W, 3
        image = cv2.Canny(image, low_thresh, high_thresh)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        canny_image = Image.fromarray(image)

        prompt = [p + ", best quality, extremely detailed" for p in prompt]

        generator = [torch.Generator(device="cpu").manual_seed(seed) for i in range(len(prompt))]

        output = self.pipe(
            prompt,
            canny_image,
            negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * len(prompt),
            generator=generator,
            num_inference_steps=20,
        )
        
        image_gen = output['images']

        return np.array(image_gen)
    
    def re_load(self):
        self.pipe.to('cuda')
    
    def offload(self):
        self.pipe.to('cpu')


def compute_sha1(file_path):
    hasher = hashlib.sha1()
    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_voxels(output_dir, instance, resolution: int = 64):
    position = utils3d.io.read_ply(os.path.join(output_dir, 'voxels', f'{instance}.ply'))[0]
    coords = ((torch.tensor(position) + 0.5) * resolution).int().contiguous()
    ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    return ss


@torch.no_grad()
def run_trellis_texture(pipeline, image, voxels, seed=1):
    image = pipeline.preprocess_image(image)
    cond = pipeline.get_cond([image])
    torch.manual_seed(seed)
    coords = torch.argwhere(voxels > 0.5)[:, [0, 2, 3, 4]].int()
    slat = pipeline.sample_slat(cond, coords, {})
    outputs = pipeline.decode_slat(slat, ['mesh', 'gaussian', 'radiance_field'], )
    
    return outputs


def generate_texture(
    mesh_path: str, 
    output_dir: str, 
    obj_desc: str, 
    image_path: str = None, 
    mask_path: str = None, 
    skip_preproc: bool = False,
):
    """Generate texture using slat flow matcher from Trellis
    """
    
    obj_name = compute_sha1(mesh_path) # unique id to avoid cache confliction
    output_dir = os.path.join(output_dir, obj_name)
    cache_dir = 'data/Trellis_Cache/datasets/cache'
    os.makedirs(output_dir, exist_ok=True)
    
    if not skip_preproc: # preproc mesh
        ret = process_mesh(mesh_path, cache_dir, obj_name, verbal=True)
        assert ret['success']
    
    # controlnet canny
    if image_path is None:
        image_path = os.path.join(cache_dir, 'renders', obj_name, '000.png')
    
    image = cv2.imread(image_path)
    # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ctrl_net = ControlNet(variant='canny')    
    image_ctrlnet = ctrl_net.run_controlnet_canny(image, [obj_desc])[0] # h,w,3 in np
    ctrl_net.offload()
    
    plt.imsave(os.path.join(output_dir, 'control_net.png'), image_ctrlnet)
    
    # mask if apply
    image_mask = image_ctrlnet
    if mask_path is not None:
        image_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) > 128 # h,w of 0 and 255
        image_mask = image_ctrlnet * image_mask[..., None]
        plt.imsave(os.path.join(output_dir, 'control_net_masked.png'), image_mask)
    
    image_mask = Image.fromarray(image_mask)
    
    # generate trellis slat
    pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
    pipeline.cuda()
    
    voxels = get_voxels(cache_dir, obj_name, resolution=64)[None].cuda().float()
    outputs = run_trellis_texture(pipeline, image_mask, voxels, seed=1)

    # render video
    video = render_utils.render_video(outputs['gaussian'][0])['color']
    imageio.mimsave(os.path.join(output_dir, 'sample_gs.mp4'), video, fps=30)
    
    # save glb mesh
    glb = postprocessing_utils.to_glb(
        outputs['gaussian'][0],
        outputs['mesh'][0],
        simplify=0.95,          # Ratio of triangles to remove in the simplification process
        texture_size=1024,      # Size of the texture used for the GLB
    )
    glb.export(os.path.join(output_dir, 'sample.glb'))
    
    print(f'Output @ {output_dir}')
    
    
if __name__ == '__main__':
    mesh_path = 'data/Trellis_Cache/raw_mesh/foxYSY_Jump_00.obj'
    output_dir = 'data/Trellis_Cache/texture_gen'
    obj_desc = 'A fox'
    image_path = 'res_.png'
    mask_path = 'res1_.png' # mask corresponding to the image
    skip_preproc = True
    
    generate_texture(
        mesh_path=mesh_path, 
        output_dir=output_dir, 
        obj_desc=obj_desc, 
        image_path=image_path, 
        mask_path=mask_path, 
        skip_preproc=skip_preproc,
    )
    