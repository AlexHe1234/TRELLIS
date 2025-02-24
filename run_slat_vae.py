"""Inference structured latents vae.
"""

import os
import json
from typing import Literal, Dict, List

import torch
import imageio
import trimesh
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download

import trellis.models as models
import trellis.modules.sparse as sp
from trellis.utils.render_utils import *
from trellis.utils import render_utils, postprocessing_utils
from trellis.representations.mesh.cube2mesh import MeshExtractResult


def inference_slat_vae(
    output_dir: str, 
    obj_name: str, 
    save_path: str,
    decode_format: Literal['mesh', 'gaussian', 'radiance_field'] = 'mesh',
):
    
    assert decode_format in ['mesh', 'gaussian', 'radiance_field'], \
        f'Unsupported decode format {decode_format}'
    
    torch.set_grad_enabled(False)
    
    feat_model = 'dinov2_vitl14_reg'
    enc_pretrained = 'JeffreyXiang/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16'
    latent_name = f'{feat_model}_{enc_pretrained.split("/")[-1]}'
    encoder = models.from_pretrained(enc_pretrained).eval().cuda()                

    os.makedirs(os.path.join(output_dir, 'latents', latent_name), exist_ok=True)

    feats = np.load(os.path.join(output_dir, 'features', feat_model, f'{obj_name}.npz'))
    feats = sp.SparseTensor(
        feats = torch.from_numpy(feats['patchtokens']).float(),
        coords = torch.cat([
            torch.zeros(feats['patchtokens'].shape[0], 1).int(),
            torch.from_numpy(feats['indices']).int(),
        ], dim=1),
    ).cuda()
    
    latent = encoder(feats, sample_posterior=False) # is this the latent used in decoding
    
    assert torch.isfinite(latent.feats).all(), "Non-finite latent"
    pack = {
        'feats': latent.feats.cpu().numpy().astype(np.float32), # n,c=8
        'coords': latent.coords[:, 1:].cpu().numpy().astype(np.uint8), # n,3
    }

    pipeline_path = "JeffreyXiang/TRELLIS-image-large"
    is_local = os.path.exists(f"{pipeline_path}/pipeline.json")
    if is_local:
        config_file = f"{path}/pipeline.json"
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(pipeline_path, "pipeline.json")

    with open(config_file, 'r') as f:
        model_args = json.load(f)['args']

    # initalize slat_decoder
    if decode_format == 'mesh':
        value = model_args['models']['slat_decoder_mesh']
    elif decode_format == 'gaussian':
        value = model_args['models']['slat_decoder_gs']
    else:
        value = model_args['models']['slat_decoder_rf']
        
    decoder = models.from_pretrained(f'{pipeline_path}/{value}').cuda()
    outputs = decoder(latent)[0]
    
    if decode_format == 'mesh':
        vertices, faces = outputs.vertices.cpu().numpy(), outputs.faces.cpu().numpy()
        mesh = trimesh.Trimesh(vertices, faces)
        mesh.export(save_path)
    else:
        raise NotImplementedError()
    
    
def render_video_pair(src_mesh_path, rec_mesh_path):
    mesh_src = trimesh.load(src_mesh_path)
    mesh_rec = trimesh.load(rec_mesh_path)
    
    mesh_src_ = MeshExtractResult(
        torch.from_numpy(mesh_src.vertices).cuda().float(),
        torch.from_numpy(mesh_src.faces).cuda().long(),
    )
    mesh_rec_ = MeshExtractResult(
        torch.from_numpy(mesh_rec.vertices).cuda().float(),
        torch.from_numpy(mesh_rec.faces).cuda().long(),
    )
    
    video_src = render_utils.render_video(mesh_src_)['normal']
    video_rec = render_utils.render_video(mesh_rec_)['normal']

    return video_src, video_rec


def run_single(
    output_dir: str,
    save_path: str,
    obj_name: str,
    video_src_path: str,
    video_rec_path: str,
):
    inference_slat_vae(output_dir, obj_name, save_path)
    
    src_mesh_path = os.path.join(output_dir, 'renders', obj_name, 'mesh.ply')
    rec_mesh_path = os.path.join(save_path)
    
    video_src, video_rec = render_video_pair(src_mesh_path, rec_mesh_path)
    imageio.mimsave(video_src_path, video_src, fps=30)
    imageio.mimsave(video_rec_path, video_rec, fps=30)
    
    print('Done. Videos saved.')
    
    
def render_seq_video(
    samples: List[MeshExtractResult], 
    resolution=512, 
    bg_color=(0, 0, 0), 
    num_frames=300, r=2, fov=40, **kwargs,
):
    # fixed yaw, pitch
    yaws = [-3.14159 / 2.] # left and right
    pitch = [0] # up and down
    extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitch, r, fov)
    
    frames = []
    for f, sample in enumerate(tqdm(samples, desc='Rendering...')):
        ret = render_frames(
            sample, 
            extrinsics, 
            intrinsics, 
            {'resolution': resolution, 'bg_color': bg_color}, 
            **kwargs,
        )['normal'][0] # H, W, 3
        frames.append(ret)
    return frames
    
    
def run_multi(
    working_dir: str, 
    obj_name_kw: str, 
    save_meshes_dir: str = None, 
    save_video_path: str = None,
    save_video_gt_path: str = None,
):
    obj_names = sorted([f for f in os.listdir(os.path.join(working_dir, 'renders')) \
        if obj_name_kw in f])
    
    # TODO:
    # if save_meshes_dir is not None:
    #     os.makedirs(save_meshes_dir, exist_ok=True)
    #     for i, obj_name in enumerate(tqdm(obj_names, desc='Inferencing...')):
    #         inference_slat_vae(working_dir, obj_name, os.path.join(save_meshes_dir, f'{i:06d}.obj'))
    
    if  save_meshes_dir is not None and save_video_path is not None: # assumes meshes are already there
        
        # read objects
        def load_mesh(obj_paths):
            objs = [trimesh.load(obj_path) for obj_path in tqdm(obj_paths, desc='Loading...')]
            meshes = [
                MeshExtractResult(
                    torch.from_numpy(obj.vertices).cuda().float(), 
                    torch.from_numpy(obj.faces).cuda().long(),
                ) for obj in tqdm(objs, desc='Converting...')
            ]
            return meshes
        
        obj_paths = sorted([os.path.join(save_meshes_dir, n) for n in os.listdir(save_meshes_dir) if '.obj' in n]) # saved meshes
        gt_paths = [os.path.join(working_dir, 'renders', obj_name, 'mesh.ply') for obj_name in obj_names]
        
        save_meshes = load_mesh(obj_paths)
        gt_meshes = load_mesh(gt_paths)
        
        video_save = render_seq_video(save_meshes)
        video_gt = render_seq_video(gt_meshes)
        
        imageio.mimsave(save_video_path, video_save, fps=6)
        imageio.mimsave(save_video_gt_path, video_gt, fps=6)
        
        print('Saved video.')


if __name__ == '__main__':
    
    # # * single object example
    # output_dir = 'datasets/dt4d'
    # save_path = 'slat_vae_out_004.obj'
    # obj_name = 'tiger-03'
    # video_src_path = 'slat_src.mp4'
    # video_rec_path = 'slat_rec.mp4'
    # run_single(output_dir, save_path, obj_name, video_src_path, video_rec_path)
    
    # * sequence example, save single-view video instead of 360 video
    working_dir = './data/Trellis_Cache/datasets/dt4d' # dataset root
    obj_name_kw = 'deer' # all objects under the dataset root with this kw will be sorted and processed
    save_meshes_dir = './deer' # if None no mesh output
    save_video_path = './deer_seq.mp4' # if None no video
    save_video_gt_path = './deer_gt.mp4' # gt video path
    run_multi(working_dir, obj_name_kw, save_meshes_dir, save_video_path, save_video_gt_path)
