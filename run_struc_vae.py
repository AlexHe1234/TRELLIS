"""Inference sparse structure vae, output in the form of extracted mesh
"""

import os
import json
from typing import Literal, Dict, List

import torch
import mcubes
import imageio
import trimesh
import numpy as np
from huggingface_hub import hf_hub_download

import utils3d
import trellis.models as models
import trellis.modules.sparse as sp
from trellis.utils import render_utils, postprocessing_utils
from trellis.representations.mesh.cube2mesh import MeshExtractResult, SparseFeatures2Mesh


def inference_struc_vae(
    output_dir: str,
    obj_name: str,
    save_path: str,    
):
    torch.set_grad_enabled(False)
    
    enc_pretrained = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16'
    latent_name = f'{enc_pretrained.split("/")[-1]}'    
    encoder = models.from_pretrained(enc_pretrained).eval().cuda()
    
    os.makedirs(os.path.join(output_dir, 'ss_latents', latent_name), exist_ok=True)
    resolution = 64

    def get_voxels(instance):
        position = utils3d.io.read_ply(os.path.join(output_dir, 'voxels', f'{instance}.ply'))[0]
        coords = ((torch.tensor(position) + 0.5) * resolution).int().contiguous()
        ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
        ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        return ss
    
    ss = get_voxels(obj_name)[None].float()
    ss = ss.cuda().float()
    latent = encoder(ss, sample_posterior=False)

    # decoder
    pipeline_path = "JeffreyXiang/TRELLIS-image-large"
    is_local = os.path.exists(f"{pipeline_path}/pipeline.json")
    if is_local:
        config_file = f"{path}/pipeline.json"
    else:
        from huggingface_hub import hf_hub_download
        config_file = hf_hub_download(pipeline_path, "pipeline.json")

    with open(config_file, 'r') as f:
        model_args = json.load(f)['args']
    
    decoder = models.from_pretrained(f'{pipeline_path}/{model_args["models"]["sparse_structure_decoder"]}').cuda()
    
    output = decoder(latent)
    coords = torch.argwhere(output>0)[:, [0, 2, 3, 4]].int()
    feats = sp.SparseTensor(
        feats = torch.zeros(coords.shape[0], 101, device=coords.device).float(),
        coords = coords,
    ).cuda()
    
    mesh_extractor = SparseFeatures2Mesh(res=resolution, use_color=False)
    
    mesh_rec = mesh_extractor(feats, training=False)
    vertices, faces = mesh_rec.vertices.cpu().numpy(), mesh_rec.faces.cpu().numpy()
    mesh = trimesh.Trimesh(vertices, faces)
    mesh.export(save_path)
    
    coords_src = torch.argwhere(ss>0.5)[:, [0, 2, 3, 4]].int()
    feats_src = sp.SparseTensor(
        feats = torch.zeros(coords_src.shape[0], 101, device=coords_src.device).float(),
        coords = coords_src,
    ).cuda()
    mesh_src = mesh_extractor(feats_src, training=False)

    return mesh_src, mesh_rec


def render_video_pair(mesh_src, mesh_rec):
            
    video_src = render_utils.render_video(mesh_src)['normal']
    video_rec = render_utils.render_video(mesh_rec)['normal']

    return video_src, video_rec

    
if __name__ == '__main__':
    
    output_dir = 'datasets/dt4d'
    save_path = 'struc_vae_out_001.obj'
    obj_name = 'tiger-03'
    
    video_src_path = 'struc_src.mp4'
    video_rec_path = 'struc_rec.mp4'
    
    input_mesh, result_mesh = inference_struc_vae(output_dir, obj_name, save_path)    

    video_src, video_rec = render_video_pair(input_mesh, result_mesh)
    
    imageio.mimsave(video_src_path, video_src, fps=30)
    imageio.mimsave(video_rec_path, video_rec, fps=30)
