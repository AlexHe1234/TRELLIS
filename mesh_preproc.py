"""Process mesh to be ready for Trellis VAE input

Contain following steps from a single given mesh (trimesh.Trimesh):
1. Render multi-view images
2. Voxelize model into grid
3. Extract DINOv2 features
"""

import os
import json
import shutil
import subprocess
from functools import partial
from subprocess import DEVNULL, call
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

import torch
import trimesh
import numpy as np
import open3d as o3d
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

import utils3d
from dataset_toolkits.utils import sphere_hammersley_sequence


def copy_object(mesh_path: str, output_dir: str, obj_name: str) -> str:
    tar_path = os.path.join(output_dir, 'raw', obj_name)
    os.makedirs(tar_path, exist_ok=True)
    shutil.copy(mesh_path, tar_path)
    return tar_path


def render_multiview(
    mesh_path: str, 
    output_dir: str, 
    obj_name: str, 
    num_views: int = 150, 
    verbal: bool = False,
) -> Dict:
    
    BLENDER_PATH = '/viscam/u/alexhe/projects/generic_rigging/Dino3D/blender-4.2.0-linux-x64/blender'

    output_folder = os.path.join(output_dir, 'renders', obj_name)
    
    # Build camera {yaw, pitch, radius, fov}
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_views):
        y, p = sphere_hammersley_sequence(i, num_views, offset)
        yaws.append(y)
        pitchs.append(p)
    radius = [2] * num_views
    fov = [40 / 180 * np.pi] * num_views
    views = [{'yaw': y, 'pitch': p, 'radius': r, 'fov': f} for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    
    _args = [
        BLENDER_PATH, '-b', '-P', os.path.join(os.path.dirname(__file__), 'dataset_toolkits', 'blender_script', 'render.py'),
        '--',
        '--views', json.dumps(views),
        '--object', mesh_path,
        '--resolution', '512',
        '--output_folder', output_folder,
        '--engine', 'CYCLES',
        '--save_mesh',
    ]
    if mesh_path.endswith('.blend'):
        _args.insert(1, mesh_path)
    
    if not verbal: # default
        call(_args, stdout=DEVNULL, stderr=DEVNULL) 
    else:
        call(_args) 
    
    if os.path.exists(os.path.join(output_folder, 'transforms.json')):
        return {'obj_name': obj_name, 'rendered': True}    
    
    
def voxelize_mesh(output_dir: str, obj_name: str):
    # use the blender saved mesh  
    mesh = o3d.io.read_triangle_mesh(os.path.join(output_dir, 'renders', obj_name, 'mesh.ply'))
    
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    
    vertices = (vertices + 0.5) / 64 - 0.5
    os.makedirs(os.path.join(output_dir, 'voxels'), exist_ok=True)
    utils3d.io.write_ply(os.path.join(output_dir, 'voxels', f'{obj_name}.ply'), vertices)
    
    return {'obj_name': obj_name, 'voxelized': True, 'num_voxels': len(vertices)}


# load image from all views
def get_data(output_dir, frames, sha256):
    with ThreadPoolExecutor(max_workers=16) as executor:
        def worker(view):
            image_path = os.path.join(output_dir, 'renders', sha256, view['file_path'])
            try:
                image = Image.open(image_path)
            except:
                print(f"Error loading image {image_path}")
                return None
            image = image.resize((518, 518), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255
            image = image[:, :, :3] * image[:, :, 3:]
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            c2w = torch.tensor(view['transform_matrix'])
            c2w[:3, 1:3] *= -1
            extrinsics = torch.inverse(c2w)
            fov = view['camera_angle_x']
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(torch.tensor(fov), torch.tensor(fov))

            return {
                'image': image,
                'extrinsics': extrinsics,
                'intrinsics': intrinsics
            }
        
        datas = executor.map(worker, frames)
        for data in datas:
            if data is not None:
                yield data


def extract_feature_and_aggregate(output_dir: str, obj_name: str, batch_size: int = 16):
    
    torch.set_grad_enabled(False)
    
    model_name = 'dinov2_vitl14_reg' # default dino model
    os.makedirs(os.path.join(output_dir, 'features', model_name), exist_ok=True)
    
    dinov2_model = torch.hub.load('facebookresearch/dinov2', model_name)
    dinov2_model.eval().cuda()
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    n_patch = 518 // 14
    
    def loader():
        with open(os.path.join(output_dir, 'renders', obj_name, 'transforms.json'), 'r') as f:
            metadata = json.load(f)
        frames = metadata['frames']
        data = []
        for datum in get_data(output_dir, frames, obj_name):
            datum['image'] = transform(datum['image'])
            data.append(datum)
        positions = utils3d.io.read_ply(os.path.join(output_dir, 'voxels', f'{obj_name}.ply'))[0]
        # load_queue.put((sha256, data, positions))
        return obj_name, data, positions

    def saver(pack, patchtokens, uv):
        pack['patchtokens'] = F.grid_sample(
            patchtokens,
            uv.unsqueeze(1),
            mode='bilinear',
            align_corners=False,
        ).squeeze(2).permute(0, 2, 1).cpu().numpy()
        pack['patchtokens'] = np.mean(pack['patchtokens'], axis=0).astype(np.float16)
        save_path = os.path.join(output_dir, 'features', model_name, f'{obj_name}.npz')
        np.savez_compressed(save_path, **pack)
        # records.append({'obj_name': obj_name, f'feature_{model_name}' : True})

    _, data, positions = loader()
    positions = torch.from_numpy(positions).float().cuda()
    indices = ((positions + 0.5) * 64).long()
    assert torch.all(indices >= 0) and torch.all(indices < 64), "Some vertices are out of bounds"
    n_views = len(data)
    N = positions.shape[0]
    pack = {
        'indices': indices.cpu().numpy().astype(np.uint8),
    }
    patchtokens_lst = []
    uv_lst = []
    for i in tqdm(range(0, n_views, batch_size), desc='Extracting features'):
        batch_data = data[i:i+batch_size]
        bs = len(batch_data)
        batch_images = torch.stack([d['image'] for d in batch_data]).cuda()
        batch_extrinsics = torch.stack([d['extrinsics'] for d in batch_data]).cuda()
        batch_intrinsics = torch.stack([d['intrinsics'] for d in batch_data]).cuda()
        features = dinov2_model(batch_images, is_training=True)
        uv = utils3d.torch.project_cv(positions, batch_extrinsics, batch_intrinsics)[0] * 2 - 1
        patchtokens = features['x_prenorm'][:, dinov2_model.num_register_tokens + 1:].permute(0, 2, 1).reshape(bs, 1024, n_patch, n_patch)
        patchtokens_lst.append(patchtokens)
        uv_lst.append(uv)
    patchtokens = torch.cat(patchtokens_lst, dim=0)
    uv = torch.cat(uv_lst, dim=0)

    # save features
    saver(pack, patchtokens, uv)
    

def process_mesh(
    mesh_path: str, 
    output_dir: str, 
    obj_name: str, 
    skip_dino: bool = False, 
    verbal: bool = False,
) -> Dict:
    
    try:
        os.makedirs(output_dir, exist_ok=True)   
        copy_object(mesh_path, output_dir, obj_name)
        
        print("|------------ STAGE 1: Multi-view Rendering ------------|", flush=True)
        render_multiview(mesh_path, output_dir, obj_name, verbal=verbal) # step 1
        
        print("|------------ STAGE 2: Voxelization ------------|", flush=True)
        voxelize_mesh(output_dir, obj_name) # step 2
        
        if not skip_dino:
            print("|------------ STAGE 3: DINO Extraction and Feature Aggregation ------------|", flush=True)
            extract_feature_and_aggregate(output_dir, obj_name) # step 3
        
        print(f'DONE, Check {os.path.join(output_dir, obj_name)} for results.', flush=True)
        
        return {
            'success': True,
            'error': '',
        }
    
    except Exception as e:
    
        return {
            'success': False,
            'error': e
        }
        
        
def run_single_mesh(mesh_path: str, obj_name: str, output_dir: str):
    
    ret = process_mesh(mesh_path, output_dir, obj_name)
    
    if ret['success']: 
        print('Success', flush=True)
    else:              
        print(f'Failed, because {ret["error"]}', flush=True)
        
        
def run_multi_mesh(
    meshes_path: List[str], 
    obj_names: List[str], 
    output_dirs: List[str], 
    num_worker: int = 4,
):
    assert len(meshes_path) == len(obj_names) == len(output_dirs), \
        'Require same length for all arguments'
    inputs = list(zip(meshes_path, obj_names, output_dirs))
    
    from concurrent.futures import ProcessPoolExecutor
    
    with ProcessPoolExecutor(max_workers=num_worker) as executor:
        
        # skip dino bc it will hang on multiprocess
        futures = [executor.submit(partial(process_mesh, skip_dino=True), mesh_path, output_dir, obj_name) \
            for mesh_path, obj_name, output_dir in inputs]
        results = [f.result() for f in futures]  # Collect results
        
    # dino
    for mesh_path, obj_name, output_dir in tqdm(inputs, desc='Extracing DINO features...'):
        extract_feature_and_aggregate(output_dir, obj_name)
        
    print(' <-------- Finished Multiprocessing --------> ')
    
    
if __name__ == '__main__':
    
    
    # # * example for running single mesh
    # mesh_path = 'data/Trellis_Cache/raw_mesh/deer_00.obj'
    # obj_name = 'deer-00'
    # output_dir = 'data/Trellis_Cache/datasets/dt4d'
    # run_single_mesh(mesh_path, obj_name, output_dir)
    
    
    # * example for running all meshes within a directory
    meshes_path = 'data/Trellis_Cache/raw_mesh/' # directory containing all the .obj meshes to be processed
    meshes_keyword = 'deer' # only path containing keyword will be processed
    obj_name_base = 'deer' # will add indices after it
    output_dir = 'data/Trellis_Cache/datasets/dt4d' # output directory for all the processed data, does not have to be empty
    num_worker = 8
    
    mesh_paths = sorted([os.path.join(meshes_path, f) for f in os.listdir(meshes_path) if '.obj' in f and meshes_keyword in f])
    obj_names = []
    output_dirs = []
    
    for i, mp in enumerate(mesh_paths): # build args
        obj_names.append(obj_name_base + f'_{i:06d}')
        output_dirs.append(output_dir)
    
    run_multi_mesh(mesh_paths, obj_names, output_dirs, num_worker)
