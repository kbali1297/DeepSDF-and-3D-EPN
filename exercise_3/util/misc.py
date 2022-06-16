from pathlib import Path

import torch
import numpy as np
import trimesh
from skimage.measure import marching_cubes


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


def evaluate_model_on_grid(model, latent_code, device, grid_resolution, export_path):
    x_range = y_range = z_range = np.linspace(-1., 1., grid_resolution)
    grid_x, grid_y, grid_z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
    grid_x, grid_y, grid_z = grid_x.flatten(), grid_y.flatten(), grid_z.flatten()
    stacked = torch.from_numpy(np.hstack((grid_x[:, np.newaxis], grid_y[:, np.newaxis], grid_z[:, np.newaxis]))).float().to(device)
    stacked_split = torch.split(stacked, 32 ** 3, dim=0)
    sdf_values = []
    for points in stacked_split:
        with torch.no_grad():
            sdf = model(torch.cat([latent_code.unsqueeze(0).expand(points.shape[0], -1), points], 1))
        sdf_values.append(sdf.detach().cpu())
    sdf_values = torch.cat(sdf_values, dim=0).numpy().reshape((grid_resolution, grid_resolution, grid_resolution))
    if 0 < sdf_values.min() or 0 > sdf_values.max():
        vertices, faces = [], []
    else:
        vertices, faces, _, _ = marching_cubes(sdf_values, level=0)
    if export_path is not None:
        Path(export_path).parent.mkdir(exist_ok=True)
        trimesh.Trimesh(vertices=vertices, faces=faces).export(export_path)
    return vertices, faces


def show_gif(fname):
    import base64
    from IPython import display
    with open(fname, 'rb') as fd:
        b64 = base64.b64encode(fd.read()).decode('ascii')
    return display.HTML(f'<img src="data:image/gif;base64,{b64}" />')
