""" Visualization utilities """
from pathlib import Path

import numpy as np
import k3d
from matplotlib import cm, colors
import trimesh


def visualize_mesh(vertices, faces, flip_axes=False):
    vertices = np.array(vertices)
    plot = k3d.plot(name='mesh', grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    if flip_axes:
        rot_matrix = np.array([
            [-1.0000000, 0.0000000, 0.0000000],
            [0.0000000, 0.0000000, 1.0000000],
            [0.0000000, 1.0000000, 0.0000000]
        ])
        vertices = vertices @ rot_matrix
    plt_mesh = k3d.mesh(vertices.astype(np.float32), faces.astype(np.uint32), color=0xd0d0d0)
    plot += plt_mesh
    plt_mesh.shader = '3d'
    plot.display()


def visualize_meshes(meshes, flip_axes=False):
    assert len(meshes) == 3
    plot = k3d.plot(name='meshes', grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    for mesh_idx, mesh in enumerate(meshes):
        vertices, faces = mesh[:2]
        if flip_axes:
            vertices[:, 2] = vertices[:, 2] * -1
            vertices[:, [0, 1, 2]] = vertices[:, [0, 2, 1]]
        vertices += [[-32, -32, 0], [0, -32, 0], [32, -32, 0]][mesh_idx]
        plt_mesh = k3d.mesh(vertices.astype(np.float32), faces.astype(np.uint32), color=0xd0d0d0)
        plot += plt_mesh
        plt_mesh.shader = '3d'
    plot.display()


def visualize_pointcloud(point_cloud, point_size, colors=None, flip_axes=False, name='point_cloud'):
    point_cloud = point_cloud.copy()
    plot = k3d.plot(name=name, grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    if flip_axes:
        point_cloud[:, 2] = point_cloud[:, 2] * -1
        point_cloud[:, [0, 1, 2]] = point_cloud[:, [0, 2, 1]]
        point_cloud[:, 1] = point_cloud[:, 1] * -1
    plt_points = k3d.points(positions=point_cloud.astype(np.float32), point_size=point_size, colors=colors if colors is not None else [], color=0xd0d0d0)
    plot += plt_points
    plt_points.shader = '3d'
    plot.display()


def visualize_sdf(sdf: np.array, filename: Path) -> None:
    assert sdf.shape[0] == sdf.shape[1] == sdf.shape[2], "SDF grid has to be of cubic shape"
    print(f"Creating SDF visualization for {sdf.shape[0]}^3 grid ...")

    voxels = np.stack(np.meshgrid(range(sdf.shape[0]), range(sdf.shape[1]), range(sdf.shape[2]))).reshape(3, -1).T

    sdf[sdf < 0] /= np.abs(sdf[sdf < 0]).max()
    sdf[sdf > 0] /= sdf[sdf > 0].max()
    sdf /= 2.

    corners = np.array([
        [-.25, -.25, -.25],
        [.25, -.25, -.25],
        [-.25, .25, -.25],
        [.25, .25, -.25],
        [-.25, -.25, .25],
        [.25, -.25, .25],
        [-.25, .25, .25],
        [.25, .25, .25]
    ])[np.newaxis, :].repeat(voxels.shape[0], axis=0).reshape(-1, 3)

    scale_factors = sdf[tuple(voxels.T)].repeat(8, axis=0)
    cube_vertex_colors = cm.get_cmap('seismic')(colors.Normalize(vmin=-1, vmax=1)(scale_factors))[:, :3]
    scale_factors[scale_factors < 0] *= .25
    cube_vertices = voxels.repeat(8, axis=0) + corners * scale_factors[:, np.newaxis]

    faces = np.array([
        [1, 0, 2], [2, 3, 1], [5, 1, 3], [3, 7, 5], [4, 5, 7], [7, 6, 4],
        [0, 4, 6], [6, 2, 0], [3, 2, 6], [6, 7, 3], [5, 4, 0], [0, 1, 5]
    ])[np.newaxis, :].repeat(voxels.shape[0], axis=0).reshape(-1, 3)
    cube_faces = faces + (np.arange(0, voxels.shape[0]) * 8)[np.newaxis, :].repeat(12, axis=0).T.flatten()[:, np.newaxis]

    mesh = trimesh.Trimesh(vertices=cube_vertices, faces=cube_faces, vertex_colors=cube_vertex_colors, process=False)
    mesh.export(str(filename))
    print(f"Exported to {filename}")


def visualize_shape_alignment(R=None, t=None):
    mesh_input = trimesh.load(Path(__file__).parent.parent / "resources" / "mesh_input.obj")
    mesh_target = trimesh.load(Path(__file__).parent.parent / "resources" / "mesh_target.obj")
    plot = k3d.plot(name='aligment', grid_visible=False, grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    input_vertices = np.array(mesh_input.vertices)
    if not (R is None or t is None):
        t_broadcast = np.broadcast_to(t[:, np.newaxis], (3, mesh_input.vertices.shape[0]))
        input_vertices = (R @ input_vertices.T + t_broadcast).T
    plt_mesh_0 = k3d.mesh(input_vertices.astype(np.float32), np.array(mesh_input.faces).astype(np.uint32), color=0xd00d0d)
    plt_mesh_1 = k3d.mesh(np.array(mesh_target.vertices).astype(np.float32), np.array(mesh_target.faces).astype(np.uint32), color=0x0dd00d)
    plot += plt_mesh_0
    plot += plt_mesh_1
    plt_mesh_0.shader = '3d'
    plt_mesh_1.shader = '3d'
    plot.display()
