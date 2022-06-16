import numpy as np
import pyrender
import trimesh
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from scipy.spatial.transform import Rotation
from PIL import Image, ImageOps, ImageDraw, ImageFont
from tqdm import tqdm


def create_raymond_lights():
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))
    return nodes


def write_text_to_image(array, text):
    img = Image.fromarray(array)
    img = ImageOps.expand(img, (40, 20, 0, 0), fill=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("exercise_3/util/font/FreeMono.ttf", 14)
    draw.text((10, 10), text, (0, 0, 0), font=font)
    return np.array(img)


def meshes_to_gif(mesh_paths, output_path, fps):
    w_img = 512
    r = pyrender.OffscreenRenderer(w_img, w_img)
    image_buffer = [np.zeros((w_img, w_img, 3), dtype=np.uint8) for i in range(len(mesh_paths))]
    try:
        for i, mesh_path in enumerate(tqdm(mesh_paths, desc='visualizing')):
            base_mesh = trimesh.load_mesh(mesh_path)
            loc = np.array([32, 32, 32])
            scale = 48
            base_mesh.apply_translation(-loc)
            base_mesh.apply_scale(1 / scale)
            mesh = pyrender.Mesh.from_trimesh(base_mesh)
            camera_rotation = np.eye(4)
            camera_rotation[:3, :3] = Rotation.from_euler('y', -135, degrees=True).as_matrix() @ Rotation.from_euler('x', -45, degrees=True).as_matrix()
            camera_translation = np.eye(4)
            camera_translation[:3, 3] = np.array([0, 0, 1.25])
            camera_pose = camera_rotation @ camera_translation
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
            scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0])
            scene.add(mesh)
            scene.add(camera, pose=camera_pose)
            for n in create_raymond_lights():
                scene.add_node(n, scene.main_camera_node)
            color, depth = r.render(scene)
            image_buffer[i] = write_text_to_image(color, f"{mesh_path.name.split('.')[0]}")
        clip = ImageSequenceClip(image_buffer, fps=fps)
        clip.write_gif(output_path, verbose=False, logger=None)

    except Exception as e:
        print("Visualization failed", e)
