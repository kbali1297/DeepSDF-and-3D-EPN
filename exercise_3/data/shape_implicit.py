from pathlib import Path
import numpy as np
import torch
import trimesh

from exercise_3.util.misc import remove_nans


class ShapeImplicit(torch.utils.data.Dataset):
    """
    Dataset for loading deep sdf training samples
    """

    dataset_path = Path("exercise_3/data/sdf_sofas")  # path to sdf data for ShapeNet sofa class - make sure you've downloaded the processed data at appropriate path

    def __init__(self, num_sample_points, split):
        """
        :param num_sample_points: number of points to sample for sdf values per shape
        :param split: one of 'train', 'val' or 'overfit' - for training, validation or overfitting split
        """
        super().__init__()
        assert split in ['train', 'val', 'overfit']

        self.num_sample_points = num_sample_points
        self.items = Path(f"exercise_3/data/splits/sofas/{split}.txt").read_text().splitlines()  # keep track of shape identifiers based on split

    def __getitem__(self, index):
        """
        PyTorch requires you to provide a getitem implementation for your dataset.
        :param index: index of the dataset sample that will be returned
        :return: a dictionary of sdf data corresponding to the shape. In particular, this dictionary has keys
                 "name", shape_identifier of the shape
                 "indices": index parameter
                 "points": a num_sample_points x 3  pytorch float32 tensor containing sampled point coordinates
                 "sdf", a num_sample_points x 1 pytorch float32 tensor containing sdf values for the sampled points
        """

        # get shape_id at index
        item = self.items[index]

        # get path to sdf data
        sdf_samples_path = ShapeImplicit.dataset_path / item / "sdf.npz"

        # read points and their sdf values from disk
        # TODO: Implement the method get_sdf_samples
        sdf_samples = self.get_sdf_samples(sdf_samples_path)

        points = sdf_samples[:, :3]
        sdf = sdf_samples[:, 3:]

        # truncate sdf values
        sdf_clamped = torch.clamp(sdf, -0.1, 0.1)

        return {
            "name": item,       # identifier of the shape
            "indices": index,   # index parameter
            "points": points,   # points, a tensor with shape num_sample_points x 3
            "sdf": sdf_clamped  # sdf values, a tensor with shape num_sample_points x 1
        }

    def __len__(self):
        """
        :return: length of the dataset
        """
        # TODO: Implement
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        """
        Utility method for moving all elements of the batch to a device
        :return: None, modifies batch inplace
        """
        batch['points'] = batch['points'].to(device)
        batch['sdf'] = batch['sdf'].to(device)
        batch['indices'] = batch['indices'].to(device)

    def get_sdf_samples(self, path_to_sdf):
        """
        Utility method for reading an sdf file; the SDF file for a shape contains a number of points, along with their sdf values
        :param path_to_sdf: path to sdf file
        :return: a pytorch float32 torch tensor of shape (num_sample_points, 4) with each row being [x, y, z, sdf_value at xyz]
        """
        npz = np.load(path_to_sdf)
        pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
        neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

        #print(pos_tensor.shape)
        # TODO: Implement such that you return a pytorch float32 torch tensor of shape (self.num_sample_points, 4)
        # the returned tensor shoud have approximately self.num_sample_points/2 randomly selected samples from pos_tensor
        # and approximately self.num_sample_points/2 randomly selected samples from neg_tensor
        indices1 = np.random.choice(pos_tensor.shape[0], int(self.num_sample_points/2), replace=False)
        indices2 = np.random.choice(neg_tensor.shape[0], int(self.num_sample_points/2) , replace=False)

        tensor = np.float32(np.concatenate( [pos_tensor[indices1,:], neg_tensor[indices2,:]], axis=0))

        return torch.from_numpy(tensor)

    @staticmethod
    def get_mesh(shape_id):
        """
        Utility method for loading a mesh from disk given shape identifier
        :param shape_id: shape identifier for ShapeNet object
        :return: trimesh object representing the mesh
        """
        return trimesh.load(ShapeImplicit.dataset_path / shape_id / "mesh.obj", force='mesh')

    @staticmethod
    def get_all_sdf_samples(shape_id):
        """
        Utility method for loading all points and their sdf values from disk
        :param shape_id: shape identifier for ShapeNet object
        :return: two torch float32 tensors, a Nx3 tensor containing point coordinates, and Nx1 tensor containing their sdf values
        """
        npz = np.load(ShapeImplicit.dataset_path / shape_id / "sdf.npz")
        pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
        neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

        samples = torch.cat([pos_tensor, neg_tensor], 0)
        points = samples[:, :3]

        # trucate sdf values
        sdf = torch.clamp(samples[:, 3:], -0.1, 0.1)

        return points, sdf
