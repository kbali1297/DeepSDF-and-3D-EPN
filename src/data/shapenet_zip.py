from pathlib import Path
import json
import zipfile

import numpy as np
import torch


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 8
    dataset_sdf_path = Path("src/data/shapenet_dim32_sdf")  # path to voxel data
    dataset_df_path = Path("src/data/shapenet_dim32_df")  # path to voxel data
    class_name_mapping = json.loads(Path("src/data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.truncation_distance = 3

        # This will be filled in worker_init_fn
        self.sdf_archive = None
        self.df_archive = None

        self.items = Path(f"src/data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        sdf_id, df_id = self.items[index].split(' ')

        input_sdf = self.get_shape_sdf(sdf_id)
        target_df = self.get_shape_df(df_id)

        #  Apply truncation to sdf and df
        #  Stack (distances, sdf sign) for the input sdf
        #  Log-scale target df

        return {
            'name': f'{sdf_id}-{df_id}',
            'input_sdf': input_sdf,
            'target_df': target_df
        }

    def __len__(self):
        return len(self.items)

    def worker_init_fn(self, worker_id):
        self.sdf_archive = zipfile.ZipFile('src/data/shapenet_dim32_sdf.zip', 'r')
        self.df_archive = zipfile.ZipFile('src/data/shapenet_dim32_df.zip', 'r')

    @staticmethod
    def move_batch_to_device(batch, device):
        #  add code to move batch to device
        pass

    def get_shape_sdf(self, shapenet_id):
        data = self.sdf_archive.read(f'shapenet_dim32_sdf/{shapenet_id}.sdf')
        sdf = None
        #  implement sdf data loading
        return sdf

    def get_shape_df(self, shapenet_id):
        data = self.df_archive.read(f'shapenet_dim32_df/{shapenet_id}.df')
        df = None
        #  implement df data loading
        return df
