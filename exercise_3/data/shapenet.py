from pathlib import Path
import json
import string

import numpy as np
from sklearn import datasets
import torch


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 8
    dataset_sdf_path = Path("exercise_3/data/shapenet_dim32_sdf")  # path to voxel data
    dataset_df_path = Path("exercise_3/data/shapenet_dim32_df")  # path to voxel data
    class_name_mapping = json.loads(Path("exercise_3/data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.truncation_distance = 3

        self.items = Path(f"exercise_3/data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        sdf_id, df_id = self.items[index].split(' ')

        input_sdf = ShapeNet.get_shape_sdf(sdf_id)
        target_df = ShapeNet.get_shape_df(df_id)

        # TODO Apply truncation to sdf and df
        input_sdf = np.clip(input_sdf, -self.truncation_distance, self.truncation_distance)
        target_df = np.clip(target_df, -self.truncation_distance, self.truncation_distance)
        # TODO Stack (distances, sdf sign) for the input sdf
        input_sdf = np.expand_dims(input_sdf, 0)
        input_sdf = np.concatenate([np.fabs(input_sdf), np.sign(input_sdf)], axis=0)
        #input_sdf = float(input_sdf)
        #input_sdf.append(np.where(input_sdf>0, 1, -1))
        # TODO Log-scale target df
        target_df = np.log(target_df+1)
        return {
            'name': f'{sdf_id}-{df_id}',
            'input_sdf': input_sdf,
            'target_df': target_df
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        # TODO add code to move batch to device
        batch['input_sdf'] = batch['input_sdf'].to(device, dtype=torch.float)
        batch['target_df'] = batch['target_df'].to(device, dtype=torch.float)
        
    @staticmethod
    def get_shape_sdf(shapenet_id):
        sdf = None
        # TODO implement sdf data loading
        #print(shapenet_id)
        path = str(ShapeNet.dataset_sdf_path)  + "/" + shapenet_id + '.sdf'
        dims = np.fromfile(path, np.uint64, 3)
        sdf = np.fromfile(path, dtype = np.float32, offset=3*8).reshape(dims)
        return sdf

    @staticmethod
    def get_shape_df(shapenet_id):
        df = None
        # TODO implement df data loading
        path = str(ShapeNet.dataset_df_path) + "/" + shapenet_id + '.df' 
        dims = np.fromfile(path,np.uint64, 3)
        df = np.fromfile(path, np.float32, offset=3*8).reshape(dims)
        return df
