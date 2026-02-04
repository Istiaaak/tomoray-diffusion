import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset


class VerseDataset(Dataset):
    def __init__(self, folder, filenames, ext='.pickle'):
        super().__init__()

        self.folder = folder

        if not os.path.exists(folder):
            raise FileNotFoundError(f"folder {folder} does not exists")
        
        self.paths = []
        if filenames is not None:
            for name in filenames:
                if name.endswith('.nii.gz'):
                    stem = name[:-7]
                elif name.endswith('.nii'):
                    stem = name[:-4]
                else:
                    stem = os.path.splitext(name)[0]
                
                pickle_name = f"{stem}{ext}"
                
                full_path = os.path.join(folder, pickle_name)
                
                if os.path.exists(full_path):
                    self.paths.append(full_path)
                else:
                    pass
        
        else:
            self.paths = [
                os.path.join(folder, f) 
                for f in os.listdir(folder) 
                if f.endswith(ext)
            ]
            self.paths.sort()

        self.name_to_index = {}
        for idx, path in enumerate(self.paths):
            filename = os.path.basename(path) 
            stem = os.path.splitext(filename)[0]
            

            self.name_to_index[filename] = idx 
            self.name_to_index[stem] = idx     

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        if isinstance(index, str):
            if index in self.name_to_index:
                index = self.name_to_index[index]
            else:
                raise KeyError(f"file '{index}' is not here.")
        path = self.paths[index]
        
        with open(path, 'rb') as f:
            data = pickle.load(f)


        ct_image = data['image']
        
        ct_tensor = torch.from_numpy(ct_image).float()

        ct_tensor = ct_tensor.permute(2, 1, 0)
        
        # (D, H, W) -> (1, D, H, W)
        if ct_tensor.ndim == 3:
            ct_tensor = ct_tensor.unsqueeze(0)
            
        # [0, 1] to [-1, 1]
        ct_tensor = (ct_tensor * 2.0) - 1.0


        # (N_angles, H, W)
        projections = data['train']['projections']
        angles = data['train']['angles']

        proj_tensor = torch.from_numpy(projections).float()
        angles_tensor = torch.from_numpy(angles).float()
        
        # (N, H, W) -> (N, 1, H, W)
        if proj_tensor.ndim == 3:
            proj_tensor = proj_tensor.unsqueeze(1)


        return {
            'image': ct_tensor,       # [1, 128, 128, 128]
            'projections': proj_tensor, # [N, 1, 240, 240]
            'angles': angles_tensor,     # [N]
            'name': os.path.basename(path)

        }