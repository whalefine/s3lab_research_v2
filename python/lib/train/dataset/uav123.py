import os
import torch
import numpy as np
import pandas
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class UAV123(BaseVideoDataset):
    """ UAV123 dataset for training.

    Publication:
        A Benchmark and Simulator for UAV Tracking
        Matthias Mueller, Neil Smith and Bernard Ghanem
        ECCV 2016
        https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx

    UAV123 contains 123 video sequences. We use the first 91 sequences (UAV123 subset)
    for training, which are located in data_seq/UAV123/ directory.
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, data_fraction=None):
        """
        args:
            root - path to the UAV123 dataset root (e.g., data/uav123/UAV123)
            image_loader (jpeg4py_loader) - The function to read the images.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().uav123_dir if root is None else root
        super().__init__('UAV123', root, image_loader)

        # UAV123 has sequences in data_seq/UAV123/ directory
        self.sequence_list = self._build_sequence_list()

        if data_fraction is not None:
            import random
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

    def _build_sequence_list(self):
        """Build list of sequences from UAV123 dataset."""
        seq_dir = os.path.join(self.root, 'data_seq', 'UAV123')
        anno_dir = os.path.join(self.root, 'anno', 'UAV123')
        
        if not os.path.exists(seq_dir):
            raise RuntimeError(f'UAV123 sequence directory not found: {seq_dir}')
        
        if not os.path.exists(anno_dir):
            raise RuntimeError(f'UAV123 annotation directory not found: {anno_dir}')
        
        # Only include sequences that have corresponding annotation files
        all_sequences = sorted([d for d in os.listdir(seq_dir) 
                               if os.path.isdir(os.path.join(seq_dir, d))])
        
        sequence_list = [seq for seq in all_sequences 
                        if os.path.exists(os.path.join(anno_dir, f'{seq}.txt'))]
        
        print(f"UAV123: Found {len(sequence_list)}/{len(all_sequences)} sequences with annotations")
        
        return sequence_list

    def get_name(self):
        return 'uav123'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    def get_num_sequences(self):
        return len(self.sequence_list)

    def _read_bb_anno(self, seq_name):
        """Read bounding box annotations."""
        anno_file = os.path.join(self.root, 'anno', 'UAV123', f'{seq_name}.txt')
        
        if not os.path.exists(anno_file):
            raise RuntimeError(f'Annotation file not found: {anno_file}')
        
        # Read CSV, allowing NaN values and converting them properly
        gt = pandas.read_csv(anno_file, delimiter=',', header=None, 
                            dtype=np.float32, na_filter=True, low_memory=False,
                            keep_default_na=True).values
        
        # Replace NaN with 0 (invalid bbox)
        gt = np.nan_to_num(gt, nan=0.0)
        
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        """Get the path to a sequence."""
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root, 'data_seq', 'UAV123', seq_name)

    def get_sequence_info(self, seq_id):
        """Get sequence information (bounding boxes and valid flags)."""
        seq_name = self.sequence_list[seq_id]
        bbox = self._read_bb_anno(seq_name)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        """Get the path to a specific frame."""
        return os.path.join(seq_path, f'{frame_id+1:06d}.jpg')  # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        """Load a specific frame."""
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        """Get frames and annotations for a sequence."""
        seq_path = self._get_sequence_path(seq_id)
        seq_name = self.sequence_list[seq_id]

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        # UAV123 metadata - categorize by prefix
        object_class = 'uav'  # Generic class for UAV tracking
        if any(seq_name.startswith(prefix) for prefix in ['person', 'group']):
            object_class = 'person'
        elif any(seq_name.startswith(prefix) for prefix in ['car', 'truck']):
            object_class = 'vehicle'
        elif any(seq_name.startswith(prefix) for prefix in ['boat', 'wakeboard']):
            object_class = 'water_vehicle'
        elif any(seq_name.startswith(prefix) for prefix in ['bike', 'motor']):
            object_class = 'bike'
        elif any(seq_name.startswith(prefix) for prefix in ['bird']):
            object_class = 'bird'

        object_meta = OrderedDict({
            'object_class_name': object_class,
            'motion_class': None,
            'major_class': None,
            'root_class': None,
            'motion_adverb': None
        })

        return frame_list, anno_frames, object_meta
