class EnvironmentSettings:
    def __init__(self):
        # NOTE: Paths must be accessible to the current user.
        # This repo is located at /home/chanyuan/02_RESEARCH/s3lab_research_v2/python
        # and `data/` is a symlink to the actual dataset storage.
        self.workspace_dir = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/pretrained_networks'
        self.lasot_dir = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/lasot'
        self.got10k_dir = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/got10k/train'
        self.got10k_val_dir = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/got10k/val'
        self.lasot_lmdb_dir = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/got10k_lmdb'
        self.trackingnet_dir = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/trackingnet'
        self.trackingnet_lmdb_dir = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/trackingnet_lmdb'
        self.coco_dir = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/coco'
        self.coco_lmdb_dir = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/vid'
        self.imagenet_lmdb_dir = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.uav123_dir = '/home/chanyuan/02_RESEARCH/s3lab_research_v2/python/data/uav123/UAV123'
