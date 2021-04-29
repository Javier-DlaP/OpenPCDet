import argparse
import glob
from pathlib import Path

import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V

from pcdet.datasets.nuscenes.nuscenes_dataset import *

import time
from nuscenes.nuscenes import NuScenes

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        def remove_ego_points(points, center_radius=1.5):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        if self.ext == '.bin':
            # Run Kitti model on Kitti
            # points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
            
            # Run Kitti model on nuScenes
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32)
            points[3::5] = points[3::5]/255
            points = np.delete(points, np.arange(4, points.size, 5)).reshape(-1, 4)
            points = remove_ego_points(points)

            # Run nuScenes model on nuScenes
            # nusc = NuScenes(version='v1.0-trainval', dataroot='/media/javier/HDD_linux/data/nuscenes/v1.0-trainval', verbose=True)
            # my_scene = nusc.scene[1]
            # points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 5)
            # points[4::5] = index

            # first_sample_token = my_scene['first_sample_token']
            # my_sample = nusc.get('sample', first_sample_token)
            # lidar_top_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
            # filename = lidar_top_data['filename']
            # points = np.fromfile(Path('/media/javier/HDD_linux/data/nuscenes/v1.0-trainval/'+filename), dtype=np.float32).reshape(-1, 5)
            # points[4::5] = lidar_top_data['timestamp']

            # sample_token = nusc.get('sample', first_sample_token)['next']
            # my_sample = nusc.get('sample', sample_token)
            # lidar_top_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
            # filename = lidar_top_data['filename']
            # points_ = np.fromfile(Path('/media/javier/HDD_linux/data/nuscenes/v1.0-trainval/'+filename), dtype=np.float32).reshape(-1, 5)
            # points_[4::5] = lidar_top_data['timestamp']

            # points = np.append(points, points_)

        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        print(index)

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():

    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    start = time.time()
    # print(demo_dataset)
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: \t{idx + 1}')
            print(data_dict)
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            mask = (pred_dicts[0]['pred_scores']>0.3).float()
            indices = torch.nonzero(mask)
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'][indices].reshape(-1, 7),
                ref_scores=pred_dicts[0]['pred_scores'][indices].reshape(-1), ref_labels=pred_dicts[0]['pred_labels'][indices].reshape(-1)
            )

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

            mlab.show(stop=True)

    end = time.time()
    print(end-start)

    logger.info('Demo done.')

if __name__ == '__main__':
    main()
