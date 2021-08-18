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
from pcdet.datasets.dataset import DatasetTemplate

import time
from functools import wraps

import statistics

def add_method(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        return func
    return decorator

@add_method(NuScenesDataset)
def get_sweep(self, sweep_info):
        def remove_ego_points(points, center_radius=1.0):
            mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            return points[mask]

        lidar_path = self.root_path / sweep_info['lidar_path']
        points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
        points_sweep = remove_ego_points(points_sweep).T
        if sweep_info['transform_matrix'] is not None:
            num_points = points_sweep.shape[1]
            points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

        return points_sweep.T

@add_method(NuScenesDataset)
def get_lidar_with_sweeps(self, index, max_sweeps=1):
    info = self.infos[index]
    lidar_path = self.root_path / info['lidar_path']
    points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

    sweep_points_list = [points]
    # sweep_times_list = [np.zeros((points.shape[0], 1))]

    for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
        points_sweep, _ = self.get_sweep(info['sweeps'][k])
        sweep_points_list.append(points_sweep)
        # sweep_times_list.append(times_sweep)

    points = np.concatenate(sweep_points_list, axis=0)
    
    points[3::5] = points[3::5]/255
    points = np.delete(points, np.arange(4, points.size, 5)).reshape(-1, 4)
    # times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

    # points = np.concatenate((points, times), axis=1)
    return points

# @add_method(NuScenesDataset)
# def get_lidar_with_sweeps(self, index, max_sweeps=1):
#     def remove_ego_points(points, center_radius=2.5):
#             mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
#             return points[mask]
#     info = self.infos[index]
#     lidar_path = self.root_path / info['lidar_path']
#     # points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
#     points = np.fromfile(str(lidar_path), dtype=np.float32)
#     points[3::5] = points[3::5]/255
#     points = np.delete(points, np.arange(4, points.size, 5)).reshape(-1, 4)
#     points = remove_ego_points(points)
#     return points

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/nuscenes_models/cbgs_pp_multihead.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/media/javier/HDD_linux/data/nuscenes',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--frames', type=int, default='6019', help='specify the number of frames to use')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():

    for sweep in [1,2,3,4,5,6,7,8,9,10]:
        lines = open('cfgs/dataset_configs/nuscenes_dataset.yaml').read().splitlines()
        lines[4] = 'MAX_SWEEPS: ' + str(sweep)
        open('cfgs/dataset_configs/nuscenes_dataset.yaml','w').write('\n'.join(lines))

        args, cfg = parse_config()
        logger = common_utils.create_logger()

        demo_dataset = NuScenesDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(args.data_path), logger=logger
        )   

        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()

        times = []
    
        for i in range(args.frames):
            with torch.no_grad():
                data_dict = demo_dataset[i]

                start = time.time()
                data_dict = demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)
                pred_dicts, _ = model.forward(data_dict)
                end = time.time()
                times.append(end-start)

        print("sweeps",sweep)
        print("min",1/min(times))
        print("1st_quantile",1/np.quantile(times,.25))
        print("median",1/statistics.median(times))
        print("1st_quantile",1/np.quantile(times,.75))
        print("max",1/max(times))
        print()

if __name__ == '__main__':
    main()
