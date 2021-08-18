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

def add_method(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        return func
    return decorator

# @add_method(NuScenesDataset)
# def get_sweep(self, sweep_info):
#         def remove_ego_points(points, center_radius=1.0):
#             mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
#             return points[mask]

#         lidar_path = self.root_path / sweep_info['lidar_path']
#         points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]
#         points_sweep = remove_ego_points(points_sweep).T
#         if sweep_info['transform_matrix'] is not None:
#             num_points = points_sweep.shape[1]
#             points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
#                 np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

#         cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
#         return points_sweep.T, cur_times.T

# @add_method(NuScenesDataset)
# def get_lidar_with_sweeps(self, index, max_sweeps=1):
#     info = self.infos[index]
#     lidar_path = self.root_path / info['lidar_path']
#     points = np.fromfile(str(lidar_path), dtype=np.float32)
    
#     points[3::5] = points[3::5]/255
#     points = np.delete(points, np.arange(4, points.size, 5)).reshape(-1, 4)

#     sweep_points_list = [points]

#     for k in np.random.choice(len(info['sweeps']), max_sweeps - 1, replace=False):
#         points_sweep, _ = self.get_sweep(info['sweeps'][k])
#         points_sweep[3::4] = points_sweep[3::4]/255
#         sweep_points_list.append(points_sweep)

#     points = np.concatenate(sweep_points_list, axis=0)
#     print(points)
    
#     return points

# @add_method(NuScenesDataset)
# def get_lidar_with_sweeps(self, index, max_sweeps=1):
#     print(index)
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
    parser.add_argument('--idx', type=int, default='0', help='specify the id o the scene to show')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():

    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    demo_dataset = NuScenesDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    start = time.time()
    with torch.no_grad():
        data_dict = demo_dataset[args.idx]
        print(type(data_dict))
        print(data_dict)
        
        logger.info(f'Visualized sample index: \t{args.idx}')
        data_dict = demo_dataset.collate_batch([data_dict])
        # print(type(data_dict))
        # print(data_dict)
        load_data_to_gpu(data_dict)
        pred_dicts, _ = model.forward(data_dict)
        # for bb in pred_dicts[0]['pred_boxes']:
        #     for x in bb:
        #         print(float(x), end="  ")
        #     print()

        # pred_dicts[0]['pred_boxes'][:,0] = 1
        # pred_dicts[0]['pred_boxes'][:,1] = 0
        # pred_dicts[0]['pred_boxes'][:,2] = 1
        # pred_dicts[0]['pred_boxes'][:,3] = 1
        # pred_dicts[0]['pred_boxes'][:,4] = 1
        # pred_dicts[0]['pred_boxes'][:,5] = 1
        # pred_dicts[0]['pred_boxes'][:,6] = 0
        
        # pred_dicts[0]['pred_boxes'] = pred_dicts[0]['pred_boxes'].cpu().numpy()
        # foo = np.zeros((len(pred_dicts[0]['pred_boxes']), 2), dtype=pred_dicts[0]['pred_boxes'].dtype)
        # pred_dicts[0]['pred_boxes'] = np.concatenate((pred_dicts[0]['pred_boxes'], foo), axis=1)


        # pred_dicts[0]['pred_boxes'][:,7] = 0
        # pred_dicts[0]['pred_boxes'][:,8] = 0

        mask = (pred_dicts[0]['pred_scores']>0.3).float()
        indices = torch.nonzero(mask)
        V.draw_scenes(
            points=data_dict['points'][:, 1:],
            ref_boxes=pred_dicts[0]['pred_boxes'][indices].reshape(-1, 9),
            ref_scores=pred_dicts[0]['pred_scores'][indices].reshape(-1), ref_labels=pred_dicts[0]['pred_labels'][indices].reshape(-1)
        )

        # mask = (pred_dicts[0]['pred_scores']>0.5).float()
        # indices = torch.nonzero(mask)
        # V.draw_scenes(
        #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'][indices].reshape(-1, 7),
        #     ref_scores=pred_dicts[0]['pred_scores'][indices].reshape(-1), ref_labels=pred_dicts[0]['pred_labels'][indices].reshape(-1)
        # )


        # print(pred_dicts[0]['pred_boxes'][indices])

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
