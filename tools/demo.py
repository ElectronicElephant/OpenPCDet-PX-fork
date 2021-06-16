import argparse
import copy
import glob
import os.path
from pathlib import Path

import cv2
import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V
import json

import torchvision.transforms as transforms
from PIL import Image
import random

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
        input_dict = {}
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
            input_dict['points'] = points
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
            input_dict['points'] = points
        else:
            raise NotImplementedError
        input_dict = {
            'frame_id': index,
        }
        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


class DemoImageDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin', args=None):
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
        data_file_list = glob.glob(str(root_path / "images" / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        # data_file_list.sort()
        random.shuffle(data_file_list)
        self.sample_file_list = data_file_list
        self.calib_path = os.path.join(root_path, "parameter.json")
        self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)
        self.transforms = None
        if args is not None:
            self.transforms = self.get_transform(args)


    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        input_dict = {}
        input_dict['images'] = cv2.imread(self.sample_file_list[index])
        # input_dict['images'] = cv2.resize(input_dict['images'], (input_dict['images'].shape[1]//2, input_dict['images'].shape[0]//2))
        input_dict['images'] = input_dict['images'][...,::-1]
        input_dict['frame_id'] = index
        input_dict['calib'] = self.get_calib(self.calib_path)
        input_dict['trans_cam_to_img'] = np.array(input_dict['calib']["camera_intrinsic"], dtype=np.float32)
        # input_dict['trans_cam_to_img'] = np.hstack((input_dict['trans_cam_to_img'], np.zeros((input_dict['trans_cam_to_img'].shape[0], 1), dtype=np.float32)))
        input_dict['trans_lidar_to_cam'] = np.array(input_dict['calib']["trans_lidar_to_cam"], dtype=np.float32)
        input_dict['image_shape'] = np.array([input_dict['images'].shape[0],input_dict['images'].shape[1]], dtype=np.float32)
        data_dict = self.prepare_data(data_dict=input_dict)
        # data_dict['images'] = data_dict['images'].permute(1,2,0)
        return data_dict

    def get_calib(self, calib_path):
        calib_file = open(calib_path, "r")
        calib_info = json.load(calib_file)
        return calib_info

    def prepare_data(self, data_dict):
        # if "images" in data_dict.keys():
        #     data_dict["images"] = self.transforms(data_dict["images"])
        data_dict['images'] = data_dict['images'] / 255.0
        return data_dict

    def get_transform(self, opt):
        transform_list = []
        osize = [opt.input_h, opt.input_w]
        transform_list.append(transforms.Resize(osize, Image.NEAREST))
        transform_list += [transforms.ToTensor(),
                           transforms.Normalize((self.mean[0][0][0], self.mean[0][0][1], self.mean[0][0][2]),
                                                (self.std[0][0][0], self.std[0][0][1], self.std[0][0][2]))]
        return transforms.Compose(transform_list)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='demo_data',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--input_h', type=int, default=-1,
                             help='input height. -1 for default from dataset.')
    parser.add_argument('--input_w', type=int, default=-1,
                             help='input width. -1 for default from dataset.')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    # demo_dataset = DemoDataset(
    #     dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    #     root_path=Path(args.data_path), ext=args.ext, logger=logger
    # )

    demo_dataset = DemoImageDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger, args=args
    )

    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            img = copy.deepcopy(data_dict["images"])
            img = img*255
            img = img.astype(np.uint8)
            img = img[...,::-1]

            logger.info(f'Visualized sample index: \t{idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # V.draw_scenes(
            #     points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
            #     ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            # )

            V.draw_scenes(ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )
            # img = V.draw_on_image(img, data_dict['calib'][0]["camera_intrinsic"], ref_boxes=pred_dicts[0]['pred_boxes'], ref_labels=pred_dicts[0]['pred_labels'])
            cv2.imshow("a",img)
            cv2.waitKey(0)
            mlab.show(stop=False)


    logger.info('Demo done.')


if __name__ == '__main__':
    main()
