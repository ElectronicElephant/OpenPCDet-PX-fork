import copy
import os
import pickle

import numpy as np
from skimage import io
import glob
import json
import math
import cv2
from .sensetime_util import get_2d_bbox, project_3D_to_image
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate
from ..centerbased.heatmap_process import get_target_info

class SenseDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
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
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        self.calibs = self.get_json_info(str(self.root_split_path / "parameter.json"))
        for key in self.calibs.keys():
            self.calibs[key] = np.array(self.calibs[key])
        self.sense_infos = []
        self.images = []
        self.img_sample = None
        self.include_sensetime_data(self.mode)

    def load_from_pkl(self, mode):
        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                self.sense_infos.extend(infos)

    def load_from_dir(self):
        image_file_list = glob.glob(str(self.root_split_path / "images" / f'*{".jpg"}'))
        json_file_list = glob.glob(str(self.root_split_path / "lidar_data" / f'*{".json"}'))
        self.images = image_file_list
        self.annots = {}
        for name in json_file_list:
            time_stmap = name.split("/")[-1].split(".")[0]
            annots = self.get_json_info(name)
            self.annots[time_stmap] = annots
        self.del_empty_info()
        self.img_sample = io.imread(self.images[0])

    def include_sensetime_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Sense dataset')
        pkl_pth = self.root_path / self.dataset_cfg.INFO_PATH[mode][0]
        if os.path.exists(pkl_pth):
            self.load_from_pkl(mode)
        else:
            self.load_from_dir()
        data_len = max(len(self.images), len(self.sense_infos))
        self.sample_id_list = [i for i in range(data_len)]
        if self.logger is not None:
            self.logger.info('Total samples for Sense dataset: %d' % (data_len))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')
        self.sense_infos = []
        self.img_sample = None
        self.include_sensetime_data(self.mode)

    def get_json_info(self, calib_path):
        calib_file = open(calib_path, "r")
        calib_info = json.load(calib_file)
        return calib_info

    def del_empty_info(self):
        inx = 0
        while inx < self.images.__len__():
            time_stmap = self.images[inx]
            time_stmap = time_stmap.split("/")[-1].split(".")[0]
            if len(self.annots[time_stmap]) == 0:
                del self.annots[time_stmap]
                del self.images[inx]
                inx -= 1
            inx += 1

    def get_lidar(self, idx):
        raise NotImplementedError

    def get_image(self, img_file, config):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        assert os.path.exists(img_file)
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_annos(self, idx):
        img_path = self.images[idx]
        time_stmap = img_path.split("/")[-1].split(".")[0]
        anns = None
        if time_stmap not in self.annots.keys():
            RuntimeWarning("Don't find gt info: " + time_stmap, sys._getframe().f_code.co_filename)
        else:
            anns = self.annots[time_stmap]
        return anns

    def get_depth_map(self, idx):
        raise NotImplementedError

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures
        self.load_from_dir()
        def get_alpha(rot_y, loc):
            loc = [loc[0], loc[1], loc[2]] # todo match with kitti
            beta = math.atan(loc[0] / loc[2])
            alpha = rot_y - beta
            if alpha > math.pi:
                alpha = alpha - math.pi * 2
            elif alpha < -math.pi:
                alpha = alpha + math.pi * 2
            return alpha

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_path': self.images[sample_idx], 'image_shape': np.array(self.img_sample.shape[:2], dtype=np.int32)}
            info['image'] = image_info
            calib_info = self.calibs
            info['calib'] = calib_info
            obj_list = self.get_annos(sample_idx)
            if len(obj_list) > 0:
                annotations = {}
                annotations['name'] = np.array([self.class_names[obj["type"] - 1] for obj in obj_list])
                annotations['label'] = np.array([obj["type"] for obj in obj_list])
                annotations['dimensions'] = np.array([[obj["size"][0], obj["size"][1], obj["size"][2]] for obj in obj_list])
                annotations['location'] = np.array([[obj["Cam3Dpoints"][0], obj["Cam3Dpoints"][1], -obj["Cam3Dpoints"][2]]
                                                          for obj in obj_list])
                annotations['rotation_y'] = np.array([obj["pry_ZYX"][2] for obj in obj_list])
                annotations['pry_ZYX'] = np.array([obj["pry_ZYX"] for obj in obj_list])
                annotations['alpha'] = np.array([get_alpha(obj["pry_ZYX"][2], loc) for obj,loc in zip(obj_list, annotations['location'])])
                annotations['bbox2d'] = np.array([get_2d_bbox(np.array(obj["Cam3Dpoints"]),
                                                                 np.array(obj["size"]),
                                                                 np.array(obj["pry_ZYX"]),
                                                                 np.array(calib_info["camera_intrinsic"]).reshape(3,4)) for obj in obj_list])
                annotations['gt_boxes'] = []
                for obj in obj_list:
                    _, _, box_3d_8 = project_3D_to_image(np.array(obj["Cam3Dpoints"]),
                                                                 np.array(obj["size"]),
                                                                 np.array(obj["pry_ZYX"]),
                                                                 np.array(calib_info["camera_intrinsic"]).reshape(3,4))
                    annotations['gt_boxes'].append(box_3d_8[:,1:])
                annotations['gt_boxes'] = np.array(annotations['gt_boxes'])
                annotations['score'] = np.array([1 for obj in obj_list])
                annotations['difficulty'] = np.array([0 for obj in obj_list], np.int32)
                info['annos'] = annotations
            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        # info = process_single_scene(10)
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)


    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sense_infos) * self.total_epochs

        return len(self.sense_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.sense_infos)

        info = copy.deepcopy(self.sense_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        calib = info["calib"]["camera_intrinsic"]
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])

        input_dict = {
            'frame_id': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_boxes = np.concatenate([loc, dims, rots[...,np.newaxis]], axis = 1)
            label = annos['label']
            pry_ZYX = annos['pry_ZYX']
            input_dict.update({
                'gt_boxes': gt_boxes,
                'pry_ZYX': pry_ZYX,
                'location': loc,
                'dimensions': dims,
                'rotation_y': rots,
                'label': label
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox2d"]

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            if self.dataset_cfg.FOV_POINTS_ONLY:
                pts_rect = calib.lidar_to_rect(points[:, 0:3])
                fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                points = points[fov_flag]
            input_dict['points'] = points

        if "images" in get_item_list:
            input_dict['images'] = self.get_image(info['image']['image_path'], self.dataset_cfg)
            target, input_dict['images'] = get_target_info(input_dict['images'], self.dataset_cfg.LOAD_SIZE, len(self.class_names), calib,
                                         input_dict, annos['gt_boxes'], self.dataset_cfg)
            input_dict['target'] = target
        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        input_dict['image_shape'] = img_shape
        return input_dict


def create_sensetime_infos(dataset_cfg, class_names, data_path, save_path, workers=8):
    dataset = SenseDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('sensetime_infos_%s.pkl' % train_split)
    val_filename = save_path / ('sensetime_infos_%s.pkl' % val_split)
    # trainval_filename = save_path / 'sensetime_infos_trainval.pkl'
    # test_filename = save_path / 'sensetime_infos_test.pkl'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    sensetime_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(sensetime_infos_train, f)
    print('Sensetime info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    sensetime_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(sensetime_infos_val, f)
    print('Sensetime info val file is saved to %s' % val_filename)

    # with open(trainval_filename, 'wb') as f:
    #     pickle.dump(sensetime_infos_train + sensetime_infos_val, f)
    # print('Sensetime info trainval file is saved to %s' % trainval_filename)

    # dataset.set_split('test')
    # sensetime_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    # with open(test_filename, 'wb') as f:
    #     pickle.dump(sensetime_infos_test, f)
    # print('Sensetime info test file is saved to %s' % test_filename)

    # print('---------------Start create groundtruth database for data augmentation---------------')
    # dataset.set_split(train_split)
    # dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_sensetime_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_sensetime_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist', 'Bus'],
            data_path=ROOT_DIR / 'data' / 'sensetime',
            save_path=ROOT_DIR / 'data' / 'sensetime'
        )
