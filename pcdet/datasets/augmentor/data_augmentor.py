from functools import partial
from functools import cmp_to_key
import math
import random
import copy
from collections.abc import Sequence

import mmcv
import numpy as np
import torch
import PIL.Image
import PIL.ImageDraw
from mmcv.parallel import DataContainer as DC
from shapely.geometry import Polygon, Point, LineString, Point, MultiLineString

from ...utils import common_utils
from . import augmentor_utils, database_sampler


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

def cal_dis(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def convert_list(p, downscale=None):
    xy = list()
    if downscale is None:
        for i in range(len(p) // 2):
            xy.append((p[2 * i], p[2 * i + 1]))
    else:
        for i in range(len(p) // 2):
            xy.append((p[2 * i] / downscale, p[2 * i + 1] / downscale))
    return xy

def clamp_line(line, box, min_length=0):
    left, top, right, bottom = box
    loss_box = Polygon([[left, top], [right, top], [right, bottom],
                        [left, bottom]])
    line_coords = np.array(line).reshape((-1, 2))
    if line_coords.shape[0] < 2:
        return None
    try:
        line_string = LineString(line_coords)
        I = line_string.intersection(loss_box)
        if I.is_empty:
            return None
        if I.length < min_length:
            return None
        if isinstance(I, LineString):

            pts = list(I.coords)
            return pts
        elif isinstance(I, MultiLineString):
            pts = []
            Istrings = list(I)
            for Istring in Istrings:
                pts += list(Istring.coords)
            return pts
    except:
        return None

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius -
                               left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(
            masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def select_mask_points(ct, r, shape, max_sample=5):

    def in_range(pt, w, h):
        if pt[0] >= 0 and pt[0] < w and pt[1] >= 0 and pt[1] < h:
            return True
        else:
            return False

    h, w = shape[:2]
    valid_points = []
    r = max(int(r // 2), 1)
    start_x, end_x = ct[0] - r, ct[0] + r
    start_y, end_y = ct[1] - r, ct[1] + r
    for x in range(start_x, end_x + 1):
        for y in range(start_y, end_y + 1):
            if x == ct[0] and y == ct[1]:
                continue
            if in_range((x, y), w, h) and cal_dis((x, y), ct) <= r + 0.1:
                valid_points.append([x, y])
    if len(valid_points) > max_sample - 1:
        valid_points = random.sample(valid_points, max_sample - 1)
    valid_points.append([ct[0], ct[1]])
    return valid_points

def extend_line(line, dis=10):
    extended = copy.deepcopy(line)
    start = line[1]
    end = line[0]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    norm = math.sqrt(dx**2 + dy**2)
    dx = dx / norm
    dy = dy / norm
    extend_point = (start[0] + dx * dis, start[1] + dy * dis)
    extended.insert(0, extend_point)
    return extended

def draw_label(mask,
               polygon_in,
               val,
               shape_type='polygon',
               width=3,
               convert=False):
    polygon = copy.deepcopy(polygon_in)
    mask = PIL.Image.fromarray(mask)
    xy = []
    if convert:
        for i in range(len(polygon) // 2):
            xy.append((polygon[2 * i], polygon[2 * i + 1]))
    else:
        for i in range(len(polygon)):
            xy.append((polygon[i][0], polygon[i][1]))

    if shape_type == 'polygon':
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=val, fill=val)
    else:
        PIL.ImageDraw.Draw(mask).line(xy=xy, fill=val, width=width)
    mask = np.array(mask, dtype=np.uint8)
    return mask

def get_line_intersection(x, y, line):
    def in_line_range(val, start, end):
        s = min(start, end)
        e = max(start, end)
        if val >= s and val <= e and s != e:
            return True
        else:
            return False

    def choose_min_reg(val, ref):
        min_val = 1e5
        index = -1
        if len(val) == 0:
            return None
        else:
            for i, v in enumerate(val):
                if abs(v - ref) < min_val:
                    min_val = abs(v - ref)
                    index = i
        return val[index]

    reg_y = []
    reg_x = []

    for i in range(len(line) - 1):
        point_start, point_end = line[i], line[i + 1]
        if in_line_range(x, point_start[0], point_end[0]):
            k = (point_end[1] - point_start[1]) / (
                point_end[0] - point_start[0])
            reg_y.append(k * (x - point_start[0]) + point_start[1])
    reg_y = choose_min_reg(reg_y, y)

    for i in range(len(line) - 1):
        point_start, point_end = line[i], line[i + 1]
        if in_line_range(y, point_start[1], point_end[1]):
            k = (point_end[0] - point_start[0]) / (
                point_end[1] - point_start[1])
            reg_x.append(k * (y - point_start[1]) + point_start[0])
    reg_x = choose_min_reg(reg_x, x)
    return reg_x, reg_y

class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def Normalize(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.Normalize, config=config)

        data_dict['img'] = mmcv.imnormalize(data_dict['img'], config['MEAN'], config['STD'], config['TO_RGB'])
        data_dict['img_norm_cfg'] = dict(
            mean=config['MEAN'], std=config['STD'], to_rgb=config['TO_RGB'])
        
        return data_dict
    
    def DefaultFormatBundle(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.DefaultFormatBundle, config=config)

        if 'img' in data_dict:
            img = data_dict['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            data_dict['img'] = DC(to_tensor(img), stack=True)
        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in data_dict:
                continue
            data_dict[key] = DC(to_tensor(data_dict[key]))
        if 'gt_masks' in data_dict:
            data_dict['gt_masks'] = DC(data_dict['gt_masks'], cpu_only=True)
        if 'gt_semantic_seg' in data_dict:
            data_dict['gt_semantic_seg'] = DC(
                to_tensor(data_dict['gt_semantic_seg'][None, ...]), stack=True)
        return data_dict

    def CollectLane(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.CollectLane, config=config)
        
        def min_dis_one_point(points, idx):
            min_dis = 1e6
            for i in range(len(points)):
                if i == idx:
                    continue
                else:
                    d = cal_dis(points[idx], points[i])
                    if d < min_dis:
                        min_dis = d
            return min_dis

        data = {}
        img_meta = {}
        
        output_h = int(data_dict['img_shape'][0])
        output_w = int(data_dict['img_shape'][1])
        mask_h = int(output_h // config['DOWN_SCALE'])
        mask_w = int(output_w // config['DOWN_SCALE'])
        hm_h = int(output_h // config['HM_DOWN_SCALE'])
        hm_w = int(output_w // config['HM_DOWN_SCALE'])
        data_dict['hm_shape'] = [hm_h, hm_w]
        data_dict['mask_shape'] = [mask_h, mask_w]
        ratio_hm_mask = config['DOWN_SCALE'] / config['HM_DOWN_SCALE']

        # gt init
        gt_hm = np.zeros((1, hm_h, hm_w), np.float32)
        gt_masks = []

        # gt heatmap and ins of bank
        gt_points = data_dict['gt_points']
        valid_gt = []
        for pts in gt_points:
            id_class = 1

            pts = convert_list(pts, config['DOWN_SCALE'])
            pts = sorted(pts, key=cmp_to_key(lambda a, b: b[-1] - a[-1]))
            pts = clamp_line(
                pts, box=[0, 0, mask_w - 1, mask_h - 1], min_length=1)
            if pts is not None and len(pts) > 1:
                valid_gt.append([pts, id_class - 1])

        # draw gt_hm_lane
        gt_hm_lane_ends = []
        radius = []
        for l in valid_gt:
            label = l[1]
            point = (l[0][0][0] * ratio_hm_mask, l[0][0][1] * ratio_hm_mask)
            gt_hm_lane_ends.append([point, l[0]])
        for i, p in enumerate(gt_hm_lane_ends):
            r = config['RADIUS']
            radius.append(r)

        if len(gt_hm_lane_ends) >= 2:
            endpoints = [p[0] for p in gt_hm_lane_ends]
            for j in range(len(endpoints)):
                dis = min_dis_one_point(endpoints, j)
                if dis < 1.5 * radius[j]:
                    radius[j] = int(max(dis / 1.5, 1) + 0.49999)
        for (end_point, line), r in zip(gt_hm_lane_ends, radius):
            pos = np.zeros((mask_h), np.float32)
            pos_mask = np.zeros((mask_h), np.float32)
            pt_int = [int(end_point[0]), int(end_point[1])]
            draw_umich_gaussian(gt_hm[0], pt_int, r)
            line_array = np.array(line)
            y_min, y_max = int(np.min(line_array[:, 1])), int(
                np.max(line_array[:, 1]))
            mask_points = select_mask_points(
                pt_int, r, (hm_h, hm_w), max_sample=config['MAX_MASK_SAMPLE'])
            reg = np.zeros((1, mask_h, mask_w), np.float32)
            reg_mask = np.zeros((1, mask_h, mask_w), np.float32)

            extended_line = extend_line(line)
            line_array = np.array(line)
            y_min, y_max = np.min(line_array[:, 1]), np.max(line_array[:, 1])
            # regression
            m = np.zeros((mask_h, mask_w), np.uint8)
            lane_range = np.zeros((1, mask_h), np.int64)
            line_array = np.array(line)

            polygon = np.array(extended_line)
            polygon_map = draw_label(
                m, polygon, 1, 'line', width=config['LINE_WIDTH'] + 9) > 0
            for y in range(polygon_map.shape[0]):
                for x in np.where(polygon_map[y, :])[0]:
                    reg_x, _ = get_line_intersection(x, y, line)
                    # kps and kps_mask:
                    if reg_x is not None:
                        offset = reg_x - x
                        reg[0, y, x] = offset
                        if abs(offset) < 10:
                            reg_mask[0, y, x] = 1
                        if y >= y_min and y <= y_max:
                            pos[y] = reg_x
                            pos_mask[y] = 1
                        lane_range[:, y] = 1

            gt_masks.append({
                'reg': reg,
                'reg_mask': reg_mask,
                'points': mask_points,
                'row': pos,
                'row_mask': pos_mask,
                'range': lane_range,
                'label': 0
            })

        data_dict['gt_hm'] = DC(
            to_tensor(gt_hm).float(), stack=True, pad_dims=None)
        data_dict['gt_masks'] = gt_masks
        data_dict['down_scale'] = config['DOWN_SCALE']
        data_dict['hm_down_scale'] = config['HM_DOWN_SCALE']
        valid = True

        if not valid:
            return None
        for key in config['META_KEYS']:
            img_meta[key] = data_dict[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in config['KEYS']:
            data[key] = data_dict[key]
        return data

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        return data_dict

    def random_image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']
            images, depth_maps, gt_boxes = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                images, depth_maps, gt_boxes, calib,
            )

        data_dict['images'] = images
        data_dict['depth_maps'] = depth_maps
        data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)

        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]

            data_dict.pop('gt_boxes_mask')
        return data_dict
