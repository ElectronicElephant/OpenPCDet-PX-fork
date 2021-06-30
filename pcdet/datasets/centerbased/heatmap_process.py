import copy

import numpy as np
import torch
import json
import cv2
import os
import math
from ..sensetime.sensetime_util import project_3D_to_image
def gaussian_radius(det_size, min_overlap=0.7):
  height, width = det_size

  a1  = 1
  b1  = (height + width)
  c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
  sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
  r1  = (b1 + sq1) / 2

  a2  = 4
  b2  = 2 * (height + width)
  c2  = (1 - min_overlap) * width * height
  sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
  r2  = (b2 + sq2) / 2

  a3  = 4 * min_overlap
  b3  = -2 * min_overlap * (height + width)
  c3  = (min_overlap - 1) * width * height
  sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
  r3  = (b3 + sq3) / 2
  return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

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
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_dense_reg(regmap, heatmap, center, value, radius, is_offset=False):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    value = np.array(value, dtype=np.float32).reshape(-1, 1, 1)
    dim = value.shape[0]
    reg = np.ones((dim, diameter * 2 + 1, diameter * 2 + 1), dtype=np.float32) * value
    if is_offset and dim == 2:
        delta = np.arange(diameter * 2 + 1) - radius
        reg[0] = reg[0] - delta.reshape(1, -1)
        reg[1] = reg[1] - delta.reshape(-1, 1)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_regmap = regmap[:, y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom,
                      radius - left:radius + right]
    masked_reg = reg[:, radius - top:radius + bottom,
                 radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        idx = (masked_gaussian >= masked_heatmap).reshape(
            1, masked_gaussian.shape[0], masked_gaussian.shape[1])
        masked_regmap = (1 - idx) * masked_regmap + idx * masked_reg
    regmap[:, y - top:y + bottom, x - left:x + right] = masked_regmap
    return regmap


def draw_msra_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0] + 0.5)
    mu_y = int(center[1] + 0.5)
    w, h = heatmap.shape[0], heatmap.shape[1]
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= h or ul[1] >= w or br[0] < 0 or br[1] < 0:
        return heatmap
    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g_x = max(0, -ul[0]), min(br[0], h) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], w) - ul[1]
    img_x = max(0, ul[0]), min(br[0], h)
    img_y = max(0, ul[1]), min(br[1], w)
    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
    return heatmap

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans

def image_process(img, input_size):
    c = np.array([img.shape[1] / 2., img.shape[0] / 2.])
    s = np.array([img.shape[1], img.shape[0]], dtype=np.int32) # w,h
    trans_input = get_affine_transform(
        c, s, 0, input_size)

    img = cv2.warpAffine(img, trans_input,
                         (input_size[1], input_size[0]),flags=cv2.INTER_LINEAR)

    return img, trans_input

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_hm(img_size, num_classes, anns, gt_bbox, cfg):
    use_mse_loss = cfg.MSE_LOSS
    num_objs = anns['label'].shape[0]
    out_put_size = [cfg.LOAD_SIZE[0]//cfg.DOWN_RATIO, cfg.LOAD_SIZE[1]//cfg.DOWN_RATIO]

    c = np.array([img_size[0] / 2., img_size[1] / 2.])
    s = np.array([cfg.LOAD_SIZE[1], cfg.LOAD_SIZE[0]], dtype=np.int32)
    out = np.array([out_put_size[0], out_put_size[1]], dtype=np.int32)
    trans = get_affine_transform(c, s, 0, out)

    heat_map = np.zeros([num_classes, img_size[1], img_size[0]], dtype=np.float32)
    regression = np.zeros([num_objs, 3, 8], dtype=np.float32)
    cls_ids = np.zeros([num_objs], dtype=np.float32)
    proj_points = np.zeros([num_objs, 2], dtype=np.float32)
    p_offsets = np.zeros([num_objs, 2], dtype=np.float32)
    dimensions = np.zeros([num_objs, 3], dtype=np.float32)
    locations = np.zeros([num_objs, 3], dtype=np.float32)
    rotys = np.zeros([num_objs], dtype=np.float32)
    reg_mask = np.zeros([num_objs], dtype=np.float32)
    flip_mask = np.zeros([num_objs], dtype=np.float32)


    draw_gaussian = draw_msra_gaussian if use_mse_loss else \
        draw_umich_gaussian

    for i in range(num_objs):
        cls = anns["label"][i] - 1
        _, plane_cen, _ = project_3D_to_image(anns["location"][i], anns['dimensions'][i],
                                              anns['pry_ZYX'][i], anns['calib'])
        plane_cen[0] = min(heat_map.shape[2] - 1, max(0, plane_cen[0]))
        plane_cen[1] = min(heat_map.shape[1] - 1, max(0, plane_cen[1]))
        center_int = plane_cen.astype(np.int32)
        center_offset = plane_cen - center_int
        box_2d = anns['gt_boxes2d'][i]
        h, w = box_2d[3] - box_2d[1], box_2d[2] - box_2d[0]
        radius = gaussian_radius((h, w))
        radius = max(0, int(radius))
        radius = min(heat_map.shape[1] - 1, heat_map.shape[2] - 1, int(radius))
        heat_map[cls] = draw_gaussian(heat_map[cls], center_int, radius)
        cls_ids[i] = cls
        regression[i] = gt_bbox[i]
        proj_points[i] = center_int
        p_offsets[i] = center_offset
        dimensions[i] = anns['dimensions'][i]
        locations[i] = anns['location'][i]
        rotys[i] = anns['rotation_y'][i]
        reg_mask[i] = 1
        flip_mask[i] = 1
    hm_tans = []
    for i in range(heat_map.shape[0]):
        hm_tans.append(cv2.warpAffine(heat_map[i], trans,
                         (out_put_size[1], out_put_size[0]), flags=cv2.INTER_LINEAR))
    heat_map = np.array(hm_tans)
    for i in range(proj_points.shape[0]):
        # sub_info = copy.deepcopy(proj_points[i])
        proj_points[i] = affine_transform(proj_points[i], trans)
        proj_points[i][0] = min(out_put_size[1] - 1, proj_points[i][0])
        proj_points[i][1] = min(out_put_size[0] - 1, proj_points[i][1])
        # if proj_points[i][0] > 240 or proj_points[i][1] > 148:
        #     print(proj_points)
        p_offsets[i] = affine_transform(p_offsets[i], trans)
    sub = np.zeros((1,3))
    sub[0][2] = 1
    trans = np.concatenate((trans, sub))
    target = {'hm': heat_map, 'reg': regression, 'cls_ids': cls_ids, 'proj_points': proj_points, 'dimensions': dimensions,
           'locations': locations, 'rotys': rotys, "reg_mask": reg_mask, "flip_mask": flip_mask, "p_offsets": p_offsets,
              "trans_mat": trans}
    return target


def get_target_info(img, input_size, num_classes, calib, anns, gt_bbox, cfg):
    height, width = img.shape[0], img.shape[1]
    img, trans_input = image_process(img, input_size)
    target = get_hm([width, height], num_classes, anns,
                    gt_bbox, cfg)

    # target["trans_mat"] = trans_input
    target["K"] = calib[:,:3]
    target["size"] = np.array([width, height], dtype=np.float32)
    return target, img