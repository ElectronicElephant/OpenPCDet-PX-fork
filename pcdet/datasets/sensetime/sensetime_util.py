import numpy as np
from ...utils import box_utils


def transform_annotations_to_kitti_format(annos, map_name_to_kitti=None, info_with_fakelidar=False):
    """
    Args:
        annos:
        map_name_to_kitti: dict, map name to KITTI names (Car, Pedestrian, Cyclist)
        info_with_fakelidar:
    Returns:

    """
    for anno in annos:
        for k in range(anno['name'].shape[0]):
            anno['name'][k] = map_name_to_kitti[anno['name'][k]]

        anno['bbox'] = np.zeros((len(anno['name']), 4))
        anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
        anno['truncated'] = np.zeros(len(anno['name']))
        anno['occluded'] = np.zeros(len(anno['name']))
        if 'boxes_lidar' in anno:
            gt_boxes_lidar = anno['boxes_lidar'].copy()
        else:
            gt_boxes_lidar = anno['gt_boxes_lidar'].copy()

        if len(gt_boxes_lidar) > 0:
            if info_with_fakelidar:
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

            gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
            anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
            anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
            anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
            anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
            dxdydz = gt_boxes_lidar[:, 3:6]
            anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
            anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
            anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
        else:
            anno['location'] = anno['dimensions'] = np.zeros((0, 3))
            anno['rotation_y'] = anno['alpha'] = np.zeros(0)

    return annos


def calib_to_matricies(calib):
    """
    Converts calibration object to transformation matricies
    Args:
        calib: calibration.Calibration, Calibration object
    Returns
        V2R: (4, 4), Lidar to rectified camera transformation matrix
        P2: (3, 4), Camera projection matrix
    """
    V2C = np.vstack((calib.V2C, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    R0 = np.hstack((calib.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
    R0 = np.vstack((R0, np.array([0, 0, 0, 1], dtype=np.float32)))  # (4, 4)
    V2R = R0 @ V2C
    P2 = calib.P2
    return V2R, P2

def project_3D_to_image(position, size, ZYX, calib):
    if calib.shape[1] !=3:
        calib = calib[:,:3]
    corner_point = generate_obj_cam_cor(position, size, ZYX)
    for i in range(9):
        if corner_point[2, i] < 0:
            corner_point[0, i] = -corner_point[0, i]
            corner_point[1, i] = -corner_point[1, i]
    corner_point_3d = corner_point.copy()
    corner_point = np.matmul(calib, corner_point)
    plane_cor = corner_point[0:2, :]
    for i in range(9):
        plane_cor[0][i] = corner_point[0][i] / corner_point[2][i]
        plane_cor[1][i] = corner_point[1][i] / corner_point[2][i]
    plane_cen = plane_cor[:, 0]
    plane_cor = plane_cor[:, 1:]
    return plane_cor, plane_cen, corner_point_3d

def generate_obj_cam_cor(position, size, ZYX):
    pitch = ZYX[0]
    roll = ZYX[1]
    yaw = ZYX[2]

    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
    R_pitch = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                        [0, 1, 0],
                        [-np.sin(pitch), 0, np.cos(pitch)]])
    R_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])

    R = np.matmul(R_pitch, R_roll)
    R = np.matmul(R_yaw, R)
    size = 0.5 * size
    arithm_list = []
    for i in ['+', '-']:
        for j in ['+', '-']:
            for k in ['+', '-']:
                arithm_list.append(i + j + k)
    corner_point = np.array([0, 0, 0])
    for arithm in arithm_list:
        point = np.array([eval(str(0) + arithm[0] + str(size[0])), eval(str(0) + arithm[1] + str(size[1])),
                          eval(str(0) + arithm[2] + str(size[2]))])
        corner_point = np.vstack((corner_point, point))
    corner_point = corner_point.T
    corner_point = np.matmul(R, corner_point)
    for i in range(9):
        corner_point[:, i] = corner_point[:, i] + position
    return corner_point

def _8_box_2_bbox(box_8):
    xmin = box_8[0][0]
    ymin = box_8[1][0]
    xmax = box_8[0][0]
    ymax = box_8[1][0]
    for inx in range(box_8.shape[1]):
        xmin = min(xmin, box_8[0][inx])
        ymin = min(ymin, box_8[1][inx])
        xmax = max(xmax, box_8[0][inx])
        ymax = max(ymax, box_8[1][inx])
    bbox = np.array([xmin, ymin, xmax, ymax],
                    dtype=np.float32)
    return bbox

def get_2d_bbox(position, size, ZYX, calib):
    box_8, center, box_3d_8 = project_3D_to_image(position, size, ZYX, calib)
    box_2d = _8_box_2_bbox(box_8)
    return box_2d