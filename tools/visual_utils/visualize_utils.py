import mayavi.mlab as mlab
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

box_colormap = [
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
]
label_dict = {0:"background", 1:"Car", 2:"Ped", 3:"Cycler", 4:"Truck"}
color_dict = {"0":[255, 255, 255], "1":[255, 255, 0], "2":[51, 153, 255], "3":[255, 0, 255], "4":[0, 0, 0]}

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def visualize_pts(pts, fig=None, bgcolor=(0, 0, 0), fgcolor=(1.0, 1.0, 1.0),
                  show_intensity=False, size=(600, 600), draw_origin=True):
    if pts is None:
        if fig is None:
            fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)
        return fig
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=fgcolor, engine=None, size=size)

    if show_intensity:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    else:
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='point',
                          colormap='gnuplot', scale_factor=1, figure=fig)
    if draw_origin:
        mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
        mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), tube_radius=0.1)
        mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), tube_radius=0.1)

    return fig


def draw_sphere_pts(pts, color=(0, 1, 0), fig=None, bgcolor=(0, 0, 0), scale_factor=0.2):
    if not isinstance(pts, np.ndarray):
        pts = pts.cpu().numpy()

    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(600, 600))

    if isinstance(color, np.ndarray) and color.shape[0] == 1:
        color = color[0]
        color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

    if isinstance(color, np.ndarray):
        pts_color = np.zeros((pts.__len__(), 4), dtype=np.uint8)
        pts_color[:, 0:3] = color
        pts_color[:, 3] = 255
        G = mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], np.arange(0, pts_color.__len__()), mode='sphere',
                          scale_factor=scale_factor, figure=fig)
        G.glyph.color_mode = 'color_by_scalar'
        G.glyph.scale_mode = 'scale_by_vector'
        G.module_manager.scalar_lut_manager.lut.table = pts_color
    else:
        mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], mode='sphere', color=color,
                      colormap='gnuplot', scale_factor=scale_factor, figure=fig)

    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='cube', scale_factor=0.2)
    mlab.plot3d([0, 3], [0, 0], [0, 0], color=(0, 0, 1), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 3], [0, 0], color=(0, 1, 0), line_width=3, tube_radius=None, figure=fig)
    mlab.plot3d([0, 0], [0, 0], [0, 3], color=(1, 0, 0), line_width=3, tube_radius=None, figure=fig)

    return fig


def draw_grid(x1, y1, x2, y2, fig, tube_radius=None, color=(0.5, 0.5, 0.5)):
    mlab.plot3d([x1, x1], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0], color=color, tube_radius=tube_radius, line_width=1, figure=fig)
    return fig


def draw_multi_grid_range(fig, grid_size=20, bv_range=(-60, -60, 60, 60)):
    for x in range(bv_range[0], bv_range[2], grid_size):
        for y in range(bv_range[1], bv_range[3], grid_size):
            fig = draw_grid(x, y, x + grid_size, y + grid_size, fig)

    return fig


def draw_scenes(points=None, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None):
    if points is not None:
        if not isinstance(points, np.ndarray):
            points = points.cpu().numpy()
    if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
        ref_boxes = ref_boxes.cpu().numpy()
    if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
        gt_boxes = gt_boxes.cpu().numpy()
    if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
        ref_scores = ref_scores.cpu().numpy()
    if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
        ref_labels = ref_labels.cpu().numpy()

    fig = visualize_pts(points)
    fig = draw_multi_grid_range(fig, bv_range=(0, -40, 80, 40))
    if gt_boxes is not None:
        corners3d = boxes_to_corners_3d(gt_boxes)
        fig = draw_corners3d(corners3d, fig=fig, color=(0, 0, 1), max_num=100)

    if ref_boxes is not None and len(ref_boxes) > 0:
        ref_corners3d = boxes_to_corners_3d(ref_boxes)
        if ref_labels is None:
            fig = draw_corners3d(ref_corners3d, fig=fig, color=(0, 1, 0), cls=ref_scores, max_num=100)
        else:
            for k in range(ref_labels.min(), ref_labels.max() + 1):
                cur_color = tuple(box_colormap[k % len(box_colormap)])
                mask = (ref_labels == k)
                fig = draw_corners3d(ref_corners3d[mask], fig=fig, color=cur_color, cls=ref_scores[mask], max_num=5)
    mlab.view(azimuth=-179, elevation=54.0, distance=104.0, roll=90.0)
    return fig


def draw_corners3d(corners3d, fig, color=(1, 1, 1), line_width=2, cls=None, tag='', max_num=500, tube_radius=None):
    """
    :param corners3d: (N, 8, 3)
    :param fig:
    :param color:
    :param line_width:
    :param cls:
    :param tag:
    :param max_num:
    :return:
    """
    import mayavi.mlab as mlab
    num = min(max_num, len(corners3d))
    for n in range(num):
        b = corners3d[n]  # (8, 3)

        if cls is not None:
            if isinstance(cls, np.ndarray):
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%.2f' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)
            else:
                mlab.text3d(b[6, 0], b[6, 1], b[6, 2], '%s' % cls[n], scale=(0.3, 0.3, 0.3), color=color, figure=fig)

        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

            i, j = k, k + 4
            mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                        line_width=line_width, figure=fig)

        i, j = 0, 5
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)
        i, j = 1, 4
        mlab.plot3d([b[i, 0], b[j, 0]], [b[i, 1], b[j, 1]], [b[i, 2], b[j, 2]], color=color, tube_radius=tube_radius,
                    line_width=line_width, figure=fig)

    return fig

# def draw_on_image(image, calib, points=None, gt_boxes=None, ref_boxes=None, ref_scores=None, ref_labels=None):
#     if points is not None:
#         if not isinstance(points, np.ndarray):
#             points = points.cpu().numpy()
#     if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
#         ref_boxes = ref_boxes.cpu().numpy()
#     if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
#         gt_boxes = gt_boxes.cpu().numpy()
#     if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
#         ref_scores = ref_scores.cpu().numpy()
#     if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
#         ref_labels = ref_labels.cpu().numpy()

#     if gt_boxes is not None:
#         corners3d = boxes_to_corners_3d(gt_boxes)
#         for i in range(len(corners3d)):
#             image = draw_one_frame_on_img(image, corners3d[i], color_dict, label_dict, 0, calib, pretty=True)

#     if ref_boxes is not None and len(ref_boxes) > 0:
#         ref_corners3d = boxes_to_corners_3d(ref_boxes)
#         for i in range(len(ref_corners3d)):
#             if ref_labels is None:
#                 label = 0
#             else:
#                 label = ref_labels[i]
#             image = draw_one_frame_on_img(image, ref_corners3d[i], color_dict, label_dict, label, calib, pretty=True)
#     return image

# def draw_one_frame_on_img(img_draw, corners3d, color_dict, label_dict, label, calib, pretty = True):
#     box_list = [(0, 1), (0, 2), (0, 4), (1, 5), (1, 3), (2, 6), (2, 3), (3, 7), (4, 6), (4, 5), (5, 7), (6, 7)]
#     rect_min_w_h = [50, 20]
#     if not str(label) in color_dict.keys():
#         obj_color = (255, 255, 255)
#     else:
#         obj_color = color_dict[str(label)]
#     plane_cor, plane_obj_cen = project_3D_to_image(corners3d, calib)
#     print(plane_cor)
#     if pretty:
#         img_draw = draw_color_on_contour_roi(img_draw, plane_cor, obj_color)
#     for point_pair in box_list:
#         cv2.line(img_draw, (int(plane_cor[0][point_pair[0]]), int(plane_cor[1][point_pair[0]])),
#                  (int(plane_cor[0][point_pair[1]]), int(plane_cor[1][point_pair[1]])), obj_color, 1)
#     for index in range(8):
#         cv2.circle(img_draw, (int(plane_cor[0][index]), int(plane_cor[1][index])), 2, (0, 255, 255), -1)
#     cv2.circle(img_draw, (int(plane_obj_cen[0]), int(plane_obj_cen[1])), 3, (255, 0, 255), -1)
#     if not label_dict == None:
#         left_corner_cor = []
#         image_label = img_draw.copy()
#         for index in range(8):
#             left_corner_cor.append([plane_cor[0][index], plane_cor[1][index]])
#         left_corner_cor.sort(key=lambda x: x[0])
#         left_corner_cor = left_corner_cor[0:2]
#         left_corner_cor.sort(key=lambda x: x[1])
#         left_corner_cor = left_corner_cor[0]
#         rect_left_top = (int(left_corner_cor[0]), int(left_corner_cor[1] - rect_min_w_h[1]))
#         rect_right_down = (int(left_corner_cor[0] + rect_min_w_h[0]), int(left_corner_cor[1]))
#         cv2.rectangle(image_label, rect_left_top, rect_right_down, (102, 178, 255), -1)
#         if label in label_dict:
#             text = label_dict[label]
#         else:
#             text = "None"
#         cv2.putText(image_label, text, (int(left_corner_cor[0]), int(left_corner_cor[1]) - 5), cv2.FONT_HERSHEY_DUPLEX,
#                     0.5, (0, 0, 0), 1, cv2.LINE_AA)
#         img_draw = cv2.addWeighted(img_draw, 0.4, image_label, 0.6, 0)
#     return img_draw

# def draw_one_bv_frame(corners3d, color_dict):
#     if plt.gcf().number > 1:
#         plt.close('all')
#     if plt.gcf().number < 1:
#         plt.figure()
#     plt.ion()
#     fig = plt.gcf()
#     ax = plt.gca()
#     fig.set_size_inches(5, 12.5)
#     # point_list = [(1, 3), (3, 7), (7, 5), (5, 1)]
#     point_list = [(3, 7), (7, 6), (6, 2), (2, 3)]
#     plt.cla()
#
#     for cat in dets:
#         for i in range(len(dets[cat])):
#             if dets[cat][i, -1] > center_thresh:
#                 dim_ = dets[cat][i, 5:8]
#                 loc_ = dets[cat][i, 8:11]
#                 rot_y = dets[cat][i, 11]
#                 loc = np.array([loc_[0], loc_[1] - dim_[0] / 2, loc_[2]])
#                 dim = np.array([dim_[2], dim_[0], dim_[1]])
#                 if not str(cat) in obj_color_dict.keys():
#                     obj_color = (1, 1, 1)
#                 else:
#                     obj_color = (obj_color_dict[str(cat)][2] / 255, obj_color_dict[str(cat)][1] / 255,
#                                  obj_color_dict[str(cat)][0] / 255)
#                 corner_point = self.generate_obj_cam_cor(np.array(loc), np.array(dim), np.array([rot_y, 0, 0]))
#                 for point_ in point_list:
#                     ax.plot((corner_point[0][point_[0]], corner_point[0][point_[1]]),
#                             (corner_point[2][point_[0]], corner_point[2][point_[1]]), color=obj_color)
#     ax.axis(xmin=-25, xmax=25)
#     ax.axis(ymin=0, ymax=100)
#     ax.grid()
#     plt.title("bird view")
#     plt.xlabel("horizontal distance/m")
#     plt.ylabel("vertical distance/m")
#     fig.canvas.draw()
#     img_from_mat = cv2.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
#     img_from_mat = cv2.resize(img_from_mat, (400, 1000))
#
#     return img_from_mat


# def project_3D_to_image(corner_point, calib):
#     calib_ = np.array(calib)[:, 0:3]
#     center_3d = corner_point[0]
#     for i in range(1, 4):
#         center_3d += corner_point[i]
#     center_3d = center_3d/4
#     center_3d = center_3d[np.newaxis,...]
#     corner_point = np.append(center_3d, corner_point, axis=0)
#     corner_point = corner_point.T
#     tmp = corner_point[0,...]
#     corner_point[0, ...] = corner_point[2,...]
#     corner_point[2, ...] = tmp

#     tmp = corner_point[0,...]
#     corner_point[0, ...] = corner_point[1,...]
#     corner_point[1, ...] = tmp

#     for i in range(9):
#         if corner_point[2, i] < 0:
#             corner_point[0, i] = -corner_point[0, i]
#             corner_point[1, i] = -corner_point[1, i]
#     corner_point = np.matmul(calib_, corner_point)
#     plane_cor = corner_point[0:2, :]
#     for i in range(9):
#         plane_cor[0][i] = corner_point[0][i] / corner_point[2][i]
#         plane_cor[1][i] = corner_point[1][i] / corner_point[2][i]
#     plane_cen = plane_cor[:, 0]
#     plane_cor = plane_cor[:, 1:]
#     #
#     return plane_cor, plane_cen

def draw_color_on_contour_roi(image, plane_cor, color):
    plane_list = [[2, 3, 7, 6], [0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [0, 2, 6, 4], [1, 3, 7 ,5]]
    contour = []
    image_draw_contour = image.copy()
    for plane in plane_list:
        contour_ = []
        for index in plane:
            contour_.append([plane_cor[0][index], plane_cor[1][index]])
        contour.append(np.array(contour_).reshape((-1,1,2)).astype(np.int32))
    for contour_ in contour:
        cv2.drawContours(image_draw_contour, [contour_], -1, color, thickness = -1)
    image = cv2.addWeighted(image, 0.75, image_draw_contour, 0.25, 0)
    return image



def draw_one_frame_on_img(img_draw, boxes, labels, calib, point_color, center_color, obj_color_dict, pretty = False, label_dict = None):
    img_draw = np.ascontiguousarray(img_draw, dtype=np.uint8)
    box_list = [(0, 1),(0, 2),(0, 4), (1, 5), (1, 3), (2, 6), (2, 3), (3, 7), (4, 6), (4, 5), (5, 7), (6, 7)]
    rect_min_w_h = [50, 20]
    objs = list(zip(torch.Tensor.cpu(boxes), labels))
    for obj in objs:
        if not obj[1] in obj_color_dict.keys():
            obj_color = (255, 255, 255)
        else:
            obj_color = obj_color_dict[obj[1]]
        plane_cor, plane_obj_cen = project_3D_to_image(np.array([-obj[0][1], -obj[0][2],obj[0][0]]), np.array([obj[0][4], obj[0][5], obj[0][3]]), np.array([-obj[0][6], 0, 0]), calib)
        if pretty:
            img_draw = draw_color_on_contour_roi(img_draw, plane_cor, obj_color)
        for point_pair in box_list:
            cv2.line(img_draw, (int(plane_cor[0][point_pair[0]]), int(plane_cor[1][point_pair[0]])), (int(plane_cor[0][point_pair[1]]), int(plane_cor[1][point_pair[1]])), obj_color, 1)
        for index in range(8):
            cv2.circle(img_draw, (int(plane_cor[0][index]), int(plane_cor[1][index])), 2, point_color, -1)
        cv2.circle(img_draw, (int(plane_obj_cen[0]), int(plane_obj_cen[1])), 3, center_color, -1)
        if not label_dict == None:
            left_corner_cor = []
            image_label = img_draw.copy()
            for index in range(8):
                left_corner_cor.append([plane_cor[0][index], plane_cor[1][index]])
            left_corner_cor.sort(key = lambda x: x[0])
            left_corner_cor = left_corner_cor[0:2]
            left_corner_cor.sort(key = lambda x:x[1])
            left_corner_cor = left_corner_cor[0]
            rect_left_top = (int(left_corner_cor[0]), int(left_corner_cor[1] - rect_min_w_h[1]))
            rect_right_down = (int(left_corner_cor[0] + rect_min_w_h[0]), int(left_corner_cor[1]))
            cv2.rectangle(image_label, rect_left_top, rect_right_down, (102, 178, 255), -1)
            if obj[1] in label_dict:
                text = label_dict[obj[1]]
            else:
                text = "None"
            cv2.putText(image_label, text, (int(left_corner_cor[0]), int(left_corner_cor[1]) - 5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            img_draw = cv2.addWeighted(img_draw, 0.4, image_label, 0.6, 0)

    return img_draw

# def draw_color_on_contour_roi(self, image, plane_cor, color):
#     plane_list = [[2, 3, 7, 6], [0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [0, 2, 6, 4], [1, 3, 7 ,5]]
#     contour = []
#     image_draw_contour = image.copy()
#     for plane in plane_list:
#         contour_ = []
#         for index in plane:
#             contour_.append([plane_cor[0][index], plane_cor[1][index]])
#         contour.append(np.array(contour_).reshape((-1,1,2)).astype(np.int32))
#     for contour_ in contour:
#         cv2.drawContours(image_draw_contour, [contour_], -1, color, thickness = -1)
#     image = cv2.addWeighted(image, 0.75, image_draw_contour, 0.25, 0)
#     return image
    

# def generate_obj_cam_cor(self, position, size, ZYX):

#     pitch = ZYX[0]
#     roll  = ZYX[1]
#     yaw   = ZYX[2]

#     R_roll  = np.array([[1             , 0           , 0            ],
#                         [0             , np.cos(roll), -np.sin(roll)],
#                         [0             , np.sin(roll), np.cos(roll)]])
#     R_pitch = np.array([[np.cos(pitch) , 0           , np.sin(pitch)],
#                         [0             , 1           , 0            ],
#                         [-np.sin(pitch), 0           , np.cos(pitch)]])
#     R_yaw   = np.array([[np.cos(yaw)   , -np.sin(yaw), 0            ],
#                         [np.sin(yaw)   , np.cos(yaw) , 0            ],
#                         [0             , 0           , 1            ]])
    
#     R = np.matmul(R_pitch, R_roll)
#     R = np.matmul(R_yaw, R)
#     size = 0.5 * size
#     arithm_list = []
#     for i in ['+','-']:
#         for j in ['+', '-']:
#             for k in ['+', '-']:
#                 arithm_list.append(i+j+k)
#     corner_point = np.array([0, 0, 0])
#     for arithm in arithm_list:
#         point = np.array([eval(str(0) + arithm[0] + str(size[0])),eval(str(0) + arithm[1] + str(size[1])),eval(str(0) + arithm[2] + str(size[2]))])
#         corner_point = np.vstack((corner_point, point))
#     corner_point = corner_point.T
#     corner_point = np.matmul(R, corner_point)
#     for i in range(9):
#         corner_point[:,i] = corner_point[:,i] + position
#     return corner_point


def generate_obj_cam_cor(position, size, ZYX):
# 
    pitch = ZYX[0]
    roll  = ZYX[1]
    yaw   = ZYX[2]
# 
    R_roll  = np.array([[1             , 0           , 0            ],
                        [0             , np.cos(roll), -np.sin(roll)],
                        [0             , np.sin(roll), np.cos(roll)]])
    R_pitch = np.array([[np.cos(pitch) , 0           , np.sin(pitch)],
                        [0             , 1           , 0            ],
                        [-np.sin(pitch), 0           , np.cos(pitch)]])
    R_yaw   = np.array([[np.cos(yaw)   , -np.sin(yaw), 0            ],
                        [np.sin(yaw)   , np.cos(yaw) , 0            ],
                        [0             , 0           , 1            ]])
    # 
    R = np.matmul(R_pitch, R_roll)
    R = np.matmul(R_yaw, R)
    size = 0.5 * size
    arithm_list = []
    for i in ['+','-']:
        for j in ['+', '-']:
            for k in ['+', '-']:
                arithm_list.append(i+j+k)
    corner_point = np.array([0, 0, 0])
    for arithm in arithm_list:
        point = np.array([eval(str(0) + arithm[0] + str(size[0])),eval(str(0) + arithm[1] + str(size[1])),eval(str(0) + arithm[2] + str(size[2]))])
        corner_point = np.vstack((corner_point, point))
    corner_point = corner_point.T
    corner_point = np.matmul(R, corner_point)
    for i in range(9):
        corner_point[:,i] = corner_point[:,i] + position
    return corner_point


def project_3D_to_image(position, size, ZYX, calib):
    calib_ = np.array(calib)
    calib_ = calib_[:, 0:3]
    corner_point = generate_obj_cam_cor(position, size, ZYX)
    for i in range(9):
        if corner_point[2, i] < 0:
            corner_point[0, i] = -corner_point[0, i]
            corner_point[1, i] = -corner_point[1, i]
    corner_point = np.matmul(calib_, corner_point)
    plane_cor = corner_point[0:2,:]
    for i in range(9):
        plane_cor[0][i] = corner_point[0][i] / corner_point[2][i]
        plane_cor[1][i] = corner_point[1][i] / corner_point[2][i]
    plane_cen = plane_cor[:,0]
    plane_cor = plane_cor[:,1:]
# 
    return plane_cor, plane_cen
