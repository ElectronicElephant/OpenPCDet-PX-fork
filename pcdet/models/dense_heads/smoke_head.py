import torch
from torch.nn import functional as F
import torch.nn as nn
import math

from ..model_utils.weight_process import _fill_fc_weights, _HEAD_NORM_SPECS
from ..model_utils.center_based_utils import sigmoid_hm, select_point_of_interest, nms_hm, select_topk

def get_channel_spec(reg_channels, name):
    if name == "dim":
        s = sum(reg_channels[:2])
        e = sum(reg_channels[:3])
    elif name == "ori":
        s = sum(reg_channels[:3])
        e = sum(reg_channels)

    return slice(s, e, 1)

class SMOKEPredictor(nn.Module):
    def __init__(self, heads, in_channels):
        super(SMOKEPredictor, self).__init__()

        classes = len(heads["class"])
        regression = heads["reg"]
        regression_channels = heads["reg_c"]
        head_conv = 256
        norm_func = _HEAD_NORM_SPECS[heads["norm"]]

        assert sum(regression_channels) == regression, \
            "the sum of {} must be equal to regression channel of {}".format(
                regression, regression_channels
            )

        self.dim_channel = get_channel_spec(regression_channels, name="dim")
        self.ori_channel = get_channel_spec(regression_channels, name="ori")

        self.class_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      head_conv,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            norm_func(head_conv),

            nn.ReLU(inplace=True),

            nn.Conv2d(head_conv,
                      classes,
                      kernel_size=1,
                      padding=1 // 2,
                      bias=True)
        )

        # todo: what is datafill here
        self.class_head[-1].bias.data.fill_(-2.19)

        self.regression_head = nn.Sequential(
            nn.Conv2d(in_channels,
                      head_conv,
                      kernel_size=3,
                      padding=1,
                      bias=True),

            norm_func(head_conv),

            nn.ReLU(inplace=True),

            nn.Conv2d(head_conv,
                      regression,
                      kernel_size=1,
                      padding=1 // 2,
                      bias=True)
        )
        _fill_fc_weights(self.regression_head)

    def forward(self, features):
        head_class = self.class_head(features)
        head_regression = self.regression_head(features)

        head_class = sigmoid_hm(head_class)
        # (N, C, H, W)
        offset_dims = head_regression[:, self.dim_channel, ...].clone()
        head_regression[:, self.dim_channel, ...] = torch.sigmoid(offset_dims) - 0.5

        vector_ori = head_regression[:, self.ori_channel, ...].clone()
        head_regression[:, self.ori_channel, ...] = F.normalize(vector_ori)

        return [head_class, head_regression]

class SMOKECoder():
    def __init__(self, depth_ref, dim_ref, device="cuda"):
        self.depth_ref = torch.as_tensor(depth_ref).to(device=device)
        self.dim_ref = torch.as_tensor(dim_ref).to(device=device)

    def encode_box2d(self, K, rotys, dims, locs, img_size):
        device = rotys.device
        K = K.to(device=device)

        img_size = img_size.flatten()

        box3d = self.encode_box3d(rotys, dims, locs)
        N = box3d.shape[0]
        batch_size = rotys.shape[0]
        K = K.type(box3d.type())
        K = K.repeat(N//batch_size, 1, 1).view(-1, 3, 3)
        box3d_image = torch.matmul(K, box3d)
        box3d_image = box3d_image[:, :2, :] / box3d_image[:, 2, :].view(
            box3d.shape[0], 1, box3d.shape[2]
        )

        xmins, _ = box3d_image[:, 0, :].min(dim=1)
        xmaxs, _ = box3d_image[:, 0, :].max(dim=1)
        ymins, _ = box3d_image[:, 1, :].min(dim=1)
        ymaxs, _ = box3d_image[:, 1, :].max(dim=1)

        xmins = xmins.clamp(0, img_size[0])
        xmaxs = xmaxs.clamp(0, img_size[0])
        ymins = ymins.clamp(0, img_size[1])
        ymaxs = ymaxs.clamp(0, img_size[1])

        bboxfrom3d = torch.cat((xmins.unsqueeze(1), ymins.unsqueeze(1),
                                xmaxs.unsqueeze(1), ymaxs.unsqueeze(1)), dim=1)

        return bboxfrom3d

    @staticmethod
    def rad_to_matrix(rotys, N):
        device = rotys.device

        cos, sin = rotys.cos(), rotys.sin()

        i_temp = torch.tensor([[1, 0, 1],
                               [0, 1, 0],
                               [-1, 0, 1]]).to(dtype=torch.float32,
                                               device=device)
        ry = i_temp.repeat(N, 1).view(N, -1, 3)

        ry[:, 0, 0] *= cos.squeeze()
        ry[:, 0, 2] *= sin.squeeze()
        ry[:, 2, 0] *= sin.squeeze()
        ry[:, 2, 2] *= cos.squeeze()

        return ry

    def encode_box3d(self, rotys, dims, locs):
        '''
        construct 3d bounding box for each object.
        Args:
            rotys: rotation in shape N
            dims: dimensions of objects
            locs: locations of objects

        Returns:

        '''
        rotys = rotys.view(-1, 1)
        dims = dims.view(-1, 3)
        locs = locs.view(-1, 3)

        device = rotys.device
        N = rotys.shape[0]
        ry = self.rad_to_matrix(rotys, N)

        dims = dims.view(-1, 1).repeat(1, 8)
        dims[::3, :4], dims[2::3, :4] = 0.5 * dims[::3, :4], 0.5 * dims[2::3, :4]
        dims[::3, 4:], dims[2::3, 4:] = -0.5 * dims[::3, 4:], -0.5 * dims[2::3, 4:]
        dims[1::3, :4], dims[1::3, 4:] = 0., -dims[1::3, 4:]
        index = torch.tensor([[4, 0, 1, 2, 3, 5, 6, 7],
                              [4, 5, 0, 1, 6, 7, 2, 3],
                              [4, 5, 6, 0, 1, 2, 3, 7]]).repeat(N, 1).to(device=device)
        box_3d_object = torch.gather(dims, 1, index)
        box_3d = torch.matmul(ry, box_3d_object.view(N, 3, -1))
        box_3d += locs.unsqueeze(-1).repeat(1, 1, 8)

        return box_3d

    def decode_depth(self, depths_offset):
        '''
        Transform depth offset to depth
        '''
        device = depths_offset.device
        self.depth_ref[0] = self.depth_ref[0].to(device=device)
        self.depth_ref[1] = self.depth_ref[1].to(device=device)
        depth = depths_offset * self.depth_ref[1] + self.depth_ref[0]

        return depth

    def decode_location(self,
                        points,
                        points_offset,
                        depths,
                        Ks,
                        trans_mats):
        '''
        retrieve objects location in camera coordinate based on projected points
        Args:
            points: projected points on feature map in (x, y)
            points_offset: project points offset in (delata_x, delta_y)
            depths: object depth z
            Ks: camera intrinsic matrix, shape = [N, 3, 3]
            trans_mats: transformation matrix from image to feature map, shape = [N, 3, 3]

        Returns:
            locations: objects location, shape = [N, 3]
        '''
        # batch_size = trans_mats.shape[0]
        # sub = torch.zeros(batch_size, 1, 3).float().to(trans_mats.device)
        # sub[..., 2] = 1
        # if (trans_mats.shape[1] == 2):
        #     trans_mats = torch.cat((trans_mats, sub), 1)
        device = points.device
        tensor_type = depths.type()
        points = points.type(tensor_type)
        points_offset = points_offset.type(tensor_type)
        depths = depths.type(tensor_type)
        Ks = Ks.type(tensor_type)
        trans_mats = trans_mats.type(tensor_type)

        points = points.to(device=device)
        points_offset = points_offset.to(device=device)
        depths = depths.to(device=device)
        Ks = Ks.to(device=device)
        trans_mats = trans_mats.to(device=device)

        # number of points
        N = points_offset.shape[1]
        # batch size
        N_batch = Ks.shape[0]
        Ks_inv = Ks.inverse()

        proj_points = points.type(tensor_type) + points_offset.type(tensor_type)
        # transform project points in homogeneous form.
        proj_points_extend = torch.cat(
            (proj_points, torch.ones(N_batch, N, 1).type(tensor_type).to(device=device)), dim=2)
        # transform project points back on image
        proj_points_extend = proj_points_extend.type_as(trans_mats)
        trans_mats = trans_mats.repeat(N, 1, 1)
        proj_points_extend = proj_points_extend.view(-1, 3, 1)
        proj_points_img = torch.matmul(trans_mats, proj_points_extend)
        # with depth
        proj_points_img = proj_points_img * depths.view(-1, 1, 1)
        # transform image coordinates back to object locations
        Ks_inv = Ks_inv.type_as(proj_points_img)
        Ks_inv = Ks_inv.repeat(N, 1, 1)
        locations = torch.matmul(Ks_inv, proj_points_img)

        return locations.squeeze(2)

    def decode_dimension(self, cls_id, dims_offset):
        '''
        retrieve object dimensions
        Args:
            cls_id: each object id
            dims_offset: dimension offsets, shape = (N, 3)

        Returns:

        '''
        cls_id = cls_id.flatten().long()

        dims_select = self.dim_ref[cls_id, :]
        dims_offset = dims_offset.view(-1, 3)
        dims_select = dims_select.to(device=dims_offset.device)
        dimensions = dims_offset.exp() * dims_select

        return dimensions

    def decode_orientation(self, vector_ori, locations, flip_mask=None):
        '''
        retrieve object orientation
        Args:
            vector_ori: local orientation in [sin, cos] format
            locations: object location

        Returns: for training we only need roty
                 for testing we need both alpha and roty

        '''
        device = locations.device
        tensor_type = locations.type()

        vector_ori = vector_ori.type(tensor_type)
        vector_ori = vector_ori.to(device=device)

        locations = locations.view(-1, 3)
        vector_ori = vector_ori.view(-1, 2)
        rays = torch.atan(locations[:, 0] / (locations[:, 2] + 1e-7))
        alphas = torch.atan(vector_ori[:, 0] / (vector_ori[:, 1] + 1e-7))

        # get cosine value positive and negtive index.
        cos_pos_idx = (vector_ori[:, 1] >= 0).nonzero()
        cos_neg_idx = (vector_ori[:, 1] < 0).nonzero()

        alphas[cos_pos_idx] -= math.pi / 2
        alphas[cos_neg_idx] += math.pi / 2

        # retrieve object rotation y angle.
        rotys = alphas + rays

        # in training time, it does not matter if angle lies in [-PI, PI]
        # it matters at inference time? todo: does it really matter if it exceeds.
        larger_idx = (rotys > math.pi).nonzero()
        small_idx = (rotys < -math.pi).nonzero()

        if len(larger_idx) != 0:
            rotys[larger_idx] -= 2 * math.pi
        if len(small_idx) != 0:
            rotys[small_idx] += 2 * math.pi

        if flip_mask is not None:
            flip_mask = flip_mask.view(-1, 1)
            fm = flip_mask.flatten()
            rotys_flip = fm.float() * rotys

            rotys_flip_pos_idx = rotys_flip > 0
            rotys_flip_neg_idx = rotys_flip < 0
            rotys_flip[rotys_flip_pos_idx] -= math.pi
            rotys_flip[rotys_flip_neg_idx] += math.pi

            rotys_all = fm.float() * rotys_flip + (1 - fm.float()) * rotys

            return rotys_all, alphas

        else:
            return rotys, alphas

class SMOKELossComputation():
    def __init__(self,
                 smoke_coder,
                 cls_loss,
                 reg_loss,
                 loss_weight,
                 max_objs):
        self.smoke_coder = smoke_coder
        self.cls_loss = cls_loss
        self.reg_loss = reg_loss
        self.loss_weight = loss_weight
        self.max_objs = max_objs

    def prepare_targets(self, targets):
        heatmaps = targets["hm"]
        regression = targets["reg"]
        cls_ids = targets["cls_ids"]
        proj_points = targets["proj_points"]
        dimensions = targets["dimensions"]
        locations = targets["locations"]
        rotys = targets["rotys"]
        trans_mat = targets["trans_mat"]
        K = targets["K"]
        reg_mask = targets["reg_mask"]
        flip_mask = targets["flip_mask"]

        return heatmaps, regression, dict(cls_ids=cls_ids,
                                          proj_points=proj_points,
                                          dimensions=dimensions,
                                          locations=locations,
                                          rotys=rotys,
                                          trans_mat=trans_mat,
                                          K=K,
                                          reg_mask=reg_mask,
                                          flip_mask=flip_mask)

    def prepare_predictions(self, targets_variables, pred_regression):
        batch, channel = pred_regression.shape[0], pred_regression.shape[1]
        targets_proj_points = targets_variables["proj_points"]

        # obtain prediction from points of interests
        pred_regression_pois = select_point_of_interest(
            batch, targets_proj_points, pred_regression
        )
        # pred_regression_pois = pred_regression_pois.view(-1, channel)

        # FIXME: fix hard code here
        pred_depths_offset = pred_regression_pois[..., 0:1]
        pred_proj_offsets = pred_regression_pois[..., 1:3]
        pred_dimensions_offsets = pred_regression_pois[..., 3:6]
        pred_orientation = pred_regression_pois[..., 6:]

        pred_depths = self.smoke_coder.decode_depth(pred_depths_offset)
        pred_locations = self.smoke_coder.decode_location(
            targets_proj_points,
            pred_proj_offsets,
            pred_depths,
            targets_variables["K"],
            targets_variables["trans_mat"]
        )
        pred_dimensions = self.smoke_coder.decode_dimension(
            targets_variables["cls_ids"],
            pred_dimensions_offsets,
        )
        # we need to change center location to bottom location
        pred_locations[:, 1] += pred_dimensions[:, 1] / 2

        pred_rotys, pred_alphas = self.smoke_coder.decode_orientation(
            pred_orientation,
            targets_variables["locations"],
            targets_variables["flip_mask"]
        )
        pred_rotys = pred_rotys.view(batch, -1, 1)
        pred_alphas = pred_alphas.view(batch, -1, 1)
        if self.reg_loss == "DisL1":
            pred_box3d_rotys = self.smoke_coder.encode_box3d(
                pred_rotys,
                targets_variables["dimensions"],
                targets_variables["locations"]
            )
            pred_box3d_dims = self.smoke_coder.encode_box3d(
                targets_variables["rotys"],
                pred_dimensions,
                targets_variables["locations"]
            )
            pred_box3d_locs = self.smoke_coder.encode_box3d(
                targets_variables["rotys"],
                targets_variables["dimensions"],
                pred_locations
            )

            return dict(ori=pred_box3d_rotys,
                        dim=pred_box3d_dims,
                        loc=pred_box3d_locs, )

        elif self.reg_loss == "L1":
            pred_box_3d = self.smoke_coder.encode_box3d(
                pred_rotys,
                pred_dimensions,
                pred_locations
            )
            return pred_box_3d

    def __call__(self, predictions, targets):
        pred_heatmap, pred_regression = predictions[0], predictions[1]

        targets_heatmap, targets_regression, targets_variables \
            = self.prepare_targets(targets)

        predict_boxes3d = self.prepare_predictions(targets_variables, pred_regression)

        hm_loss = self.cls_loss(pred_heatmap, targets_heatmap) * self.loss_weight[0]

        targets_regression = targets_regression.view(
            -1, targets_regression.shape[2], targets_regression.shape[3]
        )

        reg_mask = targets_variables["reg_mask"].flatten()
        reg_mask = reg_mask.view(-1, 1, 1)
        reg_mask = reg_mask.expand_as(targets_regression)

        if self.reg_loss == "DisL1":
            reg_loss_ori = F.l1_loss(
                predict_boxes3d["ori"] * reg_mask,
                targets_regression * reg_mask,
                reduction="sum") / (self.loss_weight[1] * self.max_objs)

            reg_loss_dim = F.l1_loss(
                predict_boxes3d["dim"] * reg_mask,
                targets_regression * reg_mask,
                reduction="sum") / (self.loss_weight[1] * self.max_objs)

            reg_loss_loc = F.l1_loss(
                predict_boxes3d["loc"] * reg_mask,
                targets_regression * reg_mask,
                reduction="sum") / (self.loss_weight[1] * self.max_objs)
            loss_all = hm_loss + reg_loss_ori + reg_loss_dim + reg_loss_loc
            loss_dict = {}
            loss_dict["hm"] = hm_loss.item()
            loss_dict["ori"] = reg_loss_ori.item()
            loss_dict["dim"] = reg_loss_dim.item()
            loss_dict["loc"] = reg_loss_loc.item()
            return loss_all, loss_dict, {}

class PostProcessor(nn.Module):
    def __init__(self,
                 smoker_coder,
                 reg_head,
                 det_threshold,
                 max_detection,
                 pred_2d):
        super(PostProcessor, self).__init__()
        self.smoke_coder = smoker_coder
        self.reg_head = reg_head
        self.det_threshold = det_threshold
        self.max_detection = max_detection
        self.pred_2d = pred_2d

    def prepare_targets(self, targets):
        dict_ret = {}
        dict_ret["trans_mat"] = targets["trans_mat"].view(1,3,3)
        dict_ret["K"] = targets["K"].view(1,3,3)
        dict_ret["size"] = targets["size"].view(1,2)
        return None, None, dict_ret

    def forward(self, predictions, targets):
        pred_heatmap, pred_regression = predictions[0], predictions[1]
        batch, channel = pred_regression.shape[0], pred_regression.shape[1]
        _, _, target_varibales = self.prepare_targets(targets)
        heatmap = nms_hm(pred_heatmap)

        scores, indexs, clses, ys, xs = select_topk(
            heatmap,
            K=self.max_detection,
        )

        pred_regression_pois = select_point_of_interest(
            batch, indexs, pred_regression
        )

        pred_proj_points = torch.cat([xs.view(batch, -1, 1), ys.view(batch, -1, 1)], dim=2)
        # FIXME: fix hard code here
        # pred_regression_pois = pred_regression_pois.view(-1, channel)
        pred_depths_offset = pred_regression_pois[..., 0:1]
        pred_proj_offsets = pred_regression_pois[..., 1:3]
        pred_dimensions_offsets = pred_regression_pois[..., 3:6]
        pred_orientation = pred_regression_pois[..., 6:]
        pred_depths = self.smoke_coder.decode_depth(pred_depths_offset)
        pred_locations = self.smoke_coder.decode_location(
            pred_proj_points,
            pred_proj_offsets,
            pred_depths,
            target_varibales["K"],
            target_varibales["trans_mat"]
        )
        pred_dimensions = self.smoke_coder.decode_dimension(
            clses,
            pred_dimensions_offsets
        )
        # we need to change center location to bottom location
        pred_locations[:, 1] += pred_dimensions[:, 1] / 2

        pred_rotys, pred_alphas = self.smoke_coder.decode_orientation(
            pred_orientation,
            pred_locations
        )

        if self.pred_2d:
            box2d = self.smoke_coder.encode_box2d(
                target_varibales["K"],
                pred_rotys,
                pred_dimensions,
                pred_locations,
                target_varibales["size"]
            )
        else:
            box2d = torch.tensor([0, 0, 0, 0])

        # change variables to the same dimension
        clses = clses.view(batch, self.max_detection, 1)
        pred_alphas = pred_alphas.view(batch, self.max_detection, 1)
        box2d = box2d.view(batch, self.max_detection, 4)
        pred_rotys = pred_rotys.view(batch, self.max_detection, 1)
        scores = scores.view(batch, self.max_detection, 1)
        # change dimension back to h,w,l
        pred_dimensions = pred_dimensions.roll(shifts=-1, dims=1)
        pred_dimensions = pred_dimensions.view(batch, self.max_detection, -1)
        pred_locations = pred_locations.view(batch, self.max_detection, -1)

        recall_dict = {}
        pred_dicts = []
        for index in range(batch):
            pred_boxes = [pred_locations[index], pred_dimensions[index], pred_rotys[index]]
            pred_boxes = torch.cat(pred_boxes, dim=1)
            record_dict = {
                'pred_boxes': pred_boxes,
                'pred_scores': scores[index],
                'pred_labels': clses[index]
            }
            pred_dicts.append(record_dict)
        # result = [clses, pred_alphas, box2d, pred_dimensions, pred_locations, pred_rotys, scores]
        # result = [i.type_as(result[0]) for i in result]
        # result = torch.cat(result, dim=1)
        return pred_dicts, recall_dict

class SMOKEHead(nn.Module):
    def __init__(self, num_class, input_channels, model_cfg, predict_boxes_when_training=False, **kwargs):
        super(SMOKEHead, self).__init__()
        heads = {}
        heads["hm"] = num_class
        heads["wh"] = model_cfg.HEAD_INFO.WH
        heads["reg"] = model_cfg.HEAD_INFO.REG
        heads["reg_c"] = model_cfg.HEAD_INFO.REG_C
        heads["class"] = model_cfg.HEAD_INFO.CLASS_NAMES
        heads["norm"] = model_cfg.HEAD_INFO.NORM_FUNC
        heads["dep"] = model_cfg.HEAD_INFO.DEP
        heads["rot"] = model_cfg.HEAD_INFO.ROT
        heads["dim"] = model_cfg.HEAD_INFO.DIM
        self.predictor = SMOKEPredictor(heads, input_channels)

    def forward(self, features):
        x = self.predictor(features)
        return x