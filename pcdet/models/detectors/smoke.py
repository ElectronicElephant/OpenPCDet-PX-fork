from .detector3d_template import Detector3DTemplate
import kornia
from ..dense_heads.smoke_head import SMOKECoder,SMOKELossComputation,PostProcessor
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=2, beta=4):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction, target):
        positive_index = target.eq(1).float()
        negative_index = target.lt(1).float()

        negative_weights = torch.pow(1 - target, self.beta)
        loss = 0.

        positive_loss = torch.log(prediction) \
                        * torch.pow(1 - prediction, self.alpha) * positive_index
        negative_loss = torch.log(1 - prediction) \
                        * torch.pow(prediction, self.alpha) * negative_weights * negative_index

        num_positive = positive_index.float().sum()
        positive_loss = positive_loss.sum()
        negative_loss = negative_loss.sum()

        if num_positive == 0:
            loss -= negative_loss
        else:
            loss -= (positive_loss + negative_loss) / num_positive

        return loss

def make_smoke_loss_evaluator(model_cfg, smoke_coder):
    focal_loss = FocalLoss(alpha=2, beta=4)
    loss_evaluator = SMOKELossComputation(
        smoke_coder,
        cls_loss=focal_loss,
        reg_loss="DisL1",
        loss_weight=(1., 10.),
        max_objs=50,
    )
    return loss_evaluator

def make_smoke_post_processor(model_cfg, smoke_coder):
    postprocessor = PostProcessor(
        smoke_coder,
        8,
        0.25,
        50,
        True,
    )
    return postprocessor

class Smoke(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.smoke_coder = SMOKECoder(model_cfg.DENSE_HEAD.DEPTH_REFERENCE,
                                      model_cfg.DENSE_HEAD.DIMENSION_REFERENCE,
                                      "cuda")
        self.loss_evaluator = make_smoke_loss_evaluator(model_cfg, self.smoke_coder)
        self.post_processor = make_smoke_post_processor(model_cfg, self.smoke_coder)

    def forward(self, batch_dict):
        target = batch_dict["target"]
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.loss_evaluator(batch_dict, target)
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processor(batch_dict, target)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict_rpn = self.dense_head.get_loss()
        loss_depth, tb_dict_depth = self.vfe.get_loss()

        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            'loss_depth': loss_depth.item(),
            **tb_dict_rpn,
            **tb_dict_depth
        }

        loss = loss_rpn + loss_depth
        return loss, tb_dict, disp_dict
