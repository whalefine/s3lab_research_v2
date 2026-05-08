from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, generalized_box_iou
import torch
import torch.nn.functional as F
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


def _unwrap_module(net):
    return net.module if hasattr(net, 'module') else net


def _backbone_feat_tensor(feat):
    """Distill 比對使用單一 tensor（若為 list 則取最後一層）。"""
    if isinstance(feat, (list, tuple)):
        return feat[-1]
    return feat


class sglatrackDistillActor(BaseActor):
    """sglatrack training actor with optional feature distillation (ViT or CNN teacher)."""

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg
        self.net_teacher = None
        self._last_num_search = 1

    def __call__(self, data):
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        try:
            aqa_enable = bool(self.cfg.MODEL.AQA_QUERY.ENABLE)
        except (AttributeError, KeyError):
            aqa_enable = False
        num_search_cfg = int(getattr(self.settings, 'num_search', 1))
        use_multi_search = aqa_enable and num_search_cfg > 1 and len(data['search_images']) > 1

        template_list = []
        num_template = min(int(self.settings.num_template), int(len(data['template_images'])))
        for i in range(num_template):
            template_img_i = data['template_images'][i].view(
                -1, *data['template_images'].shape[2:])
            template_list.append(template_img_i)

        if use_multi_search:
            ns = min(num_search_cfg, int(len(data['search_images'])))
            search_img = [
                data['search_images'][i].view(-1, *data['search_images'].shape[2:])
                for i in range(ns)
            ]
            self._last_num_search = ns
        else:
            assert len(data['search_images']) >= 1
            search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])
            self._last_num_search = 1

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])
            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                            total_epochs=ce_start_epoch + ce_warm_epoch,
                                            ITERS_PER_EPOCH=1,
                                            base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        student = _unwrap_module(self.net)
        cnn_adapter = getattr(student, 'distill_cnn_adapter', None)
        if getattr(student, 'is_distill_training', False) and self.net_teacher is not None:
            teacher_search = search_img[0] if isinstance(search_img, list) else search_img
            with torch.no_grad():
                out_teacher = self.net_teacher(
                    template=template_list,
                    search=teacher_search,
                    ce_template_mask=box_mask_z,
                    ce_keep_rate=ce_keep_rate,
                    return_last_attn=False,
                    is_distill=True,
                )

        out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False,
                            is_distill=False)

        if getattr(student, 'is_distill_training', False) and cnn_adapter is not None:
            if isinstance(search_img, list):
                search_cat = torch.cat(search_img, dim=0)
                tpl = template_list
                if isinstance(tpl, list):
                    tpl = tpl[0]
                tpl = tpl.repeat(self._last_num_search, 1, 1, 1)
                feat_t = cnn_adapter(tpl, search_cat)
            else:
                feat_t = cnn_adapter(template_list, search_img)
            feat_s = out_dict['backbone_feat']
            b = feat_s.shape[0]
            distill_loss = torch.stack(
                [F.mse_loss(feat_t[i], feat_s[i]) for i in range(b)]
            )
            out_dict['distill_loss'] = distill_loss
        elif getattr(student, 'is_distill_training', False) and self.net_teacher is not None:
            feat_t = _backbone_feat_tensor(out_teacher['backbone_feat'])
            feat_s = _backbone_feat_tensor(out_dict['backbone_feat'])
            align = getattr(student, 'distill_teacher_feat_align', None)
            if align is not None:
                feat_t = align(feat_t)
            b = feat_s.shape[0]
            distill_loss = torch.stack(
                [F.mse_loss(feat_t[i], feat_s[i]) for i in range(b)]
            )
            out_dict['distill_loss'] = distill_loss

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        if self._last_num_search > 1:
            search_anno = gt_dict['search_anno'][:self._last_num_search]
            gt_bbox = search_anno.contiguous().view(-1, 4)
            gt_gaussian_maps = generate_heatmap(search_anno, self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
            gt_gaussian_maps = torch.stack(gt_gaussian_maps, dim=0).contiguous().view(
                -1, gt_gaussian_maps[0].shape[-2], gt_gaussian_maps[0].shape[-1]
            ).unsqueeze(1)
        else:
            gt_bbox = gt_dict['search_anno'][-1]
            gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
            gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(
            min=0.0, max=1.0)
        dev = pred_boxes_vec.device
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
            giou_vec, _ = generalized_box_iou(pred_boxes_vec, gt_boxes_vec)
        except Exception:
            giou_loss = torch.tensor(0.0, device=dev)
            iou = torch.tensor(0.0, device=dev)
            giou_vec = torch.zeros(pred_boxes_vec.shape[0], device=dev)
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        cos_tensor = pred_dict['cos_tensor']
        indices = torch.argmax(cos_tensor, dim=1)
        pro_target = torch.zeros_like(cos_tensor)
        pro_target.scatter_(1, indices.unsqueeze(1), 1)
        pro = pred_dict['pro']
        pro_loss = self.objective['l1'](pro, pro_target)
        sim_w = float(self.loss_weight.get('sim_loss', 0.0))
        sim_term = sim_w * pred_dict.get('sim_loss', torch.tensor(0.0, device=l1_loss.device))
        loss = (
            self.loss_weight['giou'] * giou_loss
            + self.loss_weight['l1'] * l1_loss
            + self.loss_weight['focal'] * location_loss
            + 0.2 * pro_loss
            + sim_term
        )

        student = _unwrap_module(self.net)
        if getattr(student, 'is_distill_training', False) and 'distill_loss' in pred_dict:
            d_w = float(self.loss_weight.get('distill_loss', 0.0))
            tau_0 = float(getattr(self.cfg.TRAIN, 'AFKD_TAU0', 10))
            rho = float(getattr(self.cfg.TRAIN, 'AFKD_RHO', 10))
            distill_loss = pred_dict['distill_loss']
            one_m_g = 1.0 - giou_vec
            coef = d_w * (tau_0 + rho * (one_m_g - one_m_g.mean()))
            distill_term = (coef * distill_loss).mean()
            loss = loss + distill_term

        if return_status:
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "pro_loss": pro_loss.item(),
                      "IoU": mean_iou.item()}
            if isinstance(pred_dict.get('sim_loss'), torch.Tensor):
                status["Loss/sim"] = float(pred_dict['sim_loss'].detach().item())
            if getattr(student, 'is_distill_training', False) and 'distill_loss' in pred_dict:
                status["Loss/distill"] = float(pred_dict['distill_loss'].detach().mean().item())
            return loss, status
        return loss
