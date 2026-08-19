"""
d2lcore/ObjectDetection - Object detection operations.

d2lcore functions:
  - box_corner_to_center(boxes)
  - box_center_to_corner(boxes)
  - box_iou(boxes1, boxes2)
  - nms(boxes, scores, iou_threshold)
  - multibox_prior(data, sizes, ratios)
  - multibox_detection(cls_probs, offset_preds, anchors, nms_threshold, pos_threshold)
  - offset_boxes(anchors, assigned_bb, eps)
  - offset_inverse(anchors, offset_preds)
  - assign_anchor_to_bbox(ground_truth, anchors, device_str, iou_threshold)
  - multibox_target(anchors, labels)
"""

import torch


def _ensure_2d(boxes):
    """Flatten 3D [B, N, 4] tensor to 2D [B*N, 4] for consistent downstream processing."""
    if boxes.ndim == 3:
        return boxes.reshape(-1, boxes.shape[-1])
    return boxes


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


class CdlBoxCornerToCenter:
    """Convert bounding boxes from (x1,y1,x2,y2) to (cx,cy,w,h).

    d2lcore: box_corner_to_center(boxes)
    Input: [N, 4] tensor (x1,y1,x2,y2)
    Output: [N, 4] tensor (cx,cy,w,h)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes": ("cdlTensor",),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("boxes_ccwh",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/ObjectDetection"

    def execute(self, boxes):
        boxes = _ensure_2d(boxes)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        result = torch.stack((cx, cy, w, h), dim=-1)
        return (result,)


NODE_CLASS_MAPPINGS["CdlBoxCornerToCenter"] = CdlBoxCornerToCenter
NODE_DISPLAY_NAME_MAPPINGS["CdlBoxCornerToCenter"] = "Box Corner→Center"


class CdlBoxCenterToCorner:
    """Convert bounding boxes from (cx,cy,w,h) to (x1,y1,x2,y2).

    d2lcore: box_center_to_corner(boxes)
    Input: [N, 4] tensor (cx,cy,w,h)
    Output: [N, 4] tensor (x1,y1,x2,y2)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes": ("cdlTensor",),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("boxes_xyxy",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/ObjectDetection"

    def execute(self, boxes):
        boxes = _ensure_2d(boxes)
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        result = torch.stack((x1, y1, x2, y2), dim=-1)
        return (result,)


NODE_CLASS_MAPPINGS["CdlBoxCenterToCorner"] = CdlBoxCenterToCorner
NODE_DISPLAY_NAME_MAPPINGS["CdlBoxCenterToCorner"] = "Box Center→Corner"


class CdlBoxIou:
    """Compute pairwise IoU between two sets of bounding boxes.

    d2lcore: box_iou(boxes1, boxes2)
    Input: boxes1 [N1,4], boxes2 [N2,4] (x1,y1,x2,y2)
    Output: [N1, N2] IoU matrix
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes1": ("cdlTensor",),
                "boxes2": ("cdlTensor",),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("iou",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/ObjectDetection"

    def execute(self, boxes1, boxes2):
        boxes1 = _ensure_2d(boxes1)
        boxes2 = _ensure_2d(boxes2)
        box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                                   (boxes[:, 3] - boxes[:, 1]))
        areas1 = box_area(boxes1)
        areas2 = box_area(boxes2)
        inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
        inter_areas = inters[:, :, 0] * inters[:, :, 1]
        union_areas = areas1[:, None] + areas2 - inter_areas
        return (inter_areas / union_areas,)


NODE_CLASS_MAPPINGS["CdlBoxIou"] = CdlBoxIou
NODE_DISPLAY_NAME_MAPPINGS["CdlBoxIou"] = "Box IoU"


class CdlNms:
    """Non-maximum suppression.

    d2lcore: nms(boxes, scores, iou_threshold)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes": ("cdlTensor",),
                "scores": ("cdlTensor",),
                "iou_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("keep_indices",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/ObjectDetection"

    def execute(self, boxes, scores, iou_threshold):
        B = torch.argsort(scores, dim=-1, descending=True)
        keep = []
        while B.numel() > 0:
            i = B[0]
            keep.append(i)
            if B.numel() == 1:
                break
            # Compute IoU
            iou = self._box_iou_single(boxes[i, :].reshape(-1, 4),
                                        boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
            inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
            idx = inds + 1
            idx = idx[idx < B.numel()]  # 防止最后一个元素被选中时越界
            B = B[idx]
        if keep:
            result = torch.tensor(keep, device=boxes.device)
        else:
            result = torch.tensor([], dtype=torch.long, device=boxes.device)
        return (result,)

    @staticmethod
    def _box_iou_single(boxes1, boxes2):
        box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                                   (boxes[:, 3] - boxes[:, 1]))
        areas1 = box_area(boxes1)
        areas2 = box_area(boxes2)
        inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
        inter_areas = inters[:, :, 0] * inters[:, :, 1]
        union_areas = areas1[:, None] + areas2 - inter_areas
        return inter_areas / union_areas


NODE_CLASS_MAPPINGS["CdlNms"] = CdlNms
NODE_DISPLAY_NAME_MAPPINGS["CdlNms"] = "NMS"


class CdlMultiboxPrior:
    """Generate anchor boxes with different shapes centered on each pixel.

    d2lcore: multibox_prior(data, sizes, ratios)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sizes": ("STRING", {"default": "0.75,0.5,0.25", "placeholder": "comma-separated"}),
                "ratios": ("STRING", {"default": "1,2,0.5", "placeholder": "comma-separated"}),
            },
            "optional": {
                "data": ("cdlTensor",),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("anchors",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/ObjectDetection"

    def execute(self, sizes, ratios, data=None):
        sizes_list = [float(s.strip()) for s in sizes.split(",")]
        ratios_list = [float(r.strip()) for r in ratios.split(",")]

        if data is None:
            # Default input shape: [1, C, 561, 728]
            in_height, in_width = 561, 728
            device = torch.device('cpu')
        else:
            in_height, in_width = data.shape[-2:]
            device = data.device

        num_sizes = len(sizes_list)
        num_ratios = len(ratios_list)
        boxes_per_pixel = num_sizes + num_ratios - 1
        size_tensor = torch.tensor(sizes_list, device=device)
        ratio_tensor = torch.tensor(ratios_list, device=device)

        offset_h, offset_w = 0.5, 0.5
        steps_h = 1.0 / in_height
        steps_w = 1.0 / in_width

        center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
        center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
        shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
        shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

        w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                        sizes_list[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width
        h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                        sizes_list[0] / torch.sqrt(ratio_tensor[1:])))

        anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2
        out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
        output = out_grid + anchor_manipulations
        return (output.unsqueeze(0),)


NODE_CLASS_MAPPINGS["CdlMultiboxPrior"] = CdlMultiboxPrior
NODE_DISPLAY_NAME_MAPPINGS["CdlMultiboxPrior"] = "Multibox Prior"


class CdlOffsetBoxes:
    """Transform anchor box coordinates to offsets.

    d2lcore: offset_boxes(anchors, assigned_bb, eps)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anchors": ("cdlTensor",),
                "assigned_bb": ("cdlTensor",),
                "eps": ("FLOAT", {"default": 1e-6, "min": 1e-12, "max": 1e-3, "step": 1e-6}),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("offsets",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/ObjectDetection"

    def execute(self, anchors, assigned_bb, eps):
        anchors = _ensure_2d(anchors)
        assigned_bb = _ensure_2d(assigned_bb)
        # Corner to center for anchors
        x1a, y1a, x2a, y2a = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        c_anc = torch.stack(((x1a + x2a) / 2, (y1a + y2a) / 2, x2a - x1a, y2a - y1a), dim=-1)

        # Corner to center for assigned
        x1b, y1b, x2b, y2b = assigned_bb[:, 0], assigned_bb[:, 1], assigned_bb[:, 2], assigned_bb[:, 3]
        c_assigned = torch.stack(((x1b + x2b) / 2, (y1b + y2b) / 2, x2b - x1b, y2b - y1b), dim=-1)

        offset_xy = 10 * (c_assigned[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
        offset_wh = 5 * torch.log(eps + c_assigned[:, 2:] / c_anc[:, 2:])
        result = torch.cat([offset_xy, offset_wh], dim=1)
        return (result,)


NODE_CLASS_MAPPINGS["CdlOffsetBoxes"] = CdlOffsetBoxes
NODE_DISPLAY_NAME_MAPPINGS["CdlOffsetBoxes"] = "Offset Boxes"


class CdlOffsetInverse:
    """Inverse offset transform: predict bounding boxes from anchors + offsets.

    d2lcore: offset_inverse(anchors, offset_preds)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anchors": ("cdlTensor",),
                "offset_preds": ("cdlTensor",),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("predicted_bbox",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/ObjectDetection"

    def execute(self, anchors, offset_preds):
        anchors = _ensure_2d(anchors)
        # Corner to center
        x1, y1, x2, y2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        anc = torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)

        pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
        pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
        pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), dim=1)

        # Center to corner
        cx, cy, w, h = pred_bbox[:, 0], pred_bbox[:, 1], pred_bbox[:, 2], pred_bbox[:, 3]
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        result = torch.stack((x1, y1, x2, y2), dim=-1)
        return (result,)


NODE_CLASS_MAPPINGS["CdlOffsetInverse"] = CdlOffsetInverse
NODE_DISPLAY_NAME_MAPPINGS["CdlOffsetInverse"] = "Offset Inverse"


class CdlAssignAnchorToBbox:
    """Assign ground-truth boxes to anchor boxes based on IoU.

    d2lcore: assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ground_truth": ("cdlTensor",),
                "anchors": ("cdlTensor",),
                "iou_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("anchors_bbox_map",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/ObjectDetection"

    def execute(self, ground_truth, anchors, iou_threshold):
        ground_truth = _ensure_2d(ground_truth)
        anchors = _ensure_2d(anchors)
        device = anchors.device
        num_anchors = anchors.shape[0]
        num_gt_boxes = ground_truth.shape[0]

        # Compute IoU
        jaccard = self._box_iou_matrix(anchors, ground_truth)
        anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)

        max_ious, indices = torch.max(jaccard, dim=1)
        anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
        box_j = indices[max_ious >= iou_threshold]
        anchors_bbox_map[anc_i] = box_j

        col_discard = torch.full((num_anchors,), -1)
        row_discard = torch.full((num_gt_boxes,), -1)
        for _ in range(num_gt_boxes):
            max_idx = torch.argmax(jaccard)
            box_idx = (max_idx % num_gt_boxes).long()
            anc_idx = (max_idx / num_gt_boxes).long()
            anchors_bbox_map[anc_idx] = box_idx
            jaccard[:, box_idx] = col_discard
            jaccard[anc_idx, :] = row_discard
        return (anchors_bbox_map,)

    @staticmethod
    def _box_iou_matrix(boxes1, boxes2):
        boxes1 = _ensure_2d(boxes1)
        boxes2 = _ensure_2d(boxes2)
        box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                                   (boxes[:, 3] - boxes[:, 1]))
        areas1 = box_area(boxes1)
        areas2 = box_area(boxes2)
        inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
        inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
        inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
        inter_areas = inters[:, :, 0] * inters[:, :, 1]
        union_areas = areas1[:, None] + areas2 - inter_areas
        return inter_areas / union_areas


NODE_CLASS_MAPPINGS["CdlAssignAnchorToBbox"] = CdlAssignAnchorToBbox
NODE_DISPLAY_NAME_MAPPINGS["CdlAssignAnchorToBbox"] = "Assign Anchor→BBox"


class CdlMultiboxTarget:
    """Label anchor boxes using ground-truth bounding boxes.

    d2lcore: multibox_target(anchors, labels)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anchors": ("cdlTensor",),
                "labels": ("cdlTensor",),
            }
        }

    RETURN_TYPES = ("cdlTensor", "cdlTensor", "cdlTensor")
    RETURN_NAMES = ("bbox_offset", "bbox_mask", "class_labels")
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/ObjectDetection"

    def execute(self, anchors, labels):
        batch_size = labels.shape[0]
        anchors_sq = anchors.squeeze(0)
        device = anchors.device
        num_anchors = anchors_sq.shape[0]

        batch_offset, batch_mask, batch_class_labels = [], [], []
        for i in range(batch_size):
            label = labels[i, :, :]
            anchors_bbox_map = self._assign_anchor_to_bbox(label[:, 1:], anchors_sq, device)
            bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
            class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
            assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)

            indices_true = torch.nonzero(anchors_bbox_map >= 0)
            bb_idx = anchors_bbox_map[indices_true]
            class_labels[indices_true] = label[bb_idx, 0].long() + 1
            assigned_bb[indices_true] = label[bb_idx, 1:]
            offset = self._offset_boxes(anchors_sq, assigned_bb) * bbox_mask

            batch_offset.append(offset.reshape(-1))
            batch_mask.append(bbox_mask.reshape(-1))
            batch_class_labels.append(class_labels)

        bbox_offset = torch.stack(batch_offset)
        bbox_mask = torch.stack(batch_mask)
        class_labels = torch.stack(batch_class_labels)
        return (bbox_offset, bbox_mask, class_labels)

    @staticmethod
    def _assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
        num_anchors = anchors.shape[0]
        num_gt_boxes = ground_truth.shape[0]
        box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                                   (boxes[:, 3] - boxes[:, 1]))
        areas1 = box_area(anchors)
        areas2 = box_area(ground_truth)
        inter_ul = torch.max(anchors[:, None, :2], ground_truth[:, :2])
        inter_lr = torch.min(anchors[:, None, 2:], ground_truth[:, 2:])
        inters = (inter_lr - inter_ul).clamp(min=0)
        inter_areas = inters[:, :, 0] * inters[:, :, 1]
        union_areas = areas1[:, None] + areas2 - inter_areas
        jaccard = inter_areas / union_areas

        anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
        max_ious, indices = torch.max(jaccard, dim=1)
        anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
        box_j = indices[max_ious >= iou_threshold]
        anchors_bbox_map[anc_i] = box_j
        col_discard = torch.full((num_anchors,), -1)
        row_discard = torch.full((num_gt_boxes,), -1)
        for _ in range(num_gt_boxes):
            max_idx = torch.argmax(jaccard)
            box_idx = (max_idx % num_gt_boxes).long()
            anc_idx = (max_idx / num_gt_boxes).long()
            anchors_bbox_map[anc_idx] = box_idx
            jaccard[:, box_idx] = col_discard
            jaccard[anc_idx, :] = row_discard
        return anchors_bbox_map

    @staticmethod
    def _offset_boxes(anchors, assigned_bb, eps=1e-6):
        x1a, y1a, x2a, y2a = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        c_anc = torch.stack(((x1a + x2a) / 2, (y1a + y2a) / 2, x2a - x1a, y2a - y1a), dim=-1)
        x1b, y1b, x2b, y2b = assigned_bb[:, 0], assigned_bb[:, 1], assigned_bb[:, 2], assigned_bb[:, 3]
        c_assigned = torch.stack(((x1b + x2b) / 2, (y1b + y2b) / 2, x2b - x1b, y2b - y1b), dim=-1)
        offset_xy = 10 * (c_assigned[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
        offset_wh = 5 * torch.log(eps + c_assigned[:, 2:] / c_anc[:, 2:])
        return torch.cat([offset_xy, offset_wh], dim=1)


NODE_CLASS_MAPPINGS["CdlMultiboxTarget"] = CdlMultiboxTarget
NODE_DISPLAY_NAME_MAPPINGS["CdlMultiboxTarget"] = "Multibox Target"


class CdlMultiboxDetection:
    """Predict bounding boxes using non-maximum suppression.

    d2lcore: multibox_detection(cls_probs, offset_preds, anchors, nms_threshold, pos_threshold)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cls_probs": ("cdlTensor",),
                "offset_preds": ("cdlTensor",),
                "anchors": ("cdlTensor",),
                "nms_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pos_threshold": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("cdlTensor",)
    RETURN_NAMES = ("detections",)
    FUNCTION = "execute"
    CATEGORY = "ComfyDL/ObjectDetection"

    def execute(self, cls_probs, offset_preds, anchors, nms_threshold, pos_threshold):
        device = cls_probs.device
        batch_size = cls_probs.shape[0]
        anchors_sq = anchors.squeeze(0)
        num_classes = cls_probs.shape[1]
        num_anchors = cls_probs.shape[2]
        out = []

        for i in range(batch_size):
            cls_prob = cls_probs[i]
            offset_pred = offset_preds[i].reshape(-1, 4)
            conf, class_id = torch.max(cls_prob[1:], 0)
            predicted_bb = self._offset_inverse(anchors_sq, offset_pred)
            keep = self._nms(predicted_bb, conf, nms_threshold)

            all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
            combined = torch.cat((keep, all_idx))
            uniques, counts = combined.unique(return_counts=True)
            non_keep = uniques[counts == 1]
            all_id_sorted = torch.cat((keep, non_keep))
            class_id[non_keep] = -1
            class_id = class_id[all_id_sorted]
            conf = conf[all_id_sorted]
            predicted_bb = predicted_bb[all_id_sorted]

            below_min_idx = (conf < pos_threshold)
            class_id[below_min_idx] = -1
            conf[below_min_idx] = 1 - conf[below_min_idx]

            pred_info = torch.cat((
                class_id.unsqueeze(1),
                conf.unsqueeze(1),
                predicted_bb,
            ), dim=1)
            out.append(pred_info)
        return (torch.stack(out),)

    @staticmethod
    def _nms(boxes, scores, iou_threshold):
        B = torch.argsort(scores, dim=-1, descending=True)
        keep = []
        while B.numel() > 0:
            i = B[0]
            keep.append(i)
            if B.numel() == 1:
                break
            bx1 = boxes[i, :].reshape(-1, 4)
            bx2 = boxes[B[1:], :].reshape(-1, 4)
            box_area = lambda bx: ((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]))
            areas1 = box_area(bx1)
            areas2 = box_area(bx2)
            inter_ul = torch.max(bx1[:, :2], bx2[:, :2])
            inter_lr = torch.min(bx1[:, 2:], bx2[:, 2:])
            inters = (inter_lr - inter_ul).clamp(min=0)
            inter_areas = inters[:, 0] * inters[:, 1]
            union_areas = areas1 + areas2 - inter_areas
            iou = inter_areas / union_areas
            inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
            idx = inds + 1
            idx = idx[idx < B.numel()]  # 防止最后一个元素被选中时越界
            B = B[idx]
        return torch.tensor(keep, device=boxes.device) if keep else torch.tensor([], dtype=torch.long, device=boxes.device)

    @staticmethod
    def _offset_inverse(anchors, offset_preds):
        x1, y1, x2, y2 = anchors[:, 0], anchors[:, 1], anchors[:, 2], anchors[:, 3]
        anc = torch.stack(((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1), dim=-1)
        pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
        pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
        pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), dim=1)
        cx, cy, w, h = pred_bbox[:, 0], pred_bbox[:, 1], pred_bbox[:, 2], pred_bbox[:, 3]
        return torch.stack((cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h), dim=-1)


NODE_CLASS_MAPPINGS["CdlMultiboxDetection"] = CdlMultiboxDetection
NODE_DISPLAY_NAME_MAPPINGS["CdlMultiboxDetection"] = "Multibox Detection"
