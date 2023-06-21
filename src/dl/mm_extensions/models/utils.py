import torch
import torch.nn.functional as F
from mmcv import ops
from torch import Tensor
from torch.nn.modules.utils import _pair


def _do_paste_mask(
    masks: Tensor, boxes: Tensor, img_h: int, img_w: int, skip_empty: bool = True
) -> tuple:
    """Paste instance masks according to boxes."""
    # get the centre and basis vectors of the rotated bounding boxes
    ctr, w, h, theta = torch.split(boxes.tensor, (2, 1, 1, 1), dim=-1)
    cos_value, sin_value = torch.cos(theta), torch.sin(theta)
    vec1 = torch.cat([w / 2 * cos_value, w / 2 * sin_value], dim=-1)
    vec2 = torch.cat([-h / 2 * sin_value, h / 2 * cos_value], dim=-1)

    # create a grid of points
    grid_y, grid_x = torch.meshgrid(torch.arange(img_h), torch.arange(img_w))
    grid = torch.stack((grid_x, grid_y), dim=-1).to(ctr.device)

    # subtract the center point from all points in the grid
    p_prime = grid[None, :, :, :] - ctr[:, None, None, :]

    # get factors by batched matrix inversion & multiplication
    mat = torch.stack([vec1, vec2], dim=-1)
    mat_inv = torch.inverse(mat)
    mat_inv_exp = mat_inv.unsqueeze(-3).unsqueeze(-3)
    factors = torch.matmul(mat_inv_exp, p_prime.unsqueeze(-1)).squeeze(-1)

    img_masks = F.grid_sample(
        masks.to(dtype=torch.float32), factors, align_corners=False
    )
    return img_masks[:, 0], ()


def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list, cfg):
    """Compute mask target for positive proposals in multiple images.
    Args:
        pos_proposals_list (list[Tensor]): Positive proposals in multiple
            images, each has shape (num_pos, 4).
        pos_assigned_gt_inds_list (list[Tensor]): Assigned GT indices for each
            positive proposals, each has shape (num_pos,).
        gt_masks_list (list[:obj:`BaseInstanceMasks`]): Ground truth masks of
            each image.
        cfg (dict): Config dict that specifies the mask size.
    Returns:
        Tensor: Mask target of each image, has shape (num_pos, w, h).
    """
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(
        mask_target_single,
        pos_proposals_list,
        pos_assigned_gt_inds_list,
        gt_masks_list,
        cfg_list,
    )
    mask_targets = list(mask_targets)
    if len(mask_targets) > 0:
        mask_targets = torch.cat(mask_targets)
    return mask_targets


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    """Compute mask target for each positive proposal in the image.
    Args:
        pos_proposals (Tensor): Positive proposals.
        pos_assigned_gt_inds (Tensor): Assigned GT inds of positive proposals.
        gt_masks (:obj:`BaseInstanceMasks`): GT masks in the format of Bitmap
            or Polygon.
        cfg (dict): Config dict that indicate the mask size.
    Returns:
        Tensor: Mask target of each positive proposals in the image.
    """
    device = pos_proposals.device
    mask_size = _pair(cfg.mask_size)
    num_pos = pos_proposals.size(0)
    if num_pos > 0:
        # initialise RoIAlign
        layer_cls = getattr(ops, "RoIAlignRotated")
        target_roi_extractor = layer_cls(
            spatial_scale=1, output_size=28, sampling_ratio=0, clockwise=True
        )

        # get gt masks in format N,1,H,W
        gt_masks = gt_masks.to_tensor(torch.float, device)[pos_assigned_gt_inds]
        gt_masks = torch.unsqueeze(gt_masks, 1)

        # get rpn rboxes
        batch_idxs = torch.arange(len(pos_proposals.tensor)).reshape(-1, 1).to(device)
        roi_proposals = torch.hstack((batch_idxs, pos_proposals.tensor)).to(torch.float)

        # get target masks
        mask_targets = target_roi_extractor(gt_masks, roi_proposals)[:, 0, :, :]
        mask_targets = mask_targets.float().to(device)

    else:
        mask_targets = pos_proposals.new_zeros((0,) + mask_size)

    return mask_targets
