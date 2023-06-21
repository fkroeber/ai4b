from typing import List, Tuple

import numpy as np
import torch
from mmcv.cnn import ConvTranspose2d
from mmdet.models.roi_heads.mask_heads import FCNMaskHead
from mmdet.models.roi_heads.mask_heads.maskiou_head import MaskIoUHead
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances
from mmdet.structures.bbox import scale_boxes
from mmdet.utils import InstanceList
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmrotate.registry import MODELS
from mmrotate.structures.bbox import RotatedBoxes
from torch import Tensor
from .utils import mask_target, _do_paste_mask

BYTES_PER_FLOAT = 4
# TODO: This memory limit may be too much or too little. It would be better to
#  determine it based on available resources.
GPU_MEM_LIMIT = 1024 ** 3  # 1 GB memory limit


@MODELS.register_module(force=True)
class RotatedFCNMaskHead(FCNMaskHead):
    def predict_by_feat(
        self,
        mask_preds: Tuple[Tensor],
        results_list: List[InstanceData],
        batch_img_metas: List[dict],
        rcnn_test_cfg: ConfigDict,
        rescale: bool = False,
        activate_map: bool = False,
    ) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        mask results.

        Args:
            mask_preds (tuple[Tensor]): Tuple of predicted foreground masks,
                each has shape (n, num_classes, h, w).
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.
            batch_img_metas (list[dict]): List of image information.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            activate_map (book): Whether get results with augmentations test.
                If True, the `mask_preds` will not process with sigmoid.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        assert len(mask_preds) == len(results_list) == len(batch_img_metas)

        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            results = results_list[img_id]
            bboxes = results.bboxes
            if bboxes.shape[0] == 0:
                results_list[img_id] = empty_instances(
                    [img_meta],
                    bboxes.device,
                    task_type="mask",
                    instance_results=[results],
                    mask_thr_binary=rcnn_test_cfg.mask_thr_binary,
                )[0]
            else:
                # here the rescaling of bboxes takes place
                bboxes, im_mask = self._predict_by_feat_single(
                    mask_preds=mask_preds[img_id],
                    bboxes=bboxes,
                    labels=results.labels,
                    img_meta=img_meta,
                    rcnn_test_cfg=rcnn_test_cfg,
                    rescale=rescale,
                    activate_map=activate_map,
                )
                results.bboxes = bboxes
                results.masks = im_mask
        return results_list

    def _predict_by_feat_single(
        self,
        mask_preds: Tensor,
        bboxes: Tensor,
        labels: Tensor,
        img_meta: dict,
        rcnn_test_cfg: ConfigDict,
        rescale: bool = False,
        activate_map: bool = False,
    ) -> Tensor:
        """Get segmentation masks from mask_preds and bboxes.
        Args:
            mask_preds (Tensor): Predicted foreground masks, has shape
                (n, num_classes, h, w).
            bboxes (Tensor): Predicted bboxes, has shape (n, 4)
            labels (Tensor): Labels of bboxes, has shape (n, )
            img_meta (dict): image information.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            activate_map (book): Whether get results with augmentations test.
                If True, the `mask_preds` will not process with sigmoid.
                Defaults to False.
        Returns:
            Tensor: Encoded masks, has shape (n, img_w, img_h)
        Example:
            >>> from mmengine.config import Config
            >>> from mmdet.models.roi_heads.mask_heads.fcn_mask_head import *  # NOQA
            >>> N = 7  # N = number of extracted ROIs
            >>> C, H, W = 11, 32, 32
            >>> # Create example instance of FCN Mask Head.
            >>> self = FCNMaskHead(num_classes=C, num_convs=0)
            >>> inputs = torch.rand(N, self.in_channels, H, W)
            >>> mask_preds = self.forward(inputs)
            >>> # Each input is associated with some bounding box
            >>> bboxes = torch.Tensor([[1, 1, 42, 42 ]] * N)
            >>> labels = torch.randint(0, C, size=(N,))
            >>> rcnn_test_cfg = Config({'mask_thr_binary': 0, })
            >>> ori_shape = (H * 4, W * 4)
            >>> scale_factor = (1, 1)
            >>> rescale = False
            >>> img_meta = {'scale_factor': scale_factor,
            ...             'ori_shape': ori_shape}
            >>> # Encoded masks are a list for each category.
            >>> encoded_masks = self._get_seg_masks_single(
            ...     mask_preds, bboxes, labels,
            ...     img_meta, rcnn_test_cfg, rescale)
            >>> assert encoded_masks.size()[0] == N
            >>> assert encoded_masks.size()[1:] == ori_shape
        """
        bboxes = RotatedBoxes(bboxes)
        scale_factor = bboxes.new_tensor(img_meta["scale_factor"]).repeat((1, 2))
        img_h, img_w = img_meta["ori_shape"][:2]
        device = bboxes.device

        if not activate_map:
            mask_preds = mask_preds.sigmoid()
        else:
            # In AugTest, has been activated before
            mask_preds = bboxes.new_tensor(mask_preds)

        if rescale:  # in-placed rescale the bboxes
            scale_factor = [1 / s for s in img_meta["scale_factor"]]
            bboxes = scale_boxes(bboxes, scale_factor)
        else:
            w_scale, h_scale = scale_factor[0, 0], scale_factor[0, 1]
            img_h = np.round(img_h * h_scale.item()).astype(np.int32)
            img_w = np.round(img_w * w_scale.item()).astype(np.int32)

        N = len(mask_preds)
        # The actual implementation split the input into chunks,
        # and paste them chunk by chunk.
        if device.type == "cpu":
            # CPU is most efficient when they are pasted one by one with
            # skip_empty=True, so that it performs minimal number of
            # operations.
            num_chunks = N
        else:
            # GPU benefits from parallelism for larger chunks,
            # but may have memory issue
            # the types of img_w and img_h are np.int32,
            # when the image resolution is large,
            # the calculation of num_chunks will overflow.
            # so we need to change the types of img_w and img_h to int.
            # See https://github.com/open-mmlab/mmdetection/pull/5191
            num_chunks = int(
                np.ceil(N * int(img_h) * int(img_w) * BYTES_PER_FLOAT / GPU_MEM_LIMIT)
            )
            assert (
                num_chunks <= N
            ), "Default GPU_MEM_LIMIT is too small; try increasing it"
        chunks = torch.chunk(torch.arange(N, device=device), num_chunks)

        threshold = rcnn_test_cfg.mask_thr_binary
        im_mask = torch.zeros(
            N,
            img_h,
            img_w,
            device=device,
            dtype=torch.bool if threshold >= 0 else torch.uint8,
        )

        if not self.class_agnostic:
            mask_preds = mask_preds[range(N), labels][:, None]

        for inds in chunks:
            masks_chunk, spatial_inds = _do_paste_mask(
                mask_preds[inds],
                bboxes[inds],
                img_h,
                img_w,
                skip_empty=device.type == "cpu",
            )

            if threshold >= 0:
                masks_chunk = (masks_chunk >= threshold).to(dtype=torch.bool)
            else:
                # for visualization and debugging
                masks_chunk = (masks_chunk * 255).to(dtype=torch.uint8)

            im_mask[(inds,) + spatial_inds] = masks_chunk
        return bboxes, im_mask


@MODELS.register_module(force=True)
class ORCNNFCNMaskHead(RotatedFCNMaskHead):
    def get_targets(
        self,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        rcnn_train_cfg: ConfigDict,
    ) -> Tensor:
        """Calculate the ground truth for all samples in a batch according to
        the sampling_results.

        Args:
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes``, ``labels``, and
                ``masks`` attributes.
            rcnn_train_cfg (obj:ConfigDict): `train_cfg` of RCNN.
        Returns:
            Tensor: Mask target of each positive proposals in the image.
        """
        pos_proposals = [res.pos_priors for res in sampling_results]
        pos_assigned_gt_inds = [res.pos_assigned_gt_inds for res in sampling_results]
        gt_masks = [res.masks for res in batch_gt_instances]
        mask_targets = mask_target(
            pos_proposals, pos_assigned_gt_inds, gt_masks, rcnn_train_cfg
        )
        return mask_targets


@MODELS.register_module(force=True)
class MaskIoUHead_II(MaskIoUHead):
    def __init__(self, *args, **kwargs):
        """
        A modified mask iou head that performs upsampling instead of pooling
        to make the mask_predictions match the size of the mask features,
        necessary for combined pointrend & mask-scoring head as pointrend uses
        a coarses scale mask head
        """
        super().__init__(*args, **kwargs)
        self.conv2d_t = ConvTranspose2d(1, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, mask_feat: Tensor, mask_preds: Tensor) -> Tensor:
        mask_preds = mask_preds.sigmoid()
        mask_preds_upsampled = self.conv2d_t(mask_preds.unsqueeze(1))
        x = torch.cat((mask_feat, mask_preds_upsampled), 1)

        for conv in self.convs:
            x = self.relu(conv(x))
        x = x.flatten(1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        mask_iou = self.fc_mask_iou(x)
        return mask_iou


@MODELS.register_module(force=True)
class OMaskIoUHead(MaskIoUHead):
    """Oriented Mask IoU Head.
    Base src: mmdet/models/roi_heads/mask_heads/maskiou_head.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, mask_feat: Tensor, mask_preds: Tensor) -> Tensor:
        """Forward function.

        Args:
            mask_feat (Tensor): Mask features from upstream models.
            mask_preds (Tensor): Mask predictions from mask head.

        Returns:
            Tensor: Mask IoU predictions.
        """
        mask_preds = mask_preds.sigmoid()
        mask_pred_pooled = self.max_pool(mask_preds.unsqueeze(1))

        x = torch.cat((mask_feat, mask_pred_pooled), 1)

        for conv in self.convs:
            x = self.relu(conv(x))
        x = x.flatten(1)
        for fc in self.fcs:
            x = self.relu(fc(x))
        mask_iou = self.fc_mask_iou(x)
        return mask_iou

    def loss_and_target(
        self,
        mask_iou_pred: Tensor,
        mask_preds: Tensor,
        mask_targets: Tensor,
        sampling_results: List[SamplingResult],
        batch_gt_instances: InstanceList,
        rcnn_train_cfg: ConfigDict,
    ) -> dict:
        """Calculate the loss and targets of MaskIoUHead.

        Args:
            mask_iou_pred (Tensor): Mask IoU predictions results, has shape
                (num_pos, num_classes)
            mask_preds (Tensor): Mask predictions from mask head, has shape
                (num_pos, mask_size, mask_size).
            mask_targets (Tensor): The ground truth masks assigned with
                predictions, has shape
                (num_pos, mask_size, mask_size).
            sampling_results (List[obj:SamplingResult]): Assign results of
                all images in a batch after sampling.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It includes ``masks`` inside.
            rcnn_train_cfg (obj:`ConfigDict`): `train_cfg` of RCNN.

        Returns:
            dict: A dictionary of loss and targets components.
                The targets are only used for cascade rcnn.
        """
        mask_iou_targets = self.get_targets(
            mask_preds=mask_preds,
            mask_targets=mask_targets,
            rcnn_train_cfg=rcnn_train_cfg,
        )

        pos_inds = mask_iou_targets > 0
        if pos_inds.sum() > 0:
            loss_mask_iou = self.loss_iou(
                mask_iou_pred[pos_inds], mask_iou_targets[pos_inds]
            )
        else:
            loss_mask_iou = mask_iou_pred.sum() * 0
        return dict(loss_mask_iou=loss_mask_iou)

    def get_targets(
        self,
        mask_preds: Tensor,
        mask_targets: Tensor,
        rcnn_train_cfg: ConfigDict,
    ) -> Tensor:
        """Compute target of mask IoU.

        Mask IoU target is the IoU of the predicted mask and
        the gt mask of corresponding gt mask. Different to the original MaskIOU,
        the area calculation is simplified as ROIAlign has been carried out before
        for mask targets. Therefore the two-step calculation of gt mask areas
        (firstly gt area inside the bbox, then divide it by the area ratio of gt area
        inside the bbox and the gt area of the whole instance) as a consquence of the
        gt masks being clipped and resized in the usual FCNMaskHead can be omitted.
        However, this also means that the IoU calculation isn't based on same pixel sizes
        anymore but on bin sizes for ROIAlignment. This tends to weight small objects stronger
        since the ROI bin sizes correspond to smaller pixel areas here.

        Args:
            mask_preds (Tensor): Predicted masks of each positive proposal,
                shape (num_pos, h, w).
            mask_targets (Tensor): Gt mask of each positive proposal,
                binary map of the shape (num_pos, h, w).
            rcnn_train_cfg (obj:`ConfigDict`): Training config for R-CNN part.

        Returns:
            Tensor: mask iou target (length == num positive).
        """
        # get binary mask preds & targets
        mask_preds = (mask_preds > rcnn_train_cfg.mask_thr_binary).float()
        mask_targets = (mask_targets > rcnn_train_cfg.mask_thr_binary).float()

        # calculate IoU
        overlap_areas = (mask_preds * mask_targets).sum((-1, -2))
        union_areas = ((mask_preds + mask_targets) > 0).sum((-1, -2))
        mask_iou_targets = overlap_areas / union_areas

        return mask_iou_targets

    def predict_by_feat(
        self, mask_iou_preds: Tuple[Tensor], results_list: InstanceList
    ) -> InstanceList:
        """Predict the mask iou and calculate it into ``results.scores``.

        Args:
            mask_iou_preds (Tensor): Mask IoU predictions results, has shape
                (num_proposals, num_classes)
            results_list (list[:obj:`InstanceData`]): Detection results of
                each image.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each image
            after the post process. Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
        assert len(mask_iou_preds) == len(results_list)
        for results, mask_iou_pred in zip(results_list, mask_iou_preds):
            labels = results.labels
            scores = results.scores
            results.scores = scores * mask_iou_pred[range(labels.size(0)), labels]
        return results_list
