import numpy as np
import pycocotools.mask as cocomask
from mmcv.transforms import BaseTransform
from mmdet.structures.bbox import get_box_type
from mmdet.structures.mask import PolygonMasks, BitmapMasks
from mmrotate.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class ConvertRLEMask2BoxType(BaseTransform):
    """Convert masks in results to a certain box type.
    Base src: mmrotate\datasets\transforms\transforms.py.
    Original code extended to allow processing of RLE encoding."

    Required Keys:

    - ori_shape
    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_masks (BitmapMasks | PolygonMasks)
    - instances (List[dict]) (optional)
    Modified Keys:
    - gt_bboxes
    - gt_masks
    - instances

    Args:
        box_type (str): The destination box type.
        keep_mask (bool): Whether to keep the ``gt_masks``.
            Defaults to False.
    """

    def __init__(self, box_type: str, keep_mask: bool = False) -> None:
        _, self.box_type_cls = get_box_type(box_type)
        assert hasattr(self.box_type_cls, "from_instance_masks")
        self.keep_mask = keep_mask

    def transform(self, results: dict) -> dict:
        """The transform function."""
        assert "gt_masks" in results.keys()
        masks = results["gt_masks"]
        results["gt_bboxes"] = self.box_type_cls.from_instance_masks(masks)
        if not self.keep_mask:
            results.pop("gt_masks")
        # modify results['instances'] for RotatedCocoMetric
        # only relevant for evaluation (not training)
        # added differentiation for rle vs. poly encoding
        converted_instances = []
        for instance in results["instances"]:
            if isinstance(instance["mask"], list):
                m = np.array(instance["mask"][0])
                m = PolygonMasks(
                    [[m]], results["ori_shape"][1], results["ori_shape"][0]
                )
            else:
                m = cocomask.decode(instance["mask"])
                m = BitmapMasks(
                    np.expand_dims(m, axis=0),
                    results["ori_shape"][1],
                    results["ori_shape"][0],
                )
            instance["bbox"] = (
                self.box_type_cls.from_instance_masks(m).tensor[0].numpy().tolist()
            )
            if not self.keep_mask:
                instance.pop("mask")
            converted_instances.append(instance)
        results["instances"] = converted_instances
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(box_type_cls={self.box_type_cls}, "
        repr_str += f"keep_mask={self.keep_mask})"
        return repr_str
