from mmdet.datasets.pipelines.transforms import MinIoURandomCrop, Compose
from mmdet.datasets.pipelines.formating import Collect
from mmdet.datasets import PIPELINES
import numpy as np
import random
import copy
import torch
import mmcv
from mmcv import build_from_cfg
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines.transforms import Resize
import torch.nn.functional as F


@PIPELINES.register_module()
class BernoulliChoose(object):
    def __init__(self, translate, p):
        self.translate = build_from_cfg(translate, PIPELINES)
        self.p = p

    def __call__(self, result):
        if random.random() <= self.p:
            result = self.translate(result)
        return result


@PIPELINES.register_module()
class ConcatWithAugmentation(object):
    def __init__(self, ori_translate, aug_translate):
        # TODO: pip install albumentations
        self.ori_translate = [build_from_cfg(i, PIPELINES) for i in ori_translate]
        self.aug_translate = [build_from_cfg(i, PIPELINES) for i in aug_translate]

    def __call__(self, result):
        aug_result = copy.deepcopy(result)
        for ori_aug in self.ori_translate:
            result = ori_aug(result)
        for aug_aug in self.aug_translate:
            aug_result = aug_aug(aug_result)
        return {"ori_result": result, "aug_result": aug_result}


@PIPELINES.register_module()
class InvertTransform(object):
    def __init__(self, prob=0.5):
        assert 0 <= prob <= 1.0, \
            'The probability should be in range [0,1].'
        self.prob = prob

    def _invert(self, results):
        """Invert one image."""
        for key in results.get('img_fields', ['img']):
            img = results[key]
            results[key] = mmcv.iminvert(img).astype(img.dtype)

    def __call__(self, results):
        """Call function for Invert transformation.

        Args:
            results (dict): Results dict from loading pipeline.

        Returns:
            dict: Results after the transformation.
        """
        if np.random.rand() > self.prob:
            return results
        self._invert(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob})'

@PIPELINES.register_module()
class AugCollect(Collect):
    def __init__(self,
                 defaultformatbundle,
                 keys,
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                            'img_shape', 'pad_shape', 'scale_factor', 'flip',
                            'flip_direction', 'img_norm_cfg')
                 ):
        super(AugCollect, self).__init__(keys, meta_keys)
        self.defaultformatbundle = build_from_cfg(defaultformatbundle,PIPELINES)
        self.resize = Resize()
    def __call__(self, result):
        ori_result, aug_result = result["ori_result"], result["aug_result"]
        ori_result = self.defaultformatbundle(ori_result)
        aug_result = self.defaultformatbundle(aug_result)
        ori_data = super(AugCollect, self).__call__(ori_result)
        aug_data = super(AugCollect, self).__call__(aug_result)
        for key in self.keys:
            ori_meta_data = ori_data[key]
            aug_meta_data = aug_data[key]
            assert isinstance(ori_meta_data, DC) and isinstance(aug_meta_data, DC), \
                f"when key is {key}, it's value must be Tensor, but be {type(ori_meta_data)} and {type(aug_meta_data)}"
            if key == "gt_bboxes" or key == "gt_labels":
                aug_meta_data._data = [ori_meta_data.data, aug_meta_data.data]
            else:
                aug_meta_data._data = torch.stack([ori_meta_data.data, aug_meta_data.data])
            aug_data[key] = aug_meta_data
        aug_data["img_metas"] = [ori_data["img_metas"],aug_data["img_metas"]]
        return aug_data
