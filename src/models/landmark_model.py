# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: landmark_model.py
import pdb

from .base_model import BaseModel
import cv2
import numpy as np
from src.utils.crop import crop_image, _transform_pts
import torch
from torch.cuda import nvtx
from .predictor import numpy_to_torch_dtype_dict


class LandmarkModel(BaseModel):
    """
    landmark Model
    """

    def __init__(self, **kwargs):
        super(LandmarkModel, self).__init__(**kwargs)
        self.dsize = 224

    def input_process(self, *data):
        if len(data) > 1:
            img_rgb, lmk = data
        else:
            img_rgb = data[0]
            lmk = None
        if lmk is not None:
            crop_dct = crop_image(img_rgb, lmk, dsize=self.dsize, scale=1.5, vy_ratio=-0.1)
            img_crop_rgb = crop_dct['img_crop']
        else:
            # NOTE: force resize to 224x224, NOT RECOMMEND!
            img_crop_rgb = cv2.resize(img_rgb, (self.dsize, self.dsize))
            scale = max(img_rgb.shape[:2]) / self.dsize
            crop_dct = {
                'M_c2o': np.array([
                    [scale, 0., 0.],
                    [0., scale, 0.],
                    [0., 0., 1.],
                ], dtype=np.float32),
            }

        inp = (img_crop_rgb.astype(np.float32) / 255.).transpose(2, 0, 1)[None, ...]  # HxWx3 (BGR) -> 1x3xHxW (RGB!)
        return inp, crop_dct

    def output_process(self, *data):
        out_pts, crop_dct = data
        lmk = out_pts[2].reshape(-1, 2) * self.dsize  # scale to 0-224
        lmk = _transform_pts(lmk, M=crop_dct['M_c2o'])
        return lmk

    def predict_trt(self, *data):
        nvtx.range_push("forward")
        feed_dict = {}
        for i, inp in enumerate(self.predictor.inputs):
            if isinstance(data[i], torch.Tensor):
                feed_dict[inp['name']] = data[i]
            else:
                feed_dict[inp['name']] = torch.from_numpy(data[i]).to(device=self.device,
                                                                      dtype=numpy_to_torch_dtype_dict[inp['dtype']])
        preds_dict = self.predictor.predict(feed_dict, self.cudaStream)
        outs = []
        for i, out in enumerate(self.predictor.outputs):
            outs.append(preds_dict[out["name"]].cpu().numpy())
        nvtx.range_pop()
        return outs

    def predict(self, *data):
        input, crop_dct = self.input_process(*data)
        if self.predict_type == "trt":
            preds = self.predict_trt(input)
        else:
            preds = self.predictor.predict(input)
        outputs = self.output_process(preds, crop_dct)
        return outputs
