# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: landmark_model.py

from .base_model import BaseModel
import cv2
import numpy as np
from src.utils.crop import crop_image, _transform_pts


class LandmarkModel(BaseModel):
    """
    landmark Model
    """

    def __init__(self, **kwargs):
        super(LandmarkModel, self).__init__(**kwargs)
        self.predict_type = kwargs.get("predict_type", "trt")
        print(self.predict_type)
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

    def predict(self, *data):
        input, crop_dct = self.input_process(*data)
        preds = self.predictor.predict(input)
        outputs = self.output_process(preds, crop_dct)
        return outputs
