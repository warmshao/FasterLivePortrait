# -*- coding: utf-8 -*-
# @Time    : 2024/12/15
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: helper.py
import os.path as osp


class NullableArgs:
    def __init__(self, namespace):
        for key, value in namespace.__dict__.items():
            setattr(self, key, value)

    def __getattr__(self, key):
        # when an attribute lookup has not found the attribute
        if key == 'align_mask_width':
            if 'use_alignment_mask' in self.__dict__:
                return 1 if self.use_alignment_mask else 0
            else:
                return 0
        if key == 'no_head_pose':
            return not self.predict_head_pose
        if key == 'no_use_learnable_pe':
            return not self.use_learnable_pe

        return None


def make_abs_path(fn):
    # return osp.join(osp.dirname(osp.realpath(__file__)), fn)
    return osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), fn))
