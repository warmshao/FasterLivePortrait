# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo0611@gmail.com
# @Project : FasterLivePortrait
# @FileName: face_analysis_model.py
import pdb

import numpy as np
from insightface.app.common import Face
import cv2
from .predictor import get_predictor
from ..utils import face_align
import torch
from torch.cuda import nvtx
from .predictor import numpy_to_torch_dtype_dict


def sort_by_direction(faces, direction: str = 'large-small', face_center=None):
    if len(faces) <= 0:
        return faces

    if direction == 'left-right':
        return sorted(faces, key=lambda face: face['bbox'][0])
    if direction == 'right-left':
        return sorted(faces, key=lambda face: face['bbox'][0], reverse=True)
    if direction == 'top-bottom':
        return sorted(faces, key=lambda face: face['bbox'][1])
    if direction == 'bottom-top':
        return sorted(faces, key=lambda face: face['bbox'][1], reverse=True)
    if direction == 'small-large':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]))
    if direction == 'large-small':
        return sorted(faces, key=lambda face: (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1]),
                      reverse=True)
    if direction == 'distance-from-retarget-face':
        return sorted(faces, key=lambda face: (((face['bbox'][2] + face['bbox'][0]) / 2 - face_center[0]) ** 2 + (
                (face['bbox'][3] + face['bbox'][1]) / 2 - face_center[1]) ** 2) ** 0.5)
    return faces


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class FaceAnalysisModel:
    def __init__(self, **kwargs):
        self.model_paths = kwargs.get("model_path", [])
        self.predict_type = kwargs.get("predict_type", "trt")
        self.device = torch.cuda.current_device()
        self.cudaStream = torch.cuda.current_stream().cuda_stream

        assert self.model_paths
        self.face_det = get_predictor(predict_type=self.predict_type, model_path=self.model_paths[0])
        self.face_det.input_spec()
        self.face_det.output_spec()
        self.face_pose = get_predictor(predict_type=self.predict_type, model_path=self.model_paths[1])
        self.face_pose.input_spec()
        self.face_pose.output_spec()

        # face det
        self.input_mean = 127.5
        self.input_std = 128.0
        # print(self.output_names)
        # assert len(outputs)==10 or len(outputs)==15
        self.use_kps = False
        self._anchor_ratio = 1.0
        self._num_anchors = 1
        self.center_cache = {}
        self.nms_thresh = 0.4
        self.det_thresh = 0.5
        self.input_size = (512, 512)
        if len(self.face_det.outputs) == 6:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
        elif len(self.face_det.outputs) == 9:
            self.fmc = 3
            self._feat_stride_fpn = [8, 16, 32]
            self._num_anchors = 2
            self.use_kps = True
        elif len(self.face_det.outputs) == 10:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
        elif len(self.face_det.outputs) == 15:
            self.fmc = 5
            self._feat_stride_fpn = [8, 16, 32, 64, 128]
            self._num_anchors = 1
            self.use_kps = True

        self.lmk_dim = 2
        self.lmk_num = 212 // self.lmk_dim

    def nms(self, dets):
        thresh = self.nms_thresh
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]

        return keep

    def detect_face(self, *data):
        img = data[0]  # BGR mode
        im_ratio = float(img.shape[0]) / img.shape[1]
        input_size = self.input_size
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])

        det_img = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
        det_img = np.transpose(det_img, (2, 0, 1))
        det_img = (det_img - self.input_mean) / self.input_std
        if self.predict_type == "trt":
            nvtx.range_push("forward")
            feed_dict = {}
            inp = self.face_det.inputs[0]
            det_img_torch = torch.from_numpy(det_img[None]).to(device=self.device,
                                                               dtype=numpy_to_torch_dtype_dict[inp['dtype']])
            feed_dict[inp['name']] = det_img_torch
            preds_dict = self.face_det.predict(feed_dict, self.cudaStream)
            outs = []
            for key in ["448", "471", "494", "451", "474", "497", "454", "477", "500"]:
                outs.append(preds_dict[key].cpu().numpy())
            o448, o471, o494, o451, o474, o497, o454, o477, o500 = outs
            nvtx.range_pop()
        else:
            o448, o471, o494, o451, o474, o497, o454, o477, o500 = self.face_det.predict(det_img[None])
        faces_det = [o448, o471, o494, o451, o474, o497, o454, o477, o500]
        input_height = det_img.shape[1]
        input_width = det_img.shape[2]
        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = faces_det[idx]
            bbox_preds = faces_det[idx + fmc]
            bbox_preds = bbox_preds * stride
            if self.use_kps:
                kps_preds = faces_det[idx + fmc * 2] * stride
            height = input_height // stride
            width = input_width // stride
            K = height * width
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                # solution-3:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                # print(anchor_centers.shape)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.det_thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]
            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                # kpss = kps_preds
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)
        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]
        bboxes = np.vstack(bboxes_list) / det_scale
        if self.use_kps:
            kpss = np.vstack(kpss_list) / det_scale
        pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        pre_det = pre_det[order, :]
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        if self.use_kps:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]
        else:
            kpss = None
        return det, kpss

    def estimate_face_pose(self, *data):
        """
        检测脸部关键点
        :param data:
        :return:
        """
        img, face = data
        bbox = face.bbox
        w, h = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
        center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
        rotate = 0
        input_size = (192, 192)
        _scale = input_size[0] / (max(w, h) * 1.5)
        aimg, M = face_align.transform(img, center, input_size[0], _scale, rotate)
        input_size = tuple(aimg.shape[0:2][::-1])

        aimg = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
        aimg = np.transpose(aimg, (2, 0, 1))
        if self.predict_type == "trt":
            nvtx.range_push("forward")
            feed_dict = {}
            inp = self.face_pose.inputs[0]
            det_img_torch = torch.from_numpy(aimg[None]).to(device=self.device,
                                                            dtype=numpy_to_torch_dtype_dict[inp['dtype']])
            feed_dict[inp['name']] = det_img_torch
            preds_dict = self.face_pose.predict(feed_dict, self.cudaStream)
            outs = []
            for i, out in enumerate(self.face_pose.outputs):
                outs.append(preds_dict[out["name"]].cpu().numpy())
            pred = outs[0]
            nvtx.range_pop()
        else:
            pred = self.face_pose.predict(aimg[None])[0]
        pred = pred.reshape((-1, 2))
        if self.lmk_num < pred.shape[0]:
            pred = pred[self.lmk_num * -1:, :]
        pred[:, 0:2] += 1
        pred[:, 0:2] *= (input_size[0] // 2)
        if pred.shape[1] == 3:
            pred[:, 2] *= (input_size[0] // 2)

        IM = cv2.invertAffineTransform(M)
        pred = face_align.trans_points(pred, IM)
        face["landmark"] = pred
        return pred

    def predict(self, *data, **kwargs):
        bboxes, kpss = self.detect_face(*data)
        if bboxes.shape[0] == 0:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i]
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            self.estimate_face_pose(data[0], face)
            ret.append(face)
        ret = sort_by_direction(ret, 'large-small', None)
        outs = [x.landmark for x in ret]
        return outs

    def __del__(self):
        del self.face_det
        del self.face_pose
