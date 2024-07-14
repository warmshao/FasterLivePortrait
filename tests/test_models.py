# -*- coding: utf-8 -*-
# @Time    : 2024/7/13 17:20
# @Author  : shaoguowen
# @Email   : wenshaoguo1026@gmail.com
# @Project : RealTimeLivePortrait
# @FileName: test_models.py

import os, sys
import pdb

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def test_warping_model():
    """
    test warping model in onnx and trt
    :return:
    """
    import numpy as np
    import time
    from models import WarpingModel

    # tensorrt 模型加载
    trt_kwargs = dict(
        predict_type="trt",
        model_path="./checkpoints/liveportrait_onnx/warping-fix.trt",
    )

    trt_model = WarpingModel(**trt_kwargs)

    # onnx 模型加载
    onnx_kwargs = dict(
        predict_type="ort",
        model_path="./checkpoints/liveportrait_onnx/warping.onnx",
        use_cuda=True,
        num_threads=8
    )
    onnx_model = WarpingModel(**onnx_kwargs)

    feature_3d = np.random.randn(1, 32, 16, 64, 64)
    kp_source = np.random.randn(1, 21, 3)
    kp_driving = np.random.randn(1, 21, 3)

    trt_rets = trt_model.predict(feature_3d, kp_source, kp_driving)
    onnx_rets = onnx_model.predict(feature_3d, kp_source, kp_driving)

    for i in range(len(trt_rets)):
        print(f"output {i} max diff:{np.abs(trt_rets[i] - onnx_rets[i]).max()}")

    infer_times = []
    for _ in range(20):
        t0 = time.time()
        trt_rets = trt_model.predict(feature_3d, kp_source, kp_driving)
        infer_times.append(time.time() - t0)
    print("{} tensorrt inference time: min: {}, max: {}, mean: {}".format(WarpingModel.__name__, np.min(infer_times),
                                                                          np.max(infer_times), np.median(infer_times)))

    # onnx is so slow, don't why, maybe the grid_sample op not implemented well?
    infer_times = []
    for _ in range(20):
        t0 = time.time()
        onnx_rets = onnx_model.predict(feature_3d, kp_source, kp_driving)
        infer_times.append(time.time() - t0)
    print("{} onnx inference time: min: {}, max: {}, mean: {}".format(WarpingModel.__name__, np.min(infer_times),
                                                                      np.max(infer_times), np.median(infer_times)))


def test_spade_gen_model():
    """
    test spade generator model in onnx and trt
    :return:
    """
    import numpy as np
    import time
    from models import SpadeGenModel

    # tensorrt 模型加载
    trt_kwargs = dict(
        predict_type="trt",
        model_path="./checkpoints/liveportrait_onnx/spade_generator.trt",
    )

    trt_model = SpadeGenModel(**trt_kwargs)

    # onnx 模型加载
    onnx_kwargs = dict(
        predict_type="ort",
        model_path="./checkpoints/liveportrait_onnx/spade_generator.onnx",
        use_cuda=True,
        num_threads=4
    )
    onnx_model = SpadeGenModel(**onnx_kwargs)

    input = np.random.randn(1, 256, 64, 64)

    trt_rets = trt_model.predict(input)
    onnx_rets = onnx_model.predict(input)
    pdb.set_trace()
    print(f"output max diff:{np.abs(trt_rets - onnx_rets).max()}")

    infer_times = []
    for _ in range(30):
        t0 = time.time()
        trt_rets = trt_model.predict(input)
        infer_times.append(time.time() - t0)
    print("{} tensorrt inference time: min: {}, max: {}, mean: {}".format(SpadeGenModel.__name__, np.min(infer_times),
                                                                          np.max(infer_times), np.median(infer_times)))

    infer_times = []
    for _ in range(300):
        t0 = time.time()
        onnx_rets = onnx_model.predict(input)
        infer_times.append(time.time() - t0)
    print("{} onnx inference time: min: {}, max: {}, mean: {}".format(SpadeGenModel.__name__, np.min(infer_times),
                                                                      np.max(infer_times), np.median(infer_times)))


def test_motion_extractor_model():
    """
    test motion_extractor model in onnx and trt
    :return:
    """
    import numpy as np
    import time
    from models import MotionExtractorModel

    # tensorrt 模型加载
    trt_kwargs = dict(
        predict_type="trt",
        model_path="./checkpoints/liveportrait_onnx/motion_extractor.trt",
    )

    trt_model = MotionExtractorModel(**trt_kwargs)

    # onnx 模型加载
    onnx_kwargs = dict(
        predict_type="ort",
        model_path="./checkpoints/liveportrait_onnx/motion_extractor.onnx",
        use_cuda=True,
        num_threads=8
    )
    onnx_model = MotionExtractorModel(**onnx_kwargs)

    input = np.random.randn(1, 3, 256, 256)

    trt_rets = trt_model.predict(input)
    onnx_rets = onnx_model.predict(input)
    for i in range(len(trt_rets)):
        print(f"output {i} max diff:{np.abs(trt_rets[i] - onnx_rets[i]).max()}")

    infer_times = []
    for _ in range(20):
        t0 = time.time()
        trt_rets = trt_model.predict(input)
        infer_times.append(time.time() - t0)
    print("{} tensorrt inference time: min: {}, max: {}, mean: {}".format(MotionExtractorModel.__name__,
                                                                          np.min(infer_times),
                                                                          np.max(infer_times), np.mean(infer_times)))

    # onnx is so slow, don't why, maybe the grid_sample op not implemented well?
    infer_times = []
    for _ in range(20):
        t0 = time.time()
        onnx_rets = onnx_model.predict(input)
        infer_times.append(time.time() - t0)
    print(
        "{} onnx inference time: min: {}, max: {}, mean: {}".format(MotionExtractorModel.__name__, np.min(infer_times),
                                                                    np.max(infer_times), np.mean(infer_times)))


def test_appearance_extractor_model():
    """
    test motion_extractor model in onnx and trt
    :return:
    """
    import numpy as np
    import time
    from models import AppearanceFeatureExtractorModel

    # tensorrt 模型加载
    trt_kwargs = dict(
        predict_type="trt",
        model_path="./checkpoints/liveportrait_onnx/appearance_feature_extractor.trt",
    )

    trt_model = AppearanceFeatureExtractorModel(**trt_kwargs)

    # onnx 模型加载
    onnx_kwargs = dict(
        predict_type="ort",
        model_path="./checkpoints/liveportrait_onnx/appearance_feature_extractor.onnx",
        use_cuda=True,
        num_threads=8
    )
    onnx_model = AppearanceFeatureExtractorModel(**onnx_kwargs)

    input = np.random.randn(1, 3, 256, 256)

    trt_rets = trt_model.predict(input)
    onnx_rets = onnx_model.predict(input)
    print(f"output max diff:{np.abs(trt_rets - onnx_rets).max()}")

    infer_times = []
    for _ in range(20):
        t0 = time.time()
        trt_rets = trt_model.predict(input)
        infer_times.append(time.time() - t0)
    print("{} tensorrt inference time: min: {}, max: {}, mean: {}".format(AppearanceFeatureExtractorModel.__name__,
                                                                          np.min(infer_times),
                                                                          np.max(infer_times), np.mean(infer_times)))

    # onnx is so slow, don't why, maybe the grid_sample op not implemented well?
    infer_times = []
    for _ in range(20):
        t0 = time.time()
        onnx_rets = onnx_model.predict(input)
        infer_times.append(time.time() - t0)
    print(
        "{} onnx inference time: min: {}, max: {}, mean: {}".format(AppearanceFeatureExtractorModel.__name__,
                                                                    np.min(infer_times),
                                                                    np.max(infer_times), np.mean(infer_times)))


def test_landmark_model():
    """
    test motion_extractor model in onnx and trt
    :return:
    """
    import numpy as np
    import time
    from models import LandmarkModel

    # tensorrt 模型加载
    trt_kwargs = dict(
        predict_type="trt",
        model_path="./checkpoints/liveportrait_onnx/landmark.trt",
    )

    trt_model = LandmarkModel(**trt_kwargs)

    # onnx 模型加载
    onnx_kwargs = dict(
        predict_type="ort",
        model_path="./checkpoints/liveportrait_onnx/landmark.onnx",
        use_cuda=True,
        num_threads=4
    )
    onnx_model = LandmarkModel(**onnx_kwargs)

    input = np.random.randn(1, 3, 224, 224)

    trt_rets = trt_model.predict(input)
    onnx_rets = onnx_model.predict(input)
    print(f"output max diff:{np.abs(trt_rets - onnx_rets).max()}")

    infer_times = []
    for _ in range(20):
        t0 = time.time()
        trt_rets = trt_model.predict(input)
        infer_times.append(time.time() - t0)
    print("{} tensorrt inference time: min: {}, max: {}, mean: {}".format(LandmarkModel.__name__,
                                                                          np.min(infer_times),
                                                                          np.max(infer_times), np.mean(infer_times)))

    # onnx is so slow, don't why, maybe the grid_sample op not implemented well?
    infer_times = []
    for _ in range(20):
        t0 = time.time()
        onnx_rets = onnx_model.predict(input)
        infer_times.append(time.time() - t0)
    print(
        "{} onnx inference time: min: {}, max: {}, mean: {}".format(LandmarkModel.__name__,
                                                                    np.min(infer_times),
                                                                    np.max(infer_times), np.mean(infer_times)))


if __name__ == '__main__':
    test_warping_model()
    # test_spade_gen_model()
    # test_motion_extractor_model()
    # test_landmark_model()
