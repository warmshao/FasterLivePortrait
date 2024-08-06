# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: predictor.py
import pdb
import threading
import os

import numpy as np
import onnxruntime
import platform

try:
    import tensorrt as trt
    import ctypes
except ModuleNotFoundError:
    print("No TensorRT Found")

# Use autoprimaryctx if available (pycuda >= 2021.1) to
# prevent issues with other modules that rely on the primary
# device context.
try:
    import pycuda.driver as cuda
    import pycuda.autoprimaryctx
    import pycuda.autoinit
except ModuleNotFoundError:
    print("No PyCUDA Found")

import contextlib


class TensorRTPredictor:
    """
    Implements inference for the EfficientDet TensorRT engine.
    """
    _cuda_initialized = False
    _lock = threading.Lock()

    @classmethod
    def _ensure_cuda_initialized(cls):
        with cls._lock:
            if not cls._cuda_initialized:
                cuda.init()
                cls._cuda_initialized = True

    def __init__(self, **kwargs):
        """
        :param engine_path: The path to the serialized engine to load from disk.
        """
        self._ensure_cuda_initialized()
        self.cuda_context = None
        self.stream = None
        try:
            if kwargs.get("cuda_ctx", None) is None:
                self.cuda_context = cuda.Device(0).make_context()
            else:
                self.cuda_context = kwargs.get("cuda_ctx")

            self.stream = cuda.Stream()

            # Load TRT engine
            self.logger = trt.Logger(trt.Logger.ERROR)
            # TODO: set the .so file in global path
            if platform.system().lower() == 'linux':
                ctypes.CDLL("./checkpoints/liveportrait_onnx/libgrid_sample_3d_plugin.so", mode=ctypes.RTLD_GLOBAL)
            else:
                ctypes.CDLL("./checkpoints/liveportrait_onnx/grid_sample_3d_plugin.dll", mode=ctypes.RTLD_GLOBAL,
                            winmode=0)
            trt.init_libnvinfer_plugins(self.logger, "")
            engine_path = kwargs.get("model_path", None)
            self.debug = kwargs.get("debug", False)
            assert engine_path, f"model:{engine_path} must exist!"
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                assert runtime
                self.engine = runtime.deserialize_cuda_engine(f.read())
            assert self.engine
            self.context = self.engine.create_execution_context()
            assert self.context

            # Setup I/O bindings
            self.inputs = []
            self.outputs = []
            self.allocations = []
            for i in range(self.engine.num_bindings):
                is_input = False
                if self.engine.binding_is_input(i):
                    is_input = True
                name = self.engine.get_binding_name(i)
                dtype = np.dtype(trt.nptype(self.engine.get_binding_dtype(i)))
                shape = self.context.get_binding_shape(i)
                if is_input and shape[0] < 0:
                    assert self.engine.num_optimization_profiles > 0
                    profile_shape = self.engine.get_profile_shape(0, name)
                    assert len(profile_shape) == 3  # min,opt,max
                    # Set the *max* profile as binding shape
                    self.context.set_binding_shape(i, profile_shape[2])
                    shape = self.context.get_binding_shape(i)
                if is_input:
                    self.batch_size = shape[0]
                size = dtype.itemsize
                for s in shape:
                    size *= s
                allocation = cuda.mem_alloc(size)
                host_allocation = None if is_input else np.zeros(shape, dtype)
                binding = {
                    "index": i,
                    "name": name,
                    "dtype": dtype,
                    "shape": list(shape),
                    "allocation": allocation,
                    "host_allocation": host_allocation,
                }
                self.allocations.append(allocation)
                if self.engine.binding_is_input(i):
                    self.inputs.append(binding)
                else:
                    self.outputs.append(binding)
            # print("{} '{}' with shape {} and dtype {}".format(
            #     "Input" if is_input else "Output",
            #     binding['name'], binding['shape'], binding['dtype']))

            assert self.batch_size > 0
            assert len(self.inputs) > 0
            assert len(self.outputs) > 0
            assert len(self.allocations) > 0
        except Exception as e:
            self.cleanup()
            raise e

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        specs = []
        for i, o in enumerate(self.inputs):
            specs.append((o["name"], o['shape'], o['dtype']))
            if self.debug:
                print(f"trt input {i} -> {self.inputs[i]['name']}")
        return specs

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for i, o in enumerate(self.outputs):
            specs.append((o["name"], o['shape'], o['dtype']))
            if self.debug:
                print(f"trt output {i} -> {o['name']}")
        return specs

    def predict(self, *data):
        if self.cuda_context:
            self.cuda_context.push()
        try:
            # 使用 self.stream 进行异步操作
            for i in range(len(data)):
                data_ = np.ascontiguousarray(data[i], self.inputs[i]["dtype"])
                cuda.memcpy_htod_async(self.inputs[i]['allocation'], data_, self.stream)

            # 异步执行推理
            self.context.execute_async_v2(self.allocations, self.stream.handle)

            # 异步复制输出数据
            for o in range(len(self.outputs)):
                cuda.memcpy_dtoh_async(self.outputs[o]['host_allocation'],
                                       self.outputs[o]['allocation'],
                                       self.stream)

            # 同步流，确保所有操作完成
            self.stream.synchronize()

            return [o['host_allocation'] for o in self.outputs]
        finally:
            if self.cuda_context:
                self.cuda_context.pop()

    def cleanup(self):
        if self.cuda_context:
            self.cuda_context.push()
        try:
            if hasattr(self, 'context'):
                del self.context
            if hasattr(self, 'engine'):
                del self.engine
            if hasattr(self, 'allocations'):
                for allocation in self.allocations:
                    allocation.free()
                self.allocations.clear()
        finally:
            if self.cuda_context:
                self.cuda_context.pop()
                self.cuda_context.detach()
            self.cuda_context = None
            self.stream = None

    def __del__(self):
        try:
            self.cleanup()
        except Exception as e:
            print(f"Error during cleanup: {e}")


class OnnxRuntimePredictor:
    """
    OnnxRuntime Prediction
    """

    def __init__(self, **kwargs):
        model_path = kwargs.get("model_path", "")  # 用模型路径区分是否是一样的实例
        assert os.path.exists(model_path), "model path must exist!"
        # print("loading ort model:{}".format(model_path))
        self.debug = kwargs.get("debug", False)
        providers = ['CUDAExecutionProvider', 'CoreMLExecutionProvider', 'CPUExecutionProvider']

        print(f"OnnxRuntime use {providers}")
        opts = onnxruntime.SessionOptions()
        # opts.inter_op_num_threads = kwargs.get("num_threads", 4)
        # opts.intra_op_num_threads = kwargs.get("num_threads", 4)
        opts.log_severity_level = 3
        self.onnx_model = onnxruntime.InferenceSession(model_path, providers=providers, sess_options=opts)
        self.inputs = self.onnx_model.get_inputs()
        self.outputs = self.onnx_model.get_outputs()

    def input_spec(self):
        """
        Get the specs for the input tensor of the network. Useful to prepare memory allocations.
        :return: Two items, the shape of the input tensor and its (numpy) datatype.
        """
        specs = []
        for i, o in enumerate(self.inputs):
            specs.append((o.name, o.shape, o.type))
            if self.debug:
                print(f"ort {i} -> {o.name}")
        return specs

    def output_spec(self):
        """
        Get the specs for the output tensors of the network. Useful to prepare memory allocations.
        :return: A list with two items per element, the shape and (numpy) datatype of each output tensor.
        """
        specs = []
        for i, o in enumerate(self.outputs):
            specs.append((o.name, o.shape, o.type))
            if self.debug:
                print(f"ort output {i} -> {o.name}")
        return specs

    def predict(self, *data):
        input_feeds = {}
        for i in range(len(data)):
            if "float16" in self.inputs[i].type:
                input_feeds[self.inputs[i].name] = data[i].astype(np.float16)
            else:
                input_feeds[self.inputs[i].name] = data[i].astype(np.float32)

        results = self.onnx_model.run(None, input_feeds)
        return results

    def __del__(self):
        del self.onnx_model
        self.onnx_model = None


class OnnxRuntimePredictorSingleton(OnnxRuntimePredictor):
    """
    单例模式，防止模型被加载多次
    """
    _instance_lock = threading.Lock()
    _instance = {}

    def __new__(cls, *args, **kwargs):
        model_path = kwargs.get("model_path", "")  # 用模型路径区分是否是一样的实例
        assert os.path.exists(model_path), "model path must exist!"
        # 单例模式，避免重复加载模型
        with OnnxRuntimePredictorSingleton._instance_lock:
            if model_path not in OnnxRuntimePredictorSingleton._instance or \
                    OnnxRuntimePredictorSingleton._instance[model_path].onnx_model is None:
                OnnxRuntimePredictorSingleton._instance[model_path] = OnnxRuntimePredictor(**kwargs)

        return OnnxRuntimePredictorSingleton._instance[model_path]


def get_predictor(**kwargs):
    predict_type = kwargs.get("predict_type", "trt")
    if predict_type == "ort":
        return OnnxRuntimePredictorSingleton(**kwargs)
    elif predict_type == "trt":
        return TensorRTPredictor(**kwargs)
    else:
        raise NotImplementedError
