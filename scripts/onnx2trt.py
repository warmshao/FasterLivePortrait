#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import pdb
import sys
import logging
import argparse
import platform

import tensorrt as trt
import ctypes
import numpy as np

logging.basicConfig(level=logging.INFO)
logging.getLogger("EngineBuilder").setLevel(logging.INFO)
log = logging.getLogger("EngineBuilder")


def load_plugins(logger: trt.Logger):
    # 加载插件库
    if platform.system().lower() == 'linux':
        ctypes.CDLL("./checkpoints/liveportrait_onnx/libgrid_sample_3d_plugin.so", mode=ctypes.RTLD_GLOBAL)
    else:
        ctypes.CDLL("./checkpoints/liveportrait_onnx/grid_sample_3d_plugin.dll", mode=ctypes.RTLD_GLOBAL, winmode=0)
    # 初始化TensorRT的插件库
    trt.init_libnvinfer_plugins(logger, "")


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(self, verbose=False):
        """
        :param verbose: If enabled, a higher verbosity level will be set on the TensorRT logger.
        """
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.trt_logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.trt_logger, namespace="")

        self.builder = trt.Builder(self.trt_logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = 12 * (2 ** 30)  # 12 GB

        profile = self.builder.create_optimization_profile()

        # for face_2dpose_106.onnx
        # profile.set_shape("data", (1, 3, 192, 192), (1, 3, 192, 192), (1, 3, 192, 192))
        # for retinaface_det.onnx
        # profile.set_shape("input.1", (1, 3, 512, 512), (1, 3, 512, 512), (1, 3, 512, 512))

        self.config.add_optimization_profile(profile)
        # 严格类型约束
        self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        self.batch_size = None
        self.network = None
        self.parser = None

        # 加载自定义插件
        load_plugins(self.trt_logger)

    def create_network(self, onnx_path):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.
        :param onnx_path: The path to the ONNX graph to load.
        """
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        self.network = self.builder.create_network(network_flags)
        self.parser = trt.OnnxParser(self.network, self.trt_logger)

        onnx_path = os.path.realpath(onnx_path)
        with open(onnx_path, "rb") as f:
            if not self.parser.parse(f.read()):
                log.error("Failed to load ONNX file: {}".format(onnx_path))
                for error in range(self.parser.num_errors):
                    log.error(self.parser.get_error(error))
                sys.exit(1)

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        log.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            log.info("Input '{}' with shape {} and dtype {}".format(input.name, input.shape, input.dtype))
        for output in outputs:
            log.info("Output '{}' with shape {} and dtype {}".format(output.name, output.shape, output.dtype))
        # assert self.batch_size > 0
        self.builder.max_batch_size = 1

    def create_engine(
            self,
            engine_path,
            precision
    ):
        """
        Build the TensorRT engine and serialize it to disk.
        :param engine_path: The path where to serialize the engine to.
        :param precision: The datatype to use for the engine, either 'fp32', 'fp16' or 'int8'.
        """
        engine_path = os.path.realpath(engine_path)
        engine_dir = os.path.dirname(engine_path)
        os.makedirs(engine_dir, exist_ok=True)
        log.info("Building {} Engine in {}".format(precision, engine_path))

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                log.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)

        with self.builder.build_engine(self.network, self.config) as engine, open(engine_path, "wb") as f:
            log.info("Serializing engine to file: {:}".format(engine_path))
            f.write(engine.serialize())


def main(args):
    builder = EngineBuilder(args.verbose)
    builder.create_network(args.onnx)
    builder.create_engine(
        args.engine,
        args.precision
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx", required=True, help="The input ONNX model file to load")
    parser.add_argument("-e", "--engine", help="The output path for the TRT engine")
    parser.add_argument(
        "-p",
        "--precision",
        default="fp16",
        choices=["fp32", "fp16", "int8"],
        help="The precision mode to build in, either 'fp32', 'fp16' or 'int8', default: 'fp16'",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable more verbose log output")
    args = parser.parse_args()
    if args.engine is None:
        args.engine = args.onnx.replace(".onnx", ".trt")
    main(args)
