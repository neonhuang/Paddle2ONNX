# Copyright (c) 2021  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from auto_scan_test import OPConvertAutoScanTest, BaseNet
from hypothesis import reproduce_failure
import hypothesis.strategies as st
import numpy as np
import unittest
import paddle
import paddle.fluid as fluid
from paddle2onnx.command import program2onnx
import logging
from onnxbase import randtool, compare
import onnxruntime as rt

paddle.enable_static()
np.random.seed(33)
import numpy


def test_grid_sample_align_corners():
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    # onnxruntime 中的Floor不支持float64
    dtype = 'float32'
    align_corners = True
    N = 5
    with paddle.static.program_guard(main_program, startup_program):
        x = fluid.layers.data(name='x', shape=[6])
        y = fluid.layers.data(name='y', shape=[6], lod_level=2)
        out = fluid.layers.lod_reset(x=x, y=y)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(paddle.static.default_startup_program())

        x_tensor = fluid.core.LoDTensor()
        x_ndarray = numpy.ones([6]).astype(numpy.float32)
        x_tensor.set(x_ndarray, fluid.CPUPlace())

        y_ndarray = numpy.ones([6]).astype(numpy.float32)
        y_lod = [[2, 2], [2, 2, 1, 1]]
        y_tensor = fluid.create_lod_tensor(y_ndarray, y_lod, fluid.CPUPlace())

        result, = exe.run(feed={"x": x_tensor,
                                "y": y_tensor},
                          fetch_list=[out],
                          return_numpy=False)
        result = np.array(result)
        path_prefix = "./lod_reset"
        fluid.io.save_inference_model(path_prefix, ["x", "y"], [out], exe)
        onnx_path = path_prefix + "/model.onnx"
        program2onnx(
            model_dir=path_prefix,
            save_file=onnx_path,
            opset_version=9,
            enable_onnx_checker=True)

        sess = rt.InferenceSession(onnx_path)
        input_name1 = sess.get_inputs()[0].name
        input_name2 = sess.get_inputs()[1].name
        label_name = sess.get_outputs()[0].name
        pred_onnx = sess.run([label_name],
                             {input_name1: x_ndarray,
                              input_name2: y_ndarray})[0]
        pred_onnx = np.array(pred_onnx)
        compare(pred_onnx, result, 1e-5, 1e-5)


if __name__ == "__main__":
    test_grid_sample_align_corners()
