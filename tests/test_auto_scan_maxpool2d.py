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


class Net(BaseNet):
    """
    simple Net
    """

    def forward(self, inputs):
        """
        forward
        """
        kernel_size = self.config['kernel_size']
        stride = self.config['stride']
        padding = self.config['padding']
        return_mask = self.config['return_mask']
        ceil_mode = self.config['ceil_mode']
        data_format = self.config['data_format']
        x = paddle.nn.functional.max_pool2d(
            inputs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            return_mask=return_mask,
            ceil_mode=ceil_mode,
            data_format=data_format)
        return x


class TestGroupNormConvert(OPConvertAutoScanTest):
    """
    api: paddle.fluid.layers.nn.group_norm
    OPset version: 7, 9, 15
    """

    def sample_convert_config(self, draw):
        input_shape = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=10), min_size=4, max_size=4))
        input_shape = [3, 2, 10, 10]
        dtype = draw(st.sampled_from(["float32"]))
        data_format = draw(st.sampled_from(["NCHW"]))

        return_mask = False  # draw(st.booleans()) # max_pool2d_with_index
        ceil_mode = draw(st.booleans())

        kernel_size = draw(st.integers(min_value=7, max_value=10))
        if draw(st.booleans()):
            kernel_size = draw(
                st.lists(
                    st.integers(
                        min_value=7, max_value=10),
                    min_size=2,
                    max_size=2))

        stride = None
        if draw(st.booleans()):
            stride = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=5),
                    min_size=2,
                    max_size=2))

        padding_type = draw(
            st.sampled_from(["None", "str", "list", "int", "tuple"]))
        padding = 0
        if padding_type == "str":
            padding = draw(st.sampled_from(["SAME", "VALID"]))
        elif padding_type == "int":
            padding = draw(st.integers(min_value=1, max_value=5))
        elif padding_type == "tuple":
            padding1 = np.expand_dims(
                np.array(
                    draw(
                        st.lists(
                            st.integers(
                                min_value=1, max_value=5),
                            min_size=2,
                            max_size=2))),
                axis=0).tolist()
            padding2 = np.expand_dims(
                np.array(
                    draw(
                        st.lists(
                            st.integers(
                                min_value=1, max_value=5),
                            min_size=2,
                            max_size=2))),
                axis=0).tolist()
            if data_format == "NCHW":
                padding = [[0, 0]] + [[0, 0]] + padding1 + padding2
            else:
                padding = [[0, 0]] + padding1 + padding2 + [[0, 0]]
        elif padding_type == "list":
            if draw(st.booleans()):
                padding = draw(
                    st.lists(
                        st.integers(
                            min_value=1, max_value=5),
                        min_size=2,
                        max_size=2))
            else:
                padding = draw(
                    st.lists(
                        st.integers(
                            min_value=1, max_value=5),
                        min_size=4,
                        max_size=4))

        opset_version = [[7, 9, 15]]
        if ceil_mode:
            opset_version = [10, 15]

        if padding == "VALID":
            return_mask = False

        config = {
            "op_names": ["pool2d"],
            "test_data_shapes": [input_shape],
            "test_data_types": [[dtype]],
            "opset_version": opset_version,
            "input_spec_shape": [],
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "return_mask": return_mask,
            "ceil_mode": ceil_mode,
            "data_format": data_format
        }

        models = Net(config)

        return (config, models)

    def test(self):
        self.run_and_statis(max_examples=30)


if __name__ == "__main__":
    unittest.main()
