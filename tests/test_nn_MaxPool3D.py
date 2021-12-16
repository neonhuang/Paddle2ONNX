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

import paddle
from onnxbase import APIOnnx
from onnxbase import randtool


class Net(paddle.nn.Layer):
    """
    simple Net
    """

    def __init__(self,
                 kernel_size=2,
                 stride=None,
                 padding=0,
                 ceil_mode=False,
                 return_mask=False,
                 data_format='NCDHW',
                 name=None):
        super(Net, self).__init__()
        self._max_pool = paddle.nn.MaxPool3D(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            return_mask=return_mask,
            data_format=data_format,
            name=name)

    def forward(self, inputs):
        """
        forward
        """
        x = self._max_pool(inputs)
        return x


def test_MaxPool3D_base():
    """
    api: paddle.nn.MaxPool3D
    op version: 9, 10, 11, 12
    """
    op = Net()
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_MaxPool3D', [9, 10, 11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [20, 1, 21, 20, 21]).astype('float32')))
    obj.run()


def test_MaxPool3D_base_valid():
    """
    api: paddle.nn.MaxPool3D
    op version: 9, 10, 11, 12
    """
    op = Net(padding="VALID")
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_MaxPool3D', [9, 10, 11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [20, 1, 21, 20, 21]).astype('float32')))
    obj.run()


def test_MaxPool3D_base_same():
    """
    api: paddle.nn.MaxPool3D
    op version: 9, 10, 11, 12
    """
    op = Net(padding="SAME")
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_MaxPool3D', [9, 10, 11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [20, 1, 21, 20, 21]).astype('float32')))
    obj.run()


def test_MaxPool3D_base_list0():
    """
    api: paddle.nn.MaxPool3D
    op version: 9, 10, 11, 12
    """
    op = Net(kernel_size=3, stride=3, padding=1)
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_MaxPool3D', [9, 10, 11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [20, 1, 21, 20, 21]).astype('float32')))
    obj.run()


def test_MaxPool3D_base_padding_list0():
    """
    api: paddle.nn.MaxPool3D
    op version: 9, 10, 11, 12
    """
    op = Net(kernel_size=5, stride=5, padding=[1, 2, 3])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_MaxPool3D', [9, 10, 11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [20, 1, 21, 20, 21]).astype('float32')))
    obj.run()


def test_MaxPool3D_base_padding_list1():
    """
    api: paddle.nn.MaxPool3D
    op version: 9, 10, 11, 12
    """
    op = Net(kernel_size=5, stride=5, padding=[1, 2, 3, 4, 0, 4])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_MaxPool3D', [9, 10, 11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [20, 1, 21, 20, 21]).astype('float32')))
    obj.run()


def test_MaxPool3D_base_padding_list2():
    """
    api: paddle.nn.MaxPool3D
    op version: 9, 10, 11, 12
    """
    op = Net(kernel_size=5,
             stride=5,
             padding=[[0, 0], [0, 0], [1, 2], [3, 4], [0, 4]])
    op.eval()
    # net, name, ver_list, delta=1e-6, rtol=1e-5
    obj = APIOnnx(op, 'nn_MaxPool3D', [9, 10, 11, 12])
    obj.set_input_data(
        "input_data",
        paddle.to_tensor(
            randtool("float", -1, 1, [20, 1, 21, 20, 21]).astype('float32')))
    obj.run()
