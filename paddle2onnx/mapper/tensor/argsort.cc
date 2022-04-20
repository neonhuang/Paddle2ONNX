// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle2onnx/mapper/tensor/argsort.h"

namespace paddle2onnx {
REGISTER_MAPPER(argsort, ArgsortMapper)

void ArgsortMapper::Opset7() {
  auto x_info = GetInput("X");
  auto output_info = GetOutput("Out");
  auto indices_info = GetOutput("Indices");

  auto shape = helper_->MakeNode("Shape", {x_info[0].name})->output(0);
  auto k_node = helper_->Constant({1}, ONNX_NAMESPACE::TensorProto::INT64, axis_);
  auto dim_size = helper_->MakeNode("Gather", {shape, k_node})->output(0);

  auto out_node = helper_->MakeNode("TopK", {x_info[0].name, dim_size}, {output_info[0].name, indices_info[0].name});
  AddAttribute(out_node, "axis", axis_);
  if (!descending_) {
    AddAttribute(out_node, "largest", static_cast<int64_t>(0));
  } else {
    AddAttribute(out_node, "largest", static_cast<int64_t>(1));
  }
}

}  // namespace paddle2onnx
