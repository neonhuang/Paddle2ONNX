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

#pragma once
#include <string>
#include <vector>

#include "paddle2onnx/mapper/mapper.h"

namespace paddle2onnx {

class ArgMaxMapper : public Mapper {
 public:
  ArgMaxMapper(const PaddleParser& p, OnnxHelper* helper, int64_t block_id,
               int64_t op_id)
      : Mapper(p, helper, block_id, op_id) {
    GetAttr("flatten", &flatten_);
    GetAttr("keepdims", &keepdims_);
    GetAttr("axis", &axis_);
    GetAttr("dtype", &dtype_);
  }
  void Opset7();

 private:
  bool flatten_;
  bool keepdims_;
  int64_t axis_;
  int64_t dtype_;
};

}  // namespace paddle2onnx
