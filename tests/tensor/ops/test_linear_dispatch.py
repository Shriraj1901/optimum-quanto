# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import pytest
import torch
from helpers import assert_similar, random_qactivation, random_qweight, random_tensor

from optimum.quanto import Calibration, freeze, qfloat8, qint2, qint4, qint8, quantize


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(5, 5), (32, 32), (10, 32)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16], ids=["fp32", "fp16"])
@pytest.mark.parametrize("activation_qtype", [None, qint8], ids=["a-none", "a-qint8"])
@pytest.mark.parametrize("weight_qtype", [qint2, qint4, qint8], ids=["w-qint2", "w-qint4", "w-qint8"])
def test_qactivation_qweight_linear(
    batch_size, tokens, embeddings, use_bias, dtype, activation_qtype, weight_qtype, device
):
    input_shape = (batch_size, tokens, embeddings)
    if activation_qtype is None:
        inputs = random_tensor(input_shape, dtype=dtype).to(device)
    else:
        inputs = random_qactivation(input_shape, qtype=activation_qtype, dtype=dtype).to(device)
    qweight = random_qweight((embeddings, embeddings), qtype=weight_qtype, dtype=dtype, axis=0).to(device)
    bias = random_tensor((embeddings,), dtype=dtype).to(device) if use_bias else None
    qout = torch.nn.functional.linear(inputs, qweight, bias)
    if activation_qtype is not None:
        inputs = inputs.dequantize()
    out = torch.nn.functional.linear(inputs, qweight.dequantize(), bias)
    assert_similar(out, qout)


@pytest.mark.parametrize("activation_qtype", [qint8, qfloat8], ids=["a-qint8", "a-qfloat8"])
def test_linear_fp16_small_scales(activation_qtype, device):
    # The product of the input and weight scales is far below the smallest normal fp16 number
    generator = torch.Generator().manual_seed(0)
    row_max = torch.tensor([0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0])
    weight = torch.randn(8, 256, generator=generator)
    weight = weight / weight.abs().amax(1, keepdim=True) * row_max.view(8, 1)
    inputs = torch.randn(16, 256, generator=generator)
    inputs = (inputs / inputs.abs().max() * 0.03).half().to(device)
    model = torch.nn.Sequential(torch.nn.Linear(256, 8, bias=False))
    model[0].weight.data.copy_(weight)
    model = model.half().to(device)
    expected = model(inputs).float()
    quantize(model, weights=qint8, activations=activation_qtype)
    with Calibration():
        model(inputs)
    freeze(model)
    with torch.no_grad():
        out = model(inputs)
    assert out.dtype == torch.float16
    error = (out.float() - expected).norm(dim=0) / expected.norm(dim=0)
    assert error.max() < 0.1


@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(256, 256)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
def test_linear_fp16_int4(batch_size, tokens, embeddings, use_bias, device):
    dtype = torch.float16
    weight_qtype = qint4
    inputs = torch.rand((batch_size,) + (tokens, embeddings), dtype=dtype, device=device)
    qweight = random_qweight((embeddings, embeddings), weight_qtype, dtype=dtype, axis=0, group_size=128).to(device)
    bias = random_tensor((embeddings,), dtype=dtype).to(device) if use_bias else None
    qout = torch.nn.functional.linear(inputs, qweight, bias)
    out = torch.nn.functional.linear(inputs, qweight.dequantize(), bias)
    assert_similar(out, qout)


@pytest.mark.skip_device("mps")  # Only available with pytorch 2.4
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("tokens, embeddings", [(256, 256)])
@pytest.mark.parametrize("use_bias", [True, False], ids=["bias", "no-bias"])
def test_linear_bf16_int4(batch_size, tokens, embeddings, use_bias, device):
    dtype = torch.bfloat16
    weight_qtype = qint4
    input_shape = (batch_size, tokens, embeddings)
    inputs = torch.rand(input_shape, dtype=dtype, device=device)
    weight_shape = (embeddings, embeddings)
    qweight = random_qweight(weight_shape, weight_qtype, dtype=dtype, axis=0, group_size=128, device=device)
    bias = random_tensor((embeddings,), dtype=dtype).to(device) if use_bias else None
    qout = torch.nn.functional.linear(inputs, qweight, bias)
    out = torch.nn.functional.linear(inputs, qweight.dequantize(), bias)
    assert_similar(out, qout)
