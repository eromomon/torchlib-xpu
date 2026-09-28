# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Zero-max-length regression for the jagged-output operators.

The jagged-output kernels write *every* output row: a row whose jagged
coordinate falls outside the dense tensor takes the ``truncated`` branch and is
written as ``f(x, 0, 0)`` - zero for ``dense_to_jagged`` (which ignores ``x``)
and ``x`` for ``add_jagged_output``. That is what lets both callers allocate
their output with ``at::empty``.

Their shared host helpers, however, skip the launch entirely when the dense
operand is empty (``jagged_common.h``, ``y.numel() == 0``). An empty dense
operand does not imply an empty output: a dense tensor with a zero-size jagged
dim, ``[B, 0, D]``, still pairs with a ``total_L > 0`` jagged output. Nothing
then writes that output, and it is returned uninitialized - NaNs under
``fill_uninitialized_memory``, arbitrary bytes without it.

CPU returns zeros and unchanged ``x`` for these inputs, so the outputs are
pinned against a CPU reference here rather than against hardcoded values.

.. note::
   FBGEMM's CUDA source has the same defect - the same guard in
   ``common.cuh:835`` and ``common.cuh:965``, with the same ``at::empty``
   allocations in its callers - so CUDA returns uninitialized memory for these
   inputs too. Initializing them is a deliberate deviation from the CUDA source
   rather than a porting correction; do not "restore" it to match upstream.
"""

import fbgemm_xpu  # noqa: F401  - registers the fbgemm XPU operators
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="requires an XPU device"
)

_B = 1  # batch rows
_MAX_L = 0  # padded jagged length: every output row is truncated
_E = 8  # embedding dim; a multiple of 8 keeps rows 128-bit aligned
_TOTAL_L = 2  # jagged rows, so the output is non-empty while the dense is not

# Half takes the FP16 fast path, float32 the generic kernel; the early return
# under test is duplicated in both host helpers.
_DTYPES = [torch.float16, torch.float32]


@pytest.fixture
def fill_uninitialized_memory():
    """Make an unwritten output observable as NaN instead of stale bytes."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    yield
    torch.utils.deterministic.fill_uninitialized_memory = fill
    torch.use_deterministic_algorithms(deterministic)


@pytest.mark.usefixtures("fill_uninitialized_memory")
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_dense_to_jagged_zero_max_length(dtype, index_dtype):
    """A zero-max-length dense tensor yields zeros, not uninitialized memory."""
    dense = torch.empty(_B, _MAX_L, _E, dtype=dtype)
    offsets = torch.tensor([0, _TOTAL_L], dtype=index_dtype)

    expected = torch.ops.fbgemm.dense_to_jagged_forward(
        dense, [offsets], _TOTAL_L
    )
    actual = torch.ops.fbgemm.dense_to_jagged_forward(
        dense.to("xpu"), [offsets.to("xpu")], _TOTAL_L
    )

    assert actual.shape == (_TOTAL_L, _E)  # nosec B101
    assert torch.equal(expected, torch.zeros_like(expected)), "CPU baseline"  # nosec B101
    assert torch.equal(actual.cpu(), expected)  # nosec B101


@pytest.mark.usefixtures("fill_uninitialized_memory")
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_add_jagged_output_zero_max_length(dtype, index_dtype):
    """A zero-max-length dense operand leaves the jagged operand unchanged."""
    x_values = torch.arange(1, _TOTAL_L * _E + 1, dtype=dtype).view(_TOTAL_L, _E)
    dense = torch.empty(_B, _MAX_L, _E, dtype=dtype)
    offsets = torch.tensor([0, _TOTAL_L], dtype=index_dtype)

    expected, _ = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
        x_values, [offsets], dense
    )
    actual, _ = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
        x_values.to("xpu"), [offsets.to("xpu")], dense.to("xpu")
    )

    assert torch.equal(expected, x_values), "CPU baseline"  # nosec B101
    assert torch.equal(actual.cpu(), expected)  # nosec B101
