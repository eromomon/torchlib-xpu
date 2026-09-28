# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Sentinel-range regression for the FP16 jagged fast path.

``offsets`` carries ``B + 1`` entries: one start per batch row plus a final
sentinel marking the end of the last row. ``total_L`` is a caller-supplied
parameter, so ``values`` may hold more rows than ``offsets[B]`` accounts for -
those rows belong to no batch row, and every other implementation of this
operator resolves them to zero (the generic kernel's ``truncated`` branch, which
evaluates ``f(x, 0, 0)``).

The FP16 fast path recovers the batch row with a binary search over the staged
offsets. Seeding that search with ``B - 1`` stops it at ``offsets[B - 1]`` and
never examines the sentinel, so it cannot return ``dense_row == B`` - the one
value that tells the gather kernel's bounds check that a row is past the end.
Such a row instead resolves to ``B - 1``, passes the bounds check, and gathers
the *last dense row* in place of zeros.

The case below is the smallest that reaches it: batch row 0 owns one value, batch
row 1 owns none, and ``total_L`` is 2, so values row 1 is past the last offset.
Before the fix this returned dense row 1 for both offset dtypes; the search
kernel is templated on ``index_t`` and dispatched through
``AT_DISPATCH_INDEX_TYPES``, so int32 and int64 compile to separate kernels and
both need covering.

.. note::
   FBGEMM's CUDA source has the same defect (``common.cuh:475``), and its
   ``matches_opt`` does not screen this input out either, so CUDA takes its fast
   path and returns the same wrong row. The XPU seed is therefore a deliberate
   deviation from the CUDA source rather than a porting correction - do not
   "restore" it to match upstream.
"""

import fbgemm_xpu  # noqa: F401  - registers the fbgemm XPU operators
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="requires an XPU device"
)

# matches_opt() only selects the FP16 path for Half.
_DTYPE = torch.float16

_B = 2  # batch rows
_MAX_L = 1  # padded jagged length
_E = 8  # embedding dim; a multiple of 8 keeps rows 128-bit aligned
_TOTAL_L = 2  # one row past offsets[-1], which is 1


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_dense_to_jagged_fp16_row_past_last_offset(index_dtype):
    """A values row past offsets[-1] gathers zeros, not the last dense row."""
    dense = torch.arange(
        1, _B * _MAX_L * _E + 1, device="xpu", dtype=_DTYPE
    ).view(_B, _MAX_L, _E)
    # offsets[1] == offsets[2] == 1: row 0 owns value 0, row 1 owns nothing.
    offsets = torch.tensor([0, 1, 1], device="xpu", dtype=index_dtype)

    # Guard the fast path's own selection criteria, so a future change to the
    # shapes above cannot quietly reroute this to the generic kernel and leave
    # the regression uncovered. matches_opt() also inspects the internally
    # allocated output, which is not reachable from here.
    assert dense.dim() - 2 == 1, "fast path requires exactly one jagged dim"  # nosec B101
    assert dense.stride(-1) == 1  # nosec B101
    assert dense.stride(-2) % 8 == 0  # nosec B101
    assert dense.data_ptr() % 16 == 0  # nosec B101

    output = torch.ops.fbgemm.dense_to_jagged_forward(dense, [offsets], _TOTAL_L)

    assert output.shape == (_TOTAL_L, _E)  # nosec B101
    # Row 0 is inside the jagged region and gathers dense row 0.
    assert torch.equal(output[0], dense[0, 0])  # nosec B101
    # Row 1 is past offsets[-1]. Before the fix this was dense[1, 0].
    assert torch.equal(output[1], torch.zeros_like(output[1]))  # nosec B101


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_dense_to_jagged_fp16_matches_generic_path(index_dtype):
    """The FP16 fast path and the generic kernel agree on the same input."""
    values = torch.arange(1, _B * _MAX_L * _E + 1).view(_B, _MAX_L, _E)
    offsets = torch.tensor([0, 1, 1], device="xpu", dtype=index_dtype)

    # float32 misses the Half dispatch case and takes the generic kernel.
    fast = torch.ops.fbgemm.dense_to_jagged_forward(
        values.to(device="xpu", dtype=_DTYPE), [offsets], _TOTAL_L
    )
    generic = torch.ops.fbgemm.dense_to_jagged_forward(
        values.to(device="xpu", dtype=torch.float32), [offsets], _TOTAL_L
    )

    assert torch.equal(fast.to(torch.float32), generic)  # nosec B101
