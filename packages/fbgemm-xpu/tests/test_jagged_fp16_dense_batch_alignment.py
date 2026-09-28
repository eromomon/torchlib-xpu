# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Batch-stride alignment for the FP16 jagged fast path.

The gather kernel reads a dense row as ``HalfVec8``, a 16-byte type, at
``base + dense_row * stride(0) + dense_col * stride(1)``. ``matches_opt`` proved
that load aligned with three predicates: ``stride(-1) == 1``,
``stride(-2) % 8 == 0`` and ``data_ptr() % 16 == 0``.

The dense operands reach the matcher already canonicalized to 3D
(``y.view({B, -1, E})``), so ``stride(-2)`` is their *folded* stride and the
outer batch stride was never checked. A dense tensor with strides ``(17, 8, 1)``
therefore passed: row 0 sits at the aligned base, while row 1 sits at element 17
- byte 34, which is not a multiple of 16. ``sycl::vec<half, 8>`` carries 16-byte
alignment as a type invariant, so the load is undefined rather than merely slow;
on B60 the lanes came back sourced from the wrong element offset.

``matches_opt`` now also requires ``stride(0) % 8 == 0`` on both dense operands.
Together with the aligned base and the existing ``stride(1)`` check, every
``(row, col)`` address is then a multiple of 16. Rejected layouts fall through to
the generic jagged-output kernel, which indexes through strided accessors and is
correct for any layout, so this narrows the fast path rather than adding a
slow-load branch to it.

Note that the predicate is a multiple-of-8 test and not ``is_contiguous()``: a
batch-broadcast dense operand has ``stride(0) == 0`` and is perfectly aligned,
and rejecting it would cost throughput for nothing.

.. note::
   FBGEMM's CUDA source has the same omission (``common.cuh:722``), and takes its
   fast path on this input too. The extra predicates are a deliberate deviation
   from upstream rather than a porting correction - do not "restore" them.
"""

import fbgemm_xpu  # noqa: F401  - registers the fbgemm XPU operators
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="requires an XPU device"
)

# matches_opt() only selects the FP16 path for Half; float32 takes the generic
# kernel, which is stride-correct for any layout and is the parity reference.
_FAST = torch.float16
_GENERIC = torch.float32

_B = 2  # batch rows
_MAX_L = 1  # padded jagged length
_E = 8  # embedding dim; a multiple of 8 keeps rows 128-bit aligned
_TOTAL_L = 2  # one jagged row per batch row, so row 1 reads dense row 1

_INDEX_DTYPES = [torch.int32, torch.int64]

# Element stride between batch rows. 17 is odd, so dense row 1 lands on an odd
# element offset - byte 34, misaligned for a 16-byte load. 16 is the smallest
# non-contiguous stride that keeps every row aligned.
_MISALIGNED_BATCH_STRIDE = 17
_ALIGNED_BATCH_STRIDE = 16


def _strided_dense(batch_stride, dtype):
    """A [B, MAX_L, E] dense view whose batch rows step by ``batch_stride``."""
    storage = torch.arange(
        1,
        batch_stride * (_B - 1) + _E + 1,
        device="xpu",
        dtype=dtype,
    )
    return storage.as_strided((_B, _MAX_L, _E), (batch_stride, _E, 1))


def _assert_matched_old_criteria(dense, *, batch_stride_aligned):
    """Pin the layout against every criterion but the one under test.

    The input has to satisfy all of matches_opt()'s pre-existing predicates,
    otherwise it is rejected for an unrelated reason and covers nothing.
    matches_opt() also inspects the internally allocated output and jagged
    operand, which are contiguous by construction and not reachable from here.
    """
    assert dense.dim() - 2 == 1, "fast path requires exactly one jagged dim"  # nosec B101
    assert dense.stride(-1) == 1  # nosec B101
    assert dense.stride(-2) % 8 == 0  # nosec B101
    assert dense.data_ptr() % 16 == 0  # nosec B101
    assert (dense.stride(0) % 8 == 0) == batch_stride_aligned  # nosec B101


@pytest.mark.parametrize("batch_stride", [_MISALIGNED_BATCH_STRIDE, _ALIGNED_BATCH_STRIDE])
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
def test_add_strided_dense_matches_generic_path(index_dtype, batch_stride):
    """A strided dense operand adds correctly, aligned batch rows or not."""
    x_values = torch.arange(1, _TOTAL_L * _E + 1).view(_TOTAL_L, _E)
    # One jagged row per batch row: output row r reads dense row r.
    offsets = torch.tensor([0, 1, _TOTAL_L], device="xpu", dtype=index_dtype)

    dense_fast = _strided_dense(batch_stride, _FAST)
    _assert_matched_old_criteria(
        dense_fast, batch_stride_aligned=batch_stride == _ALIGNED_BATCH_STRIDE
    )

    fast, _ = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
        x_values.to(device="xpu", dtype=_FAST), [offsets], dense_fast
    )

    # Independent reference: every value is exact in float16.
    expected = x_values.to(device="xpu", dtype=_FAST) + dense_fast.reshape(
        _TOTAL_L, _E
    )
    assert torch.equal(fast, expected)  # nosec B101

    generic, _ = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
        x_values.to(device="xpu", dtype=_GENERIC),
        [offsets],
        _strided_dense(batch_stride, _GENERIC),
    )
    assert torch.equal(fast.to(_GENERIC), generic)  # nosec B101


@pytest.mark.parametrize("batch_stride", [_MISALIGNED_BATCH_STRIDE, _ALIGNED_BATCH_STRIDE])
@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
def test_dense_to_jagged_strided_dense_matches_generic_path(
    index_dtype, batch_stride
):
    """The same gather, reached through the operator that only copies dense."""
    offsets = torch.tensor([0, 1, _TOTAL_L], device="xpu", dtype=index_dtype)

    dense_fast = _strided_dense(batch_stride, _FAST)
    _assert_matched_old_criteria(
        dense_fast, batch_stride_aligned=batch_stride == _ALIGNED_BATCH_STRIDE
    )

    fast = torch.ops.fbgemm.dense_to_jagged_forward(
        dense_fast, [offsets], _TOTAL_L
    )

    # Each output row is its dense row, verbatim.
    assert torch.equal(fast, dense_fast.reshape(_TOTAL_L, _E))  # nosec B101

    generic = torch.ops.fbgemm.dense_to_jagged_forward(
        _strided_dense(batch_stride, _GENERIC), [offsets], _TOTAL_L
    )
    assert torch.equal(fast.to(_GENERIC), generic)  # nosec B101


@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
def test_add_broadcast_dense_batch_still_correct(index_dtype):
    """stride(0) == 0 satisfies the new predicate and stays on the fast path."""
    x_values = torch.arange(1, _TOTAL_L * _E + 1).view(_TOTAL_L, _E)
    offsets = torch.tensor([0, 1, _TOTAL_L], device="xpu", dtype=index_dtype)

    row = torch.arange(1, _E + 1, device="xpu", dtype=_FAST)
    dense = row.expand(_B, _MAX_L, _E)
    assert dense.stride(0) == 0  # nosec B101
    _assert_matched_old_criteria(dense, batch_stride_aligned=True)

    fast, _ = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
        x_values.to(device="xpu", dtype=_FAST), [offsets], dense
    )

    expected = x_values.to(device="xpu", dtype=_FAST) + row
    assert torch.equal(fast, expected)  # nosec B101


@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
def test_add_contiguous_dense_unaffected(index_dtype):
    """The narrower predicate must not reject the layout it was written for."""
    x_values = torch.arange(1, _TOTAL_L * _E + 1).view(_TOTAL_L, _E)
    offsets = torch.tensor([0, 1, _TOTAL_L], device="xpu", dtype=index_dtype)
    dense = torch.arange(
        1, _B * _MAX_L * _E + 1, device="xpu", dtype=_FAST
    ).view(_B, _MAX_L, _E)
    assert dense.stride(0) % 8 == 0  # nosec B101
    _assert_matched_old_criteria(dense, batch_stride_aligned=True)

    fast, _ = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
        x_values.to(device="xpu", dtype=_FAST), [offsets], dense
    )

    expected = x_values.to(device="xpu", dtype=_FAST) + dense.reshape(
        _TOTAL_L, _E
    )
    assert torch.equal(fast, expected)  # nosec B101
