# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shape validation for the FP16 jagged fast path.

Two invariants tie a jagged operand to its dense operand:

* ``dense.size(-1) == values.size(-1)`` - the embedding widths must agree;
* ``offsets[0].numel() == dense.size(0) + 1`` - one start per batch row plus a
  final sentinel.

Both lived only inside ``check_shape_and_partition_``, which the generic kernels
call because it doubles as their launch-geometry computation. The FP16 fast path
computes its own geometry (16x16 for the gather, ``kMaxThreads`` for the search),
so it never called that function and therefore ran with neither check.
``matches_opt`` did not cover the gap either - it is purely a layout predicate
over strides, 128-bit alignment, index range and local-memory size.

The result was that ``float16`` silently accepted inputs the generic kernel and
the CPU operator both reject, in four distinct ways:

* dense *narrower* than the output row: the gather's trip count is
  ``E = dense.size(-1)``, so only the first ``E`` columns of each output row were
  written and the remainder kept whatever ``at::empty_like`` returned - NaNs;
* dense *wider* than the output row: the same trip count drove a 128-bit vector
  store past the end of the row it was handed, into the next row and, on the last
  row, past the end of the allocation - an out-of-bounds device write;
* ``offsets`` *longer* than ``B + 1``: the search kernel stages exactly ``B + 1``
  entries, so the surplus intervals were silently dropped;
* ``offsets`` *shorter* than ``B + 1``: that same staging loop read past the end
  of the offsets tensor through an unchecked accessor - an out-of-bounds device
  read.

``check_jagged_dense_shape_`` now carries both checks and runs before path
selection, so the two paths agree. Each case below is a clean ``RuntimeError``,
asserted against the generic (``float32``) kernel to confirm the messages match
rather than merely that both fail.

Widths here are multiples of 8 and batches are contiguous so that the FP16 input
actually reaches the fast path; a width that broke 128-bit alignment would be
rerouted to the generic kernel and cover nothing.

.. note::
   FBGEMM's CUDA source has the same omission (``common.cuh:816``), so this is a
   deliberate deviation from upstream rather than a porting correction.

.. note::
   The checks run *after* the ``y.numel() == 0`` early return, matching where the
   generic path validates. An empty dense operand still short-circuits on both
   paths, so mismatched shapes are not reported for it.
"""

import fbgemm_xpu  # noqa: F401  - registers the fbgemm XPU operators
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="requires an XPU device"
)

# matches_opt() only selects the FP16 path for Half; float32 takes the generic
# kernel, which has carried both checks all along and is the parity reference.
_FAST = torch.float16
_GENERIC = torch.float32

_B = 2  # batch rows
_MAX_L = 1  # padded jagged length
_E = 8  # embedding dim; a multiple of 8 keeps rows 128-bit aligned
_TOTAL_L = 2  # jagged rows

_INDEX_DTYPES = [torch.int32, torch.int64]

_BAD_WIDTH = "inner_dense_size"
_BAD_OFFSETS = r"offsets\[0\].numel\(\) - 1"


def _assert_fast_path_shapes(dense):
    """Guard matches_opt()'s layout criteria for the operand we control.

    A future change to the shapes below must not quietly reroute these cases to
    the generic kernel and leave the regression uncovered. matches_opt() also
    inspects the internally allocated output, which is not reachable from here.
    """
    assert dense.dim() - 2 == 1, "fast path requires exactly one jagged dim"  # nosec B101
    assert dense.stride(-1) == 1  # nosec B101
    assert dense.stride(-2) % 8 == 0  # nosec B101
    assert dense.data_ptr() % 16 == 0  # nosec B101


def _add(x_values, offsets, dense):
    return torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
        x_values, [offsets], dense
    )


def _dense_to_jagged(dense, offsets, total_l=_TOTAL_L):
    return torch.ops.fbgemm.dense_to_jagged_forward(dense, [offsets], total_l)


# ---------------------------------------------------------------------------
# Inner width. Only reachable through the add operator: dense_to_jagged
# allocates values as {total_L, dense.size(-1)}, so its widths agree by
# construction.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [_FAST, _GENERIC])
def test_add_rejects_dense_narrower_than_values(dtype):
    """Before the fix, FP16 left the untouched trailing columns as NaNs."""
    x_values = torch.ones(_TOTAL_L, 2 * _E, device="xpu", dtype=dtype)
    offsets = torch.tensor([0, 1, _TOTAL_L], device="xpu", dtype=torch.int64)
    dense = torch.ones(_B, _MAX_L, _E, device="xpu", dtype=dtype)
    _assert_fast_path_shapes(dense)

    with pytest.raises(RuntimeError, match=_BAD_WIDTH):
        _add(x_values, offsets, dense)


@pytest.mark.parametrize("dtype", [_FAST, _GENERIC])
def test_add_rejects_dense_wider_than_values(dtype):
    """The reverse mismatch; before the fix FP16 stored past the output row."""
    x_values = torch.ones(_TOTAL_L, _E, device="xpu", dtype=dtype)
    offsets = torch.tensor([0, 1, _TOTAL_L], device="xpu", dtype=torch.int64)
    dense = torch.ones(_B, _MAX_L, 2 * _E, device="xpu", dtype=dtype)
    _assert_fast_path_shapes(dense)

    with pytest.raises(RuntimeError, match=_BAD_WIDTH):
        _add(x_values, offsets, dense)


# ---------------------------------------------------------------------------
# Offsets length. Reachable through both operators. The check is host-side and
# dtype-independent, but the search kernel it protects is templated on
# index_t and dispatched through AT_DISPATCH_INDEX_TYPES, so both index types
# are covered.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", [_FAST, _GENERIC])
def test_add_rejects_long_offsets(dtype, index_dtype):
    """An extra interval: before the fix FP16 silently ignored it."""
    x_values = torch.ones(_TOTAL_L, _E, device="xpu", dtype=dtype)
    offsets = torch.tensor([0, 1, 2, 2], device="xpu", dtype=index_dtype)
    dense = torch.ones(_B, _MAX_L, _E, device="xpu", dtype=dtype)
    assert offsets.numel() == _B + 2  # nosec B101
    _assert_fast_path_shapes(dense)

    with pytest.raises(RuntimeError, match=_BAD_OFFSETS):
        _add(x_values, offsets, dense)


@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", [_FAST, _GENERIC])
def test_add_rejects_short_offsets(dtype, index_dtype):
    """A missing sentinel: before the fix FP16 staged past the tensor's end."""
    x_values = torch.ones(_TOTAL_L, _E, device="xpu", dtype=dtype)
    offsets = torch.tensor([0, 1], device="xpu", dtype=index_dtype)
    dense = torch.ones(_B, _MAX_L, _E, device="xpu", dtype=dtype)
    assert offsets.numel() == _B  # nosec B101
    _assert_fast_path_shapes(dense)

    with pytest.raises(RuntimeError, match=_BAD_OFFSETS):
        _add(x_values, offsets, dense)


@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", [_FAST, _GENERIC])
def test_dense_to_jagged_rejects_long_offsets(dtype, index_dtype):
    """dense_to_jagged shares the helper, so it rejects the extra interval."""
    offsets = torch.tensor([0, 1, 2, 2], device="xpu", dtype=index_dtype)
    dense = torch.ones(_B, _MAX_L, _E, device="xpu", dtype=dtype)
    assert offsets.numel() == _B + 2  # nosec B101
    _assert_fast_path_shapes(dense)

    with pytest.raises(RuntimeError, match=_BAD_OFFSETS):
        _dense_to_jagged(dense, offsets)


@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
@pytest.mark.parametrize("dtype", [_FAST, _GENERIC])
def test_dense_to_jagged_rejects_short_offsets(dtype, index_dtype):
    """The out-of-bounds staging read is rejected at the API boundary."""
    offsets = torch.tensor([0, 1], device="xpu", dtype=index_dtype)
    dense = torch.ones(_B, _MAX_L, _E, device="xpu", dtype=dtype)
    assert offsets.numel() == _B  # nosec B101
    _assert_fast_path_shapes(dense)

    with pytest.raises(RuntimeError, match=_BAD_OFFSETS):
        _dense_to_jagged(dense, offsets)


# ---------------------------------------------------------------------------
# The hoist must not reject anything that used to work.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
def test_add_well_formed_still_matches_generic_path(index_dtype):
    """Well-formed shapes keep taking the fast path and agree with generic."""
    x_values = torch.arange(1, _TOTAL_L * _E + 1).view(_TOTAL_L, _E)
    offsets = torch.tensor([0, 1, _TOTAL_L], device="xpu", dtype=index_dtype)
    dense = torch.arange(1, _B * _MAX_L * _E + 1).view(_B, _MAX_L, _E)

    fast, _ = _add(
        x_values.to(device="xpu", dtype=_FAST),
        offsets,
        dense.to(device="xpu", dtype=_FAST),
    )
    generic, _ = _add(
        x_values.to(device="xpu", dtype=_GENERIC),
        offsets,
        dense.to(device="xpu", dtype=_GENERIC),
    )

    assert torch.equal(fast.to(torch.float32), generic)  # nosec B101


@pytest.mark.parametrize("index_dtype", _INDEX_DTYPES)
def test_dense_to_jagged_well_formed_still_matches_generic_path(index_dtype):
    """The same, for the operator whose widths agree by construction."""
    offsets = torch.tensor([0, 1, _TOTAL_L], device="xpu", dtype=index_dtype)
    dense = torch.arange(1, _B * _MAX_L * _E + 1).view(_B, _MAX_L, _E)

    fast = _dense_to_jagged(dense.to(device="xpu", dtype=_FAST), offsets)
    generic = _dense_to_jagged(dense.to(device="xpu", dtype=_GENERIC), offsets)

    assert torch.equal(fast.to(torch.float32), generic)  # nosec B101


@pytest.mark.parametrize("dtype", [_FAST, _GENERIC])
def test_empty_dense_still_short_circuits(dtype):
    """The early return precedes the checks, on both paths alike."""
    x_values = torch.ones(_TOTAL_L, _E, device="xpu", dtype=dtype)
    offsets = torch.tensor([0, 1, _TOTAL_L], device="xpu", dtype=torch.int64)
    # Mismatched inner width *and* an empty dense operand.
    dense = torch.ones(_B, 0, 2 * _E, device="xpu", dtype=dtype)
    assert dense.numel() == 0, "must reach the y.numel() == 0 early return"  # nosec B101

    output, _ = _add(x_values, offsets, dense)
    torch.xpu.synchronize()

    assert torch.equal(output, x_values)  # nosec B101


@pytest.mark.parametrize("dtype", [_FAST, _GENERIC])
def test_xpu_usable_after_rejection(dtype):
    """Rejection happens before any launch, so the device stays usable."""
    x_values = torch.ones(_TOTAL_L, _E, device="xpu", dtype=dtype)
    offsets = torch.tensor([0, 1, _TOTAL_L], device="xpu", dtype=torch.int64)
    dense = torch.ones(_B, _MAX_L, 2 * _E, device="xpu", dtype=dtype)

    with pytest.raises(RuntimeError, match=_BAD_WIDTH):
        _add(x_values, offsets, dense)

    output, _ = _add(
        x_values, offsets, torch.ones(_B, _MAX_L, _E, device="xpu", dtype=dtype)
    )
    torch.xpu.synchronize()

    assert torch.equal(output, x_values + 1)  # nosec B101
