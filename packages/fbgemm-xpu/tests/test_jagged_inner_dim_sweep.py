# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Inner-dimension sweep for the jagged launch geometry.

``check_shape_and_partition_`` sizes the inner (fastest-varying) work-group axis
from the two-element chunks the kernels actually step over,
``ceil(inner_dense_size / 2)`` capped at ``kThreadGroupSize``, rather than
CUDA's ``inner_dense_size >= kWarpSize / 2 ? kWarpSize : inner_dense_size``.

The extent is only ever the stride of a strided loop, and the work-item at
``inner_pairs % threads_x`` always exits exactly on the pair count, so the odd
tail element is written for any extent >= 1. Truncating and rounding-up
derivations are therefore both correct, and this sweep deliberately does not
claim to separate them - it was checked against a truncating build and passed.

What it does cover is that the geometry stays correct across the whole width
range, for a change that only ever intended to touch throughput: an extent that
collapsed to zero, saturated the cap at the wrong width, or transposed the two
work-group axes would surface here. Every output buffer is freshly allocated, so
a dropped write shows up as whatever the buffer happened to hold.

D = 16 is where CUDA's threshold snaps (15 lanes to 32, against a pair count of
7 to 8) and the width the geometry change was measured on. This is also the only
D-parametrized coverage in the jagged suite.
"""

import itertools

import fbgemm_xpu  # noqa: F401  - registers the fbgemm XPU operators
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="requires an XPU device"
)

# float16 is excluded on purpose: matches_opt() routes it to the vectorized
# gather kernel, which carries its own geometry and never reads this config.
_DTYPES = [torch.float32, torch.bfloat16]

# Both parities either side of CUDA's threshold (16), either side of where the
# pair count saturates a work-group (63/64), and the degenerate widths.
_INNER_DIMS = [0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128]

# Uneven lengths so both branches of the dense-output kernel run: batch row 0 is
# entirely padding, row 2 fills max_l, rows 1 and 3 are partial.
_LENGTHS = [0, 1, 3, 2]
_MAX_L = 3
_TOTAL_L = sum(_LENGTHS)
_PADDING = -7.0  # exact in every dtype under test


def _operands(inner_dim, dtype):
    """Jagged values, offsets and a dense operand, all exact in bfloat16."""
    torch.manual_seed(inner_dim)
    values = torch.randint(
        -128, 128, (_TOTAL_L, inner_dim), device="xpu"
    ).to(dtype)
    dense = torch.randint(
        -128, 128, (len(_LENGTHS), _MAX_L, inner_dim), device="xpu"
    ).to(dtype)
    offsets = torch.tensor(
        [0, *itertools.accumulate(_LENGTHS)], device="xpu", dtype=torch.long
    )
    return values, offsets, dense


def _gathered(dense):
    """The dense row each jagged row reads, in jagged order."""
    return torch.stack(
        [dense[i, j] for i, n in enumerate(_LENGTHS) for j in range(n)]
    )


@pytest.mark.parametrize("inner_dim", _INNER_DIMS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_jagged_to_padded_dense_forward_inner_dim(dtype, inner_dim):
    """The dense-output kernel writes every inner element, both parities."""
    values, offsets, _ = _operands(inner_dim, dtype)

    output = torch.ops.fbgemm.jagged_to_padded_dense_forward(
        values, [offsets], [_MAX_L], _PADDING
    )

    expected = torch.full(
        (len(_LENGTHS), _MAX_L, inner_dim), _PADDING, device="xpu", dtype=dtype
    )
    start = 0
    for i, n in enumerate(_LENGTHS):
        expected[i, :n] = values[start : start + n]
        start += n
    assert torch.equal(output, expected)  # nosec B101


@pytest.mark.parametrize("inner_dim", _INNER_DIMS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_dense_to_jagged_forward_inner_dim(dtype, inner_dim):
    """The jagged-output kernel gathers every inner element, both parities."""
    _, offsets, dense = _operands(inner_dim, dtype)

    output = torch.ops.fbgemm.dense_to_jagged_forward(dense, [offsets], _TOTAL_L)

    assert torch.equal(output, _gathered(dense))  # nosec B101


@pytest.mark.parametrize("inner_dim", _INNER_DIMS)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_add_jagged_output_inner_dim(dtype, inner_dim):
    """The two-dense-operand kernel reduces every inner element."""
    values, offsets, dense = _operands(inner_dim, dtype)

    output, _ = torch.ops.fbgemm.jagged_dense_elementwise_add_jagged_output(
        values, [offsets], dense
    )

    assert torch.equal(output, values + _gathered(dense))  # nosec B101
