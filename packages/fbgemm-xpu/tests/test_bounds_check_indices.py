# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest

import fbgemm_xpu  # noqa: F401
import torch

FATAL = 0
WARNING = 1
IGNORE = 2


def run_bounds_check(
    rows_per_table: torch.Tensor,
    indices: torch.Tensor,
    offsets: torch.Tensor,
    mode: int,
    warning: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    B_offsets: torch.Tensor | None = None,
    max_B: int = -1,
) -> None:
    torch.ops.fbgemm.bounds_check_indices(
        rows_per_table,
        indices,
        offsets,
        mode,
        warning,
        weights,
        B_offsets,
        max_B,
        None,
        -1,
        -1,
        1,
        False,
    )


class BoundsCheckIndicesXpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.xpu.is_available():
            raise RuntimeError("XPU validation requires a real device; no skip/fallback")

    def setUp(self) -> None:
        self.device = torch.accelerator.current_accelerator(
            check_available=True
        )
        self.assertIsNotNone(self.device)
        self.assertEqual(self.device.type, "xpu")

    def assert_cpu_xpu_parity(
        self,
        rows_per_table: torch.Tensor,
        indices: torch.Tensor,
        offsets: torch.Tensor,
        mode: int,
        *,
        weights: torch.Tensor | None = None,
        B_offsets: torch.Tensor | None = None,
        max_B: int = -1,
        compare_warning: bool = True,
    ) -> None:
        cpu_indices = indices.clone()
        cpu_offsets = offsets.clone()
        cpu_warning = torch.tensor([99], dtype=torch.int64)
        cpu_weights = weights.clone() if weights is not None else None
        cpu_B_offsets = B_offsets.clone() if B_offsets is not None else None

        xpu_indices = indices.to(self.device)
        xpu_offsets = offsets.to(self.device)
        xpu_warning = torch.tensor([99], device=self.device, dtype=torch.int64)
        xpu_weights = weights.to(self.device) if weights is not None else None
        xpu_B_offsets = (
            B_offsets.to(self.device) if B_offsets is not None else None
        )

        run_bounds_check(
            rows_per_table,
            cpu_indices,
            cpu_offsets,
            mode,
            cpu_warning,
            weights=cpu_weights,
            B_offsets=cpu_B_offsets,
            max_B=max_B,
        )
        run_bounds_check(
            rows_per_table.to(self.device),
            xpu_indices,
            xpu_offsets,
            mode,
            xpu_warning,
            weights=xpu_weights,
            B_offsets=xpu_B_offsets,
            max_B=max_B,
        )
        torch.xpu.synchronize()

        torch.testing.assert_close(xpu_indices.cpu(), cpu_indices)
        torch.testing.assert_close(xpu_offsets.cpu(), cpu_offsets)
        if compare_warning:
            torch.testing.assert_close(xpu_warning.cpu(), cpu_warning)

    def test_registration(self) -> None:
        self.assertTrue(
            torch._C._dispatch_has_kernel_for_dispatch_key(
                "fbgemm::bounds_check_indices", "XPU"
            )
        )

    def test_out_of_range_and_negative_indices_match_cpu(self) -> None:
        rows_per_table = torch.tensor([3, 2], dtype=torch.int64)
        offsets_values = [0, 3, 5, 7, 9]
        indices_values = [0, 3, -2, -1, 2, 0, 2, -3, 1]

        for index_dtype in (torch.int32, torch.int64):
            for mode in (WARNING, IGNORE):
                with self.subTest(index_dtype=index_dtype, mode=mode):
                    self.assert_cpu_xpu_parity(
                        rows_per_table,
                        torch.tensor(indices_values, dtype=index_dtype),
                        torch.tensor(offsets_values, dtype=index_dtype),
                        mode,
                    )

    def test_pruned_index_is_preserved(self) -> None:
        for mode in (WARNING, IGNORE):
            with self.subTest(mode=mode):
                indices = torch.tensor(
                    [-1, 0, 1], device=self.device, dtype=torch.int64
                )
                warning = torch.tensor(
                    [17], device=self.device, dtype=torch.int64
                )
                run_bounds_check(
                    torch.tensor([2], device=self.device, dtype=torch.int64),
                    indices,
                    torch.tensor(
                        [0, 3], device=self.device, dtype=torch.int64
                    ),
                    mode,
                    warning,
                )
                torch.xpu.synchronize()

                torch.testing.assert_close(
                    indices.cpu(), torch.tensor([-1, 0, 1])
                )
                expected_warning = 0 if mode == WARNING else 17
                self.assertEqual(warning.item(), expected_warning)

    def test_invalid_offsets_match_cpu(self) -> None:
        rows_per_table = torch.tensor([4], dtype=torch.int64)
        indices = torch.tensor([0, 1, 2], dtype=torch.int64)
        offsets = torch.tensor([-2, 1, 3], dtype=torch.int64)

        for mode in (WARNING, IGNORE):
            with self.subTest(mode=mode):
                self.assert_cpu_xpu_parity(
                    rows_per_table,
                    indices,
                    offsets,
                    mode,
                )

    def test_overlapping_invalid_offsets_match_cpu(self) -> None:
        rows_per_table = torch.tensor([4], dtype=torch.int64)
        indices = torch.tensor([9, 9, 1], dtype=torch.int64)
        offsets = torch.tensor([2, 1, 3], dtype=torch.int64)

        for mode in (WARNING, IGNORE):
            with self.subTest(mode=mode):
                self.assert_cpu_xpu_parity(
                    rows_per_table,
                    indices,
                    offsets,
                    mode,
                )

    def test_short_terminal_offset_still_checks_trailing_index(self) -> None:
        rows_per_table = torch.tensor([4], dtype=torch.int64)
        indices = torch.tensor([0, 1, 7], dtype=torch.int64)
        offsets = torch.tensor([0, 2], dtype=torch.int64)

        for mode in (WARNING, IGNORE):
            with self.subTest(mode=mode):
                self.assert_cpu_xpu_parity(
                    rows_per_table,
                    indices,
                    offsets,
                    mode,
                )

    def test_empty_warning_input_matches_cpu(self) -> None:
        self.assert_cpu_xpu_parity(
            torch.tensor([4], dtype=torch.int64),
            torch.empty(0, dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
            WARNING,
        )

    def test_zero_bag_terminal_offset_matches_cpu(self) -> None:
        rows_per_table = torch.tensor([4], dtype=torch.int64)

        for index_dtype in (torch.int32, torch.int64):
            indices = torch.tensor([7], dtype=index_dtype)
            offsets = torch.tensor([0], dtype=index_dtype)

            for mode in (WARNING, IGNORE):
                with self.subTest(index_dtype=index_dtype, mode=mode):
                    self.assert_cpu_xpu_parity(
                        rows_per_table,
                        indices,
                        offsets,
                        mode,
                    )

            for device in (torch.device("cpu"), self.device):
                with self.subTest(
                    index_dtype=index_dtype,
                    mode=FATAL,
                    device=device.type,
                ):
                    with self.assertRaises(RuntimeError):
                        run_bounds_check(
                            rows_per_table.to(device),
                            indices.to(device),
                            offsets.to(device),
                            FATAL,
                            torch.zeros(
                                1, device=device, dtype=torch.int64
                            ),
                        )

    def test_weighted_input_matches_cpu(self) -> None:
        self.assert_cpu_xpu_parity(
            torch.tensor([3], dtype=torch.int64),
            torch.tensor([0, 4, -2], dtype=torch.int64),
            torch.tensor([0, 3], dtype=torch.int64),
            WARNING,
            weights=torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32),
        )

    def test_weights_device_contract(self) -> None:
        for weights_device in (torch.device("cpu"), self.device):
            with self.subTest(weights_device=weights_device.type):
                indices = torch.tensor(
                    [0], device=self.device, dtype=torch.int64
                )
                run_bounds_check(
                    torch.tensor(
                        [2], device=self.device, dtype=torch.int64
                    ),
                    indices,
                    torch.tensor(
                        [0, 1], device=self.device, dtype=torch.int64
                    ),
                    WARNING,
                    torch.zeros(
                        1, device=self.device, dtype=torch.int64
                    ),
                    weights=torch.empty(0, device=weights_device),
                )
                torch.xpu.synchronize()
                self.assertEqual(indices.item(), 0)

        with self.assertRaisesRegex(
            RuntimeError, "must be empty or a XPU tensor"
        ):
            run_bounds_check(
                torch.tensor([2], device=self.device, dtype=torch.int64),
                torch.tensor([0], device=self.device, dtype=torch.int64),
                torch.tensor([0, 1], device=self.device, dtype=torch.int64),
                WARNING,
                torch.zeros(1, device=self.device, dtype=torch.int64),
                weights=torch.ones(1),
            )

    def test_variable_batch_matches_cpu(self) -> None:
        rows_per_table = torch.tensor([3, 2], dtype=torch.int64)
        B_offsets = torch.tensor([0, 1, 3], dtype=torch.int32)
        offsets_values = [0, 2, 3, 5]
        indices_values = [0, 4, 1, -2, 0]

        for index_dtype in (torch.int32, torch.int64):
            for mode in (WARNING, IGNORE):
                with self.subTest(index_dtype=index_dtype, mode=mode):
                    self.assert_cpu_xpu_parity(
                        rows_per_table,
                        torch.tensor(indices_values, dtype=index_dtype),
                        torch.tensor(offsets_values, dtype=index_dtype),
                        mode,
                        B_offsets=B_offsets,
                        max_B=2,
                    )

    def test_fatal_rejects_each_invalid_index_kind(self) -> None:
        for invalid_index in (4, -2):
            for device in (torch.device("cpu"), self.device):
                with self.subTest(
                    device=device.type, invalid_index=invalid_index
                ):
                    with self.assertRaises(RuntimeError):
                        run_bounds_check(
                            torch.tensor(
                                [4], device=device, dtype=torch.int64
                            ),
                            torch.tensor(
                                [invalid_index],
                                device=device,
                                dtype=torch.int64,
                            ),
                            torch.tensor(
                                [0, 1], device=device, dtype=torch.int64
                            ),
                            FATAL,
                            torch.zeros(1, device=device, dtype=torch.int64),
                        )

        # A FATAL failure must not poison the process or the XPU context.
        indices = torch.tensor([1], device=self.device, dtype=torch.int64)
        run_bounds_check(
            torch.tensor([4], device=self.device, dtype=torch.int64),
            indices,
            torch.tensor([0, 1], device=self.device, dtype=torch.int64),
            WARNING,
            torch.zeros(1, device=self.device, dtype=torch.int64),
        )
        torch.xpu.synchronize()
        self.assertEqual(indices.item(), 1)

    def test_fatal_rejects_invalid_offsets(self) -> None:
        for device in (torch.device("cpu"), self.device):
            with self.subTest(device=device.type):
                with self.assertRaises(RuntimeError):
                    run_bounds_check(
                        torch.tensor([4], device=device, dtype=torch.int64),
                        torch.tensor([0], device=device, dtype=torch.int64),
                        torch.tensor([0, 2], device=device, dtype=torch.int64),
                        FATAL,
                        torch.zeros(1, device=device, dtype=torch.int64),
                    )

    def test_v2_and_none_are_rejected(self) -> None:
        rows_per_table = torch.tensor(
            [2], device=self.device, dtype=torch.int64
        )
        indices = torch.tensor([0], device=self.device, dtype=torch.int64)
        offsets = torch.tensor([0, 1], device=self.device, dtype=torch.int64)
        warning = torch.zeros(1, device=self.device, dtype=torch.int64)

        with self.assertRaisesRegex(RuntimeError, "bounds_check_version=1"):
            torch.ops.fbgemm.bounds_check_indices(
                rows_per_table,
                indices,
                offsets,
                WARNING,
                warning,
                None,
                None,
                -1,
                None,
                -1,
                -1,
                2,
                False,
            )
        with self.assertRaisesRegex(RuntimeError, "is not supported"):
            run_bounds_check(
                rows_per_table,
                indices,
                offsets,
                3,
                warning,
            )


if __name__ == "__main__":
    unittest.main()
