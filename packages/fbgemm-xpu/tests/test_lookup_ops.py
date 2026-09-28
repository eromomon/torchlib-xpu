# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from dataclasses import dataclass

import fbgemm_xpu  # noqa: F401
import torch

POOLING_MODE_NONE = 2
SPARSE_TYPE_FP32 = 0
SPARSE_TYPE_FP16 = 1
PLACEMENT_DEVICE = 0
INFO_B_NUM_BITS = 26
INFO_B_MASK = (1 << INFO_B_NUM_BITS) - 1
LEARNING_RATE = 0.5
LARGE_GRID_TOTAL_B = 1 << 26

TableBatches = tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True)
class LookupLayout:
    tables: tuple[torch.Tensor, ...]
    batches: TableBatches
    dev_weights: torch.Tensor
    weights_offsets: torch.Tensor
    d_offsets: torch.Tensor
    hash_size_cumsum: torch.Tensor
    total_hash_size_bits: int
    indices: torch.Tensor
    offsets: torch.Tensor
    dimension: int
    total_dimension: int


@unittest.skipUnless(torch.xpu.is_available(), "XPU is required")
class XpuLookupOpsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.accelerator.current_accelerator(check_available=True)
        self.assertIsNotNone(self.device)
        self.assertEqual(self.device.type, "xpu")

    @staticmethod
    def cumulative(values: list[int]) -> list[int]:
        result = [0]
        for value in values:
            result.append(result[-1] + value)
        return result

    def make_layout(
        self,
        tables: tuple[torch.Tensor, ...],
        batches: TableBatches,
        index_dtype: torch.dtype,
        *,
        requires_grad: bool = False,
    ) -> LookupLayout:
        self.assertIn(len(tables), (1, 2))
        self.assertEqual(len(tables), len(batches))
        self.assertIn(index_dtype, (torch.int32, torch.int64))

        dimension = tables[0].shape[1]
        storage_dtype = tables[0].dtype
        batch_count = len(batches[0])
        self.assertIn(batch_count, (1, 2))

        for table, table_batches in zip(tables, batches):
            self.assertEqual(table.ndim, 2)
            self.assertGreaterEqual(table.shape[0], 2)
            self.assertEqual(table.shape[1], dimension)
            self.assertEqual(table.dtype, storage_dtype)
            self.assertEqual(len(table_batches), batch_count)
            self.assertIn(table.dtype, (torch.float16, torch.float32))
            for batch in table_batches:
                self.assertGreater(len(batch), 0)
                for index in batch:
                    self.assertGreaterEqual(index, 0)
                    self.assertLess(index, table.shape[0])

        row_counts = [table.shape[0] for table in tables]
        flattened_batches = [
            index
            for table_batches in batches
            for batch in table_batches
            for index in batch
        ]
        batch_lengths = [
            len(batch) for table_batches in batches for batch in table_batches
        ]
        dev_weights = torch.cat([table.flatten() for table in tables]).to(
            self.device
        )
        dev_weights.requires_grad_(requires_grad)

        return LookupLayout(
            tables=tables,
            batches=batches,
            dev_weights=dev_weights,
            weights_offsets=torch.tensor(
                self.cumulative([table.numel() for table in tables])[:-1],
                device=self.device,
                dtype=torch.int64,
            ),
            d_offsets=torch.tensor(
                self.cumulative([dimension] * len(tables)),
                device=self.device,
                dtype=torch.int32,
            ),
            hash_size_cumsum=torch.tensor(
                self.cumulative(row_counts),
                device=self.device,
                dtype=torch.int64,
            ),
            total_hash_size_bits=sum(row_counts).bit_length(),
            indices=torch.tensor(
                flattened_batches,
                device=self.device,
                dtype=index_dtype,
            ),
            offsets=torch.tensor(
                self.cumulative(batch_lengths),
                device=self.device,
                dtype=index_dtype,
            ),
            dimension=dimension,
            total_dimension=dimension * len(tables),
        )

    def make_mixed_dimension_layout(self) -> LookupLayout:
        tables = (
            torch.arange(8, dtype=torch.float32).reshape(2, 4),
            torch.arange(16, dtype=torch.float32).reshape(2, 8),
        )
        batches = (((1,),), ((0,),))

        return LookupLayout(
            tables=tables,
            batches=batches,
            dev_weights=torch.cat([table.flatten() for table in tables]).to(
                self.device
            ),
            weights_offsets=torch.tensor(
                [0, tables[0].numel()],
                device=self.device,
                dtype=torch.int64,
            ),
            d_offsets=torch.tensor(
                [0, 4, 12],
                device=self.device,
                dtype=torch.int32,
            ),
            hash_size_cumsum=torch.tensor(
                [0, 2, 4],
                device=self.device,
                dtype=torch.int64,
            ),
            total_hash_size_bits=3,
            indices=torch.tensor(
                [1, 0],
                device=self.device,
                dtype=torch.int32,
            ),
            offsets=torch.tensor(
                [0, 1, 2],
                device=self.device,
                dtype=torch.int32,
            ),
            dimension=8,
            total_dimension=12,
        )

    @staticmethod
    def output_torch_dtype(output_dtype: int) -> torch.dtype:
        if output_dtype == SPARSE_TYPE_FP32:
            return torch.float32
        if output_dtype == SPARSE_TYPE_FP16:
            return torch.float16
        raise ValueError(f"Unsupported output dtype: {output_dtype}")

    def reference_forward(
        self,
        layout: LookupLayout,
        output_dtype: int,
    ) -> torch.Tensor:
        outputs = []
        for table, table_batches in zip(layout.tables, layout.batches):
            flattened_indices = [
                index for batch in table_batches for index in batch
            ]
            outputs.append(
                table.index_select(
                    0,
                    torch.tensor(flattened_indices, dtype=torch.int64),
                )
            )
        return torch.cat(outputs).to(self.output_torch_dtype(output_dtype))

    @staticmethod
    def default_output_gradient(layout: LookupLayout) -> torch.Tensor:
        return torch.ones(
            (layout.indices.numel(), layout.dimension),
            dtype=torch.float32,
        )

    @staticmethod
    def reference_weight_gradients(
        layout: LookupLayout,
        output_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        gradients = []
        output_index = 0
        for table, table_batches in zip(layout.tables, layout.batches):
            table_gradient = torch.zeros_like(table)
            for batch in table_batches:
                for index in batch:
                    table_gradient[index] += output_gradient[output_index].to(
                        table.dtype
                    )
                    output_index += 1
            gradients.append(table_gradient)
        if output_index != output_gradient.shape[0]:
            raise ValueError("Output gradient does not match the lookup layout")
        return tuple(gradients)

    def reference_dense_gradient(
        self,
        layout: LookupLayout,
        output_gradient: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                gradient.flatten()
                for gradient in self.reference_weight_gradients(
                    layout,
                    output_gradient,
                )
            ]
        )

    def reference_rowwise_adagrad(
        self,
        layout: LookupLayout,
        output_gradient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        updated_tables = []
        momentum_tables = []
        for table, gradients in zip(
            layout.tables,
            self.reference_weight_gradients(layout, output_gradient),
        ):
            gradients_float = gradients.float()
            momentum = gradients_float.square().mean(dim=1)
            multiplier = torch.zeros_like(momentum)
            updated_rows = momentum > 0
            multiplier[updated_rows] = (
                LEARNING_RATE / momentum[updated_rows].sqrt()
            )
            updated_tables.append(
                (table.float() - multiplier[:, None] * gradients_float)
                .to(table.dtype)
                .flatten()
            )
            momentum_tables.append(momentum)
        return torch.cat(updated_tables), torch.cat(momentum_tables)

    def dense_lookup(
        self,
        layout: LookupLayout,
        output_dtype: int,
    ) -> torch.Tensor:
        return torch.ops.fbgemm.dense_embedding_codegen_lookup_function(
            layout.dev_weights,
            layout.weights_offsets,
            layout.d_offsets,
            layout.total_dimension,
            layout.dimension,
            layout.hash_size_cumsum,
            layout.total_hash_size_bits,
            layout.indices,
            layout.offsets,
            POOLING_MODE_NONE,
            None,
            None,
            output_dtype,
            None,
            None,
            None,
            -1,
            -1,
            -1,
            False,
        )

    def assert_dense_large_grid_forward(self, dimension: int) -> None:
        dev_weights = torch.arange(
            1,
            dimension + 1,
            device=self.device,
            dtype=torch.float32,
        )
        offsets = torch.zeros(
            LARGE_GRID_TOTAL_B + 1,
            device=self.device,
            dtype=torch.int32,
        )
        offsets[-1] = 1

        output = torch.ops.fbgemm.dense_embedding_codegen_lookup_function(
            dev_weights,
            torch.tensor([0], device=self.device, dtype=torch.int64),
            torch.tensor([0, dimension], device=self.device, dtype=torch.int32),
            dimension,
            dimension,
            torch.tensor([0, 1], device=self.device, dtype=torch.int64),
            1,
            torch.tensor([0], device=self.device, dtype=torch.int32),
            offsets,
            POOLING_MODE_NONE,
            None,
            None,
            SPARSE_TYPE_FP32,
            None,
            None,
            None,
            -1,
            -1,
            -1,
            False,
        )

        self.assertEqual(output.shape, (1, dimension))
        torch.testing.assert_close(output.cpu(), dev_weights.cpu().unsqueeze(0))

    def split_lookup(
        self,
        layout: LookupLayout,
        output_dtype: int = SPARSE_TYPE_FP32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        table_count = len(layout.tables)
        total_rows = sum(table.shape[0] for table in layout.tables)
        momentum1_dev = torch.zeros(
            total_rows,
            device=self.device,
            dtype=torch.float32,
        )

        output = torch.ops.fbgemm.split_embedding_codegen_lookup_rowwise_adagrad_function_pt2(
            torch.zeros(
                (),
                device=self.device,
                dtype=torch.float32,
                requires_grad=True,
            ),
            [
                layout.dev_weights,
                torch.empty(
                    0,
                    device=self.device,
                    dtype=layout.dev_weights.dtype,
                ),
                torch.full(
                    (table_count,),
                    PLACEMENT_DEVICE,
                    device=self.device,
                    dtype=torch.int32,
                ),
                layout.weights_offsets,
                torch.empty(
                    (0, layout.dimension),
                    device=self.device,
                    dtype=layout.dev_weights.dtype,
                ),
            ],
            layout.d_offsets,
            layout.total_dimension,
            layout.dimension,
            layout.hash_size_cumsum,
            layout.total_hash_size_bits,
            layout.indices,
            layout.offsets,
            POOLING_MODE_NONE,
            None,
            None,
            output_dtype,
            [
                None,
                None,
                None,
                torch.empty(0, device=self.device, dtype=torch.int32),
                torch.empty(0, device=self.device, dtype=torch.int32),
                None,
                None,
            ],
            [0, INFO_B_NUM_BITS, INFO_B_MASK],
            [0.0, 0.0],
            [False, False, True, False, False, False, False],
            [
                momentum1_dev,
                torch.empty(0, device=self.device, dtype=torch.float32),
                torch.full(
                    (table_count,),
                    PLACEMENT_DEVICE,
                    device=self.device,
                    dtype=torch.int32,
                ),
                torch.tensor(
                    self.cumulative(
                        [table.shape[0] for table in layout.tables]
                    )[:-1],
                    device=self.device,
                    dtype=torch.int64,
                ),
            ],
            torch.tensor(LEARNING_RATE, dtype=torch.float32),
            [0],
            [0.0, 0.0, 0.0],
            -1,
            -1,
            -1,
            None,
        )
        return output, momentum1_dev

    def test_dense_lookup_forward(self) -> None:
        layout = self.make_layout(
            tables=(
                torch.tensor(
                    [[0, 1, 2, 3], [10, 11, 12, 13]],
                    dtype=torch.float32,
                ),
            ),
            batches=(((1,),),),
            index_dtype=torch.int64,
        )

        output = self.dense_lookup(layout, output_dtype=SPARSE_TYPE_FP16)

        self.assertEqual(output.shape, (1, 4))
        self.assertEqual(output.dtype, torch.float16)
        torch.testing.assert_close(
            output.cpu(),
            self.reference_forward(layout, SPARSE_TYPE_FP16),
        )

    def test_dense_small_forward_large_grid(self) -> None:
        self.assert_dense_large_grid_forward(dimension=4)

    def test_dense_general_forward_large_grid(self) -> None:
        self.assert_dense_large_grid_forward(dimension=36)

    def test_dense_mixed_dimensions_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "uniform embedding dimensions"):
            self.dense_lookup(
                self.make_mixed_dimension_layout(),
                output_dtype=SPARSE_TYPE_FP32,
            )

    def test_split_mixed_dimensions_are_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "uniform embedding dimensions"):
            self.split_lookup(self.make_mixed_dimension_layout())

    def test_dense_lookup_backward(self) -> None:
        layout = self.make_layout(
            tables=(
                torch.tensor(
                    [[0, 1, 2, 3], [10, 11, 12, 13]],
                    dtype=torch.float32,
                ),
            ),
            batches=(((1,),),),
            index_dtype=torch.int64,
            requires_grad=True,
        )

        self.dense_lookup(layout, output_dtype=SPARSE_TYPE_FP32).sum().backward()

        gradient = layout.dev_weights.grad
        if gradient is None:
            self.fail("Dense lookup backward did not produce a weight gradient")
        torch.testing.assert_close(
            gradient.cpu(),
            self.reference_dense_gradient(
                layout,
                self.default_output_gradient(layout),
            ),
        )

    def test_split_rowwise_adagrad_forward_and_update(self) -> None:
        layout = self.make_layout(
            tables=(
                torch.tensor(
                    [[1, 2, 3, 4], [10, 11, 12, 13]],
                    dtype=torch.float32,
                ),
            ),
            batches=(((0,),),),
            index_dtype=torch.int64,
        )
        output_gradient = self.default_output_gradient(layout)
        expected_weights, expected_momentum = self.reference_rowwise_adagrad(
            layout,
            output_gradient,
        )

        output, momentum = self.split_lookup(layout)

        torch.testing.assert_close(
            output.cpu(),
            self.reference_forward(layout, SPARSE_TYPE_FP32),
        )

        output.sum().backward()

        torch.testing.assert_close(
            layout.dev_weights.cpu(),
            expected_weights,
        )
        torch.testing.assert_close(
            momentum.cpu(),
            expected_momentum,
        )

    def test_dense_general_forward_and_cta_backward_with_fp16_weights(
        self,
    ) -> None:
        dimension = 36
        row = torch.arange(1, dimension + 1, dtype=torch.float16)
        layout = self.make_layout(
            tables=(torch.stack((torch.zeros_like(row), row)),),
            batches=(((1,) * 32,),),
            index_dtype=torch.int64,
            requires_grad=True,
        )

        output = self.dense_lookup(layout, output_dtype=SPARSE_TYPE_FP32)

        torch.testing.assert_close(
            output.cpu(),
            self.reference_forward(layout, SPARSE_TYPE_FP32),
            rtol=0,
            atol=0,
        )

        output.sum().backward()

        gradient = layout.dev_weights.grad
        if gradient is None:
            self.fail("Dense CTA backward did not produce a weight gradient")
        torch.testing.assert_close(
            gradient.cpu(),
            self.reference_dense_gradient(
                layout,
                self.default_output_gradient(layout),
            ),
            rtol=0,
            atol=0,
        )

    def test_dense_multi_cta_backward_completion(self) -> None:
        dimension = 36
        repeat_count = 1025
        row = torch.arange(1, dimension + 1, dtype=torch.float32)
        layout = self.make_layout(
            tables=(torch.stack((torch.zeros_like(row), row)),),
            batches=(((1,) * repeat_count,),),
            index_dtype=torch.int64,
            requires_grad=True,
        )
        output_gradient = (
            torch.arange(repeat_count * dimension, dtype=torch.float32)
            .remainder(5)
            .reshape(repeat_count, dimension)
        )

        output = self.dense_lookup(layout, output_dtype=SPARSE_TYPE_FP32)
        output.backward(output_gradient.to(self.device))

        gradient = layout.dev_weights.grad
        if gradient is None:
            self.fail("Dense multi-CTA backward did not produce a weight gradient")
        torch.testing.assert_close(
            gradient.cpu(),
            self.reference_dense_gradient(layout, output_gradient),
            rtol=0,
            atol=0,
        )

    def test_split_general_forward_and_cta_update_with_fp16_weights(
        self,
    ) -> None:
        dimension = 36
        row = torch.arange(1, dimension + 1, dtype=torch.float16)
        layout = self.make_layout(
            tables=(torch.stack((torch.zeros_like(row), row)),),
            batches=(((1,) * 32,),),
            index_dtype=torch.int64,
        )
        output_gradient = self.default_output_gradient(layout)
        expected_weights, expected_momentum = self.reference_rowwise_adagrad(
            layout,
            output_gradient,
        )

        output, momentum = self.split_lookup(layout)

        torch.testing.assert_close(
            output.cpu(),
            self.reference_forward(layout, SPARSE_TYPE_FP32),
            rtol=0,
            atol=0,
        )

        output.sum().backward()

        torch.testing.assert_close(
            layout.dev_weights.cpu(),
            expected_weights,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            momentum.cpu(),
            expected_momentum,
            rtol=0,
            atol=0,
        )

    def test_dense_multi_table_int32_forward_and_backward(self) -> None:
        dimension = 32
        layout = self.make_layout(
            tables=(
                torch.arange(4 * dimension, dtype=torch.float32).reshape(
                    4, dimension
                ),
                torch.arange(
                    1000,
                    1000 + 4 * dimension,
                    dtype=torch.float32,
                ).reshape(4, dimension),
            ),
            batches=(
                ((0, 1), (1, 2)),
                ((2, 2), (0, 1)),
            ),
            index_dtype=torch.int32,
            requires_grad=True,
        )

        output = self.dense_lookup(layout, output_dtype=SPARSE_TYPE_FP32)
        output_gradient = torch.arange(
            1,
            layout.indices.numel() + 1,
            dtype=torch.float32,
        )[:, None].expand(-1, layout.dimension)

        torch.testing.assert_close(
            output.cpu(),
            self.reference_forward(layout, SPARSE_TYPE_FP32),
            rtol=0,
            atol=0,
        )

        (output * output_gradient.to(self.device)).sum().backward()

        gradient = layout.dev_weights.grad
        if gradient is None:
            self.fail("Dense multi-table backward did not produce a gradient")
        torch.testing.assert_close(
            gradient.cpu(),
            self.reference_dense_gradient(layout, output_gradient),
            rtol=0,
            atol=0,
        )

    def test_split_multi_table_int64_forward_and_update(self) -> None:
        dimension = 32
        layout = self.make_layout(
            tables=(
                torch.arange(
                    1,
                    1 + 4 * dimension,
                    dtype=torch.float32,
                ).reshape(4, dimension),
                torch.arange(
                    1001,
                    1001 + 4 * dimension,
                    dtype=torch.float32,
                ).reshape(4, dimension),
            ),
            batches=(
                ((0, 0), (1, 2)),
                ((3, 1), (1, 2)),
            ),
            index_dtype=torch.int64,
        )
        output, momentum = self.split_lookup(layout)
        output_gradient = torch.arange(
            1,
            layout.indices.numel() + 1,
            dtype=torch.float32,
        )[:, None].expand(-1, layout.dimension)
        expected_weights, expected_momentum = self.reference_rowwise_adagrad(
            layout,
            output_gradient,
        )

        torch.testing.assert_close(
            output.cpu(),
            self.reference_forward(layout, SPARSE_TYPE_FP32),
            rtol=0,
            atol=0,
        )

        (output * output_gradient.to(self.device)).sum().backward()

        torch.testing.assert_close(
            layout.dev_weights.cpu(),
            expected_weights,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            momentum.cpu(),
            expected_momentum,
            rtol=0,
            atol=0,
        )
