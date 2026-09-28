#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026 Intel Corporation. All Rights Reserved.
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

from common import CodeTemplate, args


def generate() -> None:
    Path(args.install_dir, "sycl_kernels").mkdir(parents=True, exist_ok=True)
    for bit_rate in (4, 8):
        CodeTemplate.load(
            "inference/embedding_forward_quantized_kernel_template.h"
        ).write(
            f"sycl_kernels/gen_embedding_forward_int{bit_rate}_nobag.h",
            bit_rate=bit_rate,
        )


if __name__ == "__main__":
    generate()
