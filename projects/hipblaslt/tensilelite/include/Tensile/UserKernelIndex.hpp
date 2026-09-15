/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

namespace TensileLite
{
    /**
     * Solution indices are partitioned so that the index alone records where a
     * solution came from: the shipped library occupies [0, 2^30) and kernels
     * registered at runtime occupy [2^30, 2^31).
     *
     * Indices are `int` throughout (ContractionSolution::index, and the key of
     * SolutionMap), so the user range stops short of INT32_MAX rather than
     * running to 2^32.
     *
     * A user index is only meaningful within the process that assigned it.
     * Registration re-mints indices from the payload on every load, so an
     * index must not be persisted and replayed in a later run.
     */
    constexpr int UserKernelIndexBase = 1 << 30;

    constexpr bool isUserKernelIndex(int index)
    {
        return index >= UserKernelIndexBase;
    }
} // namespace TensileLite
