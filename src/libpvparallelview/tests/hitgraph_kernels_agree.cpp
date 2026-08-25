//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/core/PVCpuFeatures.h>
#include <pvparallelview/PVHitGraphDataOMP.h>
#include <pvparallelview/PVHitGraphBuffer.h>
#include <squey/PVSelection.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

/**
 * The hit graph counting kernels are compiled twice, once for the baseline instruction
 * set and once with -mavx2, and picked at runtime. Nothing else would catch the two
 * drifting apart, so fill a buffer with each and compare them bit for bit.
 */

static constexpr uint32_t NBITS = 10;

static void check_agreement(size_t nrows, int nblocks, int zoom, double alpha)
{
	std::vector<uint32_t> scaled(nrows);
	for (size_t i = 0; i < nrows; ++i) {
		// A multiplicative hash spreads the rows over every bucket, so both kernels
		// exercise their whole range rather than a single block.
		scaled[i] = uint32_t(i * 2654435761u);
	}

	PVParallelView::PVHitGraphDataOMP data(NBITS, uint32_t(nblocks));
	PVParallelView::PVHitGraphBuffer buf_baseline(NBITS, uint32_t(nblocks));
	PVParallelView::PVHitGraphBuffer buf_avx2(NBITS, uint32_t(nblocks));
	buf_baseline.set_zero();
	buf_avx2.set_zero();

	PVParallelView::PVHitGraphDataInterface::ProcessParams params(
	    scaled.data(), PVRow(nrows), 0u, zoom, int(NBITS), alpha, 0, nblocks);

	data.process_all_with(params, buf_baseline, false);
	data.process_all_with(params, buf_avx2, true);

	// process_sel() runs a third kernel, in its own OpenMP region : cover it too.
	Squey::PVSelection sel(nrows);
	sel.select_none();
	for (size_t i = 0; i < nrows; i += 3) {
		sel.set_bit_fast(i);
	}
	PVParallelView::PVHitGraphBuffer sel_baseline(NBITS, uint32_t(nblocks));
	PVParallelView::PVHitGraphBuffer sel_avx2(NBITS, uint32_t(nblocks));
	sel_baseline.set_zero();
	sel_avx2.set_zero();
	data.process_sel_with(params, sel_baseline, sel, false);
	data.process_sel_with(params, sel_avx2, sel, true);
	PV_ASSERT_VALID(memcmp(sel_baseline.buffer(), sel_avx2.buffer(),
	                       sel_baseline.size_int() * sizeof(uint32_t)) == 0,
	                "selection path, nrows", nrows, "nblocks", nblocks);

	// Guard against the comparison passing on two buffers both left empty, which would
	// make this test green whatever the kernels do.
	uint64_t counted = 0;
	for (size_t i = 0; i < buf_baseline.size_int(); ++i) {
		counted += buf_baseline.buffer()[i];
	}
	PV_ASSERT_VALID(counted > 0, "nrows", nrows, "counted", counted);

	PV_ASSERT_VALID(memcmp(buf_baseline.buffer(), buf_avx2.buffer(),
	                       buf_baseline.size_int() * sizeof(uint32_t)) == 0,
	                "nrows", nrows, "nblocks", nblocks, "zoom", zoom, "alpha", alpha);
}

int main()
{
	if (not PVCore::has_avx2()) {
		std::cout << "CPU without AVX2, both paths would run the same kernels" << std::endl;
		return 0;
	}

	// A row count that is not a multiple of 4 leaves a tail for the scalar epilogue.
	for (size_t nrows : {size_t(1000), size_t(65537), size_t(1000003)}) {
		check_agreement(nrows, 1, 0, 1.0); // the single block kernel
		check_agreement(nrows, 4, 0, 1.0); // the N blocks one
		check_agreement(nrows, 4, 3, 0.5); // zoomed, alpha below 1
	}

	return 0;
}
