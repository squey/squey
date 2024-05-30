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
#include <pvparallelview/PVHitGraphBuffer.h>

#include <iostream>
#include <cstring>

void test_left_shift(PVParallelView::PVHitGraphBuffer const& b, uint32_t n)
{
	PVParallelView::PVHitGraphBuffer bshift(b.nbits(), b.nblocks());
	PV_ASSERT_VALID(bshift.copy_from(b));

	bshift.shift_left(n);

	PV_ASSERT_VALID(memcmp(bshift.buffer(), b.buffer_block(n),
	                       (b.nblocks() - n) * b.size_block() * sizeof(uint32_t)) == 0);
}

void test_right_shift(PVParallelView::PVHitGraphBuffer const& b, uint32_t n)
{
	PVParallelView::PVHitGraphBuffer bshift(b.nbits(), b.nblocks());
	PV_ASSERT_VALID(bshift.copy_from(b));

	bshift.shift_right(n);

	PV_ASSERT_VALID(memcmp(b.buffer(), bshift.buffer_block(n),
	                       (b.nblocks() - n) * b.size_block() * sizeof(uint32_t)) == 0);
}

int main()
{
	PVParallelView::PVHitGraphBuffer buffer_11(11, 7);
	for (uint32_t i = 0; i < buffer_11.size_int(); i++) {
		buffer_11.buffer()[i] = i;
	}

	// Test shifts
	for (uint32_t i = 0; i < 7; i++) {
		test_left_shift(buffer_11, i);
		test_right_shift(buffer_11, i);
	}

	return 0;
}
