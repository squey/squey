#include <pvkernel/core/picviz_assert.h>
#include <pvparallelview/PVHitGraphBuffer.h>

#include <iostream>
#include <string.h>

void print_buffer(PVParallelView::PVHitGraphBuffer const& b)
{
	for (size_t i = 0; i < b.size_int(); i++) {
		std::cout << i << "\t = " << b.buffer()[i] << " / " << b.zoomed_buffer()[i] << std::endl;
	}
}

void test_left_shift(PVParallelView::PVHitGraphBuffer const& b, uint32_t n)
{
	PVParallelView::PVHitGraphBuffer bshift(b.nbits(), b.nblocks());
	PV_ASSERT_VALID(bshift.copy_from(b));

	bshift.process_zoom_reduction(0.5f);
	bshift.shift_left(n, 0.5f);

	PV_ASSERT_VALID(memcmp(bshift.buffer(), b.buffer_block(n), (b.nblocks()-n)*b.size_block()*sizeof(uint32_t)) == 0);
}

void test_right_shift(PVParallelView::PVHitGraphBuffer const& b, uint32_t n)
{
	PVParallelView::PVHitGraphBuffer bshift(b.nbits(), b.nblocks());
	PV_ASSERT_VALID(bshift.copy_from(b));

	bshift.process_zoom_reduction(0.5f);
	bshift.shift_right(n, 0.5f);

	PV_ASSERT_VALID(memcmp(b.buffer(), bshift.buffer_block(n), (b.nblocks()-n)*b.size_block()*sizeof(uint32_t)) == 0);
}

int main(int argc, char** argv)
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
