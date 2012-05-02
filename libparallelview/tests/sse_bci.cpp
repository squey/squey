#include <stdio.h>
#include <stdlib.h>
#include <iostream>
using namespace std;

#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/general.h>
#include <pvbase/types.h>

#include <pvparallelview/PVHSVColor.h>
#include <pvparallelview/PVBCICode.h>


#define NBITS_INDEX 10
#define NBUCKET 4

void sse_bci(unsigned int* bcodes, unsigned int* indexes, PVParallelView::PVHSVColor* colors, PVParallelView::PVBCICode* bci_codes)
{
	__m128i sse_lr;
	sse_lr = _mm_insert_epi32(sse_lr, bcodes[0], 0);
	sse_lr = _mm_insert_epi32(sse_lr, bcodes[1], 1);
	sse_lr = _mm_insert_epi32(sse_lr, bcodes[2], 2);
	sse_lr = _mm_insert_epi32(sse_lr, bcodes[3], 3);

	//  +------------+------------++------------+------------+
	//  |        lr3 |        lr2 ||        lr1 |        lr0 | (sse_lr)
	//  +------------+------------++------------+------------+

	__m128i sse_color ;
	sse_color = _mm_insert_epi32(sse_color, colors[0].h(), 0);
	sse_color = _mm_insert_epi32(sse_color, colors[1].h(), 1);
	sse_color = _mm_insert_epi32(sse_color, colors[2].h(), 2);
	sse_color = _mm_insert_epi32(sse_color, colors[3].h(), 3);
	sse_color = _mm_slli_epi32(sse_color, NBITS_INDEX*2);

	//  +------------+------------++------------+------------+
	//  |color3 << 20|color2 << 20||color1 << 20|color0 << 20| (sse_color)
	//  +------------+------------++------------+------------+

	__m128i sse_lrcolor;
	sse_lrcolor = _mm_or_si128(sse_color, sse_lr);

	//  +------------+------------++------------+------------+
	//  |   lrcolor3 |   lrcolor2 ||   lrcolor1 |   lrcolor0 | (sse_lrcolor)
	//  +------------+------------++------------+------------+

	__m128i sse_index;
	sse_index = _mm_insert_epi32(sse_index, indexes[0], 0);
	sse_index = _mm_insert_epi32(sse_index, indexes[1], 1);
	sse_index = _mm_insert_epi32(sse_index, indexes[2], 2);
	sse_index = _mm_insert_epi32(sse_index, indexes[3], 3);

	//  +------------+------------++------------+------------+
	//  | index (r3) | index (r2) || index (r1) | index (r0) | (sse_index)
	//  +------------+------------++------------+------------+

	__m128i sse_bcicodes0_1 = _mm_unpacklo_epi32(sse_index, sse_lrcolor);
	__m128i sse_bcicodes2_3 = _mm_unpackhi_epi32(sse_index, sse_lrcolor);

	//  +------------+------------++------------+------------+
	//  |   lrcolor1 | index (r1) ||   lrcolor0 | index (r0) | (sse_bcicodes0_1)
	//  +------------+------------++------------+------------+
	//  +------------+------------++------------+------------+
	//  |   lrcolor3 | index (r3) ||   lrcolor2 | index (r2) | (sse_bcicodes2_3)
	//  +------------+------------++------------+------------+

	/*if ((idx_code & 1) == 0) {
		_mm_stream_si128((__m128i*)&bci_codes[0], sse_bcicodes0_1);
		_mm_stream_si128((__m128i*)&bci_codes[2], sse_bcicodes2_3);
	}
	else {*/
		_mm_storeu_si128((__m128i*)&bci_codes[0], sse_bcicodes0_1);
		_mm_storeu_si128((__m128i*)&bci_codes[2], sse_bcicodes2_3);
	//}
}

void serial_bci(unsigned int* bcodes, unsigned int* indexes, PVParallelView::PVHSVColor* colors, PVParallelView::PVBCICode* bci_codes)
{
	for (int i=0; i<NBUCKET; i++)
	{
		PVParallelView::PVBCICode bci;
		bci.int_v = indexes[i] | ((uint64_t)bcodes[i] << 32);
		bci.s.color = colors[i].h();
		bci_codes[i] = bci;
	}
}

void print_bci_codes(PVParallelView::PVBCICode* bci_codes)
{
	for (int i=0; i<NBUCKET; i++)
	{
		std::cout << std::hex << bci_codes[i].int_v << std::endl;
	}
	std::cout << "---" << std::endl;
}

int main(void) {

	unsigned int bcodes[] = {0x00011111, 0x00022222, 0x00033333, 0x00044444};
	unsigned int indexes[] = {0xAAAAAAAA, 0xBBBBBBBB, 0xCCCCCCCC, 0xDDDDDDDD};

	PVParallelView::PVHSVColor* colors = PVParallelView::PVHSVColor::init_colors(NBUCKET);
	PVParallelView::PVBCICode* bci_codes_serial = PVParallelView::PVBCICode::allocate_codes(NBUCKET);
	PVParallelView::PVBCICode* bci_codes_sse = PVParallelView::PVBCICode::allocate_codes(NBUCKET);

	serial_bci(bcodes, indexes, colors, bci_codes_serial);
	sse_bci(bcodes, indexes, colors, bci_codes_sse);

	print_bci_codes(bci_codes_serial);
	print_bci_codes(bci_codes_sse);

	delete [] colors;
	PVParallelView::PVBCICode::free_codes(bci_codes_serial);
	PVParallelView::PVBCICode::free_codes(bci_codes_sse);


	return EXIT_SUCCESS;
}

