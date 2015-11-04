/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/general.h>
#include <pvparallelview/PVBCICode.h>

int main()
{
	PVParallelView::PVBCICode<NBITS_INDEX> code;

	code.int_v = 0;
	code.s.idx = 0xFFFFFFFF;
	printf("Mask for idx: %lx\n", code.int_v);

	code.int_v = 0;
	code.s.l = 1023;
	printf("Mask for l: %lx\n", code.int_v >> 32);

	code.int_v = 0;
	code.s.r = 1023;
	printf("Mask for r: %lx\n", code.int_v >> 32);

	code.int_v = 0;
	code.s.color = 255;
	printf("Mask for color: %lx\n", code.int_v >> 32);

	return 0;
}
