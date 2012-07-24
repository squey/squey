/**
 * \file masks.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/general.h>
#include <pvparallelview/PVBCICode.h>

int main()
{
	PVParallelView::PVBCICode code;

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
	code.s.color = 511;
	printf("Mask for color: %lx\n", code.int_v >> 32);

	return 0;
}
