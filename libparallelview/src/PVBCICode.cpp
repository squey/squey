#include <pvparallelview/common.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVHSVColor.h>

#include <stdlib.h>

PVParallelView::PVBCICode* PVParallelView::PVBCICode::allocate_codes(size_t n)
{
	PVBCICode* ret = PVBCICode::allocator().allocate(n);
	return ret;
}

void PVParallelView::PVBCICode::free_codes(PVBCICode* codes)
{
	PVBCICode::allocator().deallocate(codes, 0);
}

void PVParallelView::PVBCICode::init_random_codes(PVBCICode* codes, size_t n)
{
	for (size_t i = 0; i < n; i++) {
		PVBCICode c;
		c.int_v = 0;
		//c.s.idx = rand();
		c.s.idx = n-i;
		//c.s.l = ((i/1024)*4)%1024;
		//c.s.r = i&(MASK_INT_YCOORD);
		//c.s.l = rand()&(MASK_INT_YCOORD);
		//c.s.r = rand()&(MASK_INT_YCOORD);
		//c.s.r = (c.s.l+10)&MASK_INT_YCOORD;
		if (i < 1024) {
			c.s.l = MASK_INT_YCOORD/2;
		}
		else {
			c.s.l = MASK_INT_YCOORD/5;
		}
		c.s.r = i&(MASK_INT_YCOORD);
		//c.s.color = rand()&((1<<9)-1);
		c.s.color = i%((1<<HSV_COLOR_NBITS_ZONE)*6);
		//c.s.color = 1;
		codes[i] = c;
	}
}
