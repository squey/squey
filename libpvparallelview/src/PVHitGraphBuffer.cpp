
#include <pvparallelview/PVHitGraphBuffer.h>

#include <cstdlib>
#include <string.h>

// 
// PVHitGraphBuffer
//

PVParallelView::PVHitGraphBuffer::PVHitGraphBuffer()
{
	posix_memalign((void**) &_buf, 16, size_bytes());
}

PVParallelView::PVHitGraphBuffer::~PVHitGraphBuffer()
{
	if (_buf) {
		free(_buf);
	}
}

void PVParallelView::PVHitGraphBuffer::set_zero()
{
	memset(_buf, 0, size_bytes());
}


void PVParallelView::PVHitGraphBuffer::shift_left(int n)
{
	// TODO: implement
}

void PVParallelView::PVHitGraphBuffer::shift_right(int n)
{
	// TODO: implement
}
