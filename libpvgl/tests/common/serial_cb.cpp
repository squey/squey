#include <common/common.h>
#include <common/serial_cb.h>

void serial_cb(Point* pts, size_t n, CollisionBuffer cb)
{
	for (size_t i = 0; i < n; i++) {
		int bit = (pts[i].y1)*PIXELS_CB + (pts[i].y2);
		B_SET(cb[bit>>5], (bit&31));
	}
}

