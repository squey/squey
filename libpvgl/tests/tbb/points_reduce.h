#ifndef CBREDUCE_H
#define CBREDUCE_H

#include "../Point.h"
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#define B_SET(x, n)      ((x) |= (1<<(n)))

class PointsReduce
{
public:
	PointsReduce(Point* pts):
		_pts(pts)
	{
		_buf = allocate_CB();
	}

	PointsReduce(PointsReduce const& x, tbb::split):
		_pts(x._pts)
	{
		_buf = allocate_CB();
	}

	~PointsReduce()
	{
		free_CB(_buf);
	}
public:
	void operator()(const tbb::blocked_range<size_t>& r)
	{
		Point const* tmp;
		Point* pts = _pts;
		size_t end = r.end();
		for (size_t i = r.begin(); i != end; i++) {
			tmp = &pts[i];
			// Set CB
			int bit = tmp->y1*1024 + tmp->y2;
			B_SET(_buf[bit>>5], (bit & 31));
		}
	}

	void join(PointsReduce const& x)
	{
		CollisionBuffer xcb = x._buf;
		for (size_t i = 0; i < NB_INT_CB; i++) {
			_buf[i] |= xcb[i];
		}
	}
private:
	CollisionBuffer _buf;
	Point* _pts;
};

#endif
