#ifndef PVCORE_PVTBBMAXARRAY_H
#define PVCORE_PVTBBMAXARRAY_H

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#ifdef WIN32
#include <pvcore/win32-vs2008-stdint.h>
#else
#include <stdint.h>
#endif

namespace PVCore {

template <typename T>
class LibExport PVTBBMaxArray {
protected:
	const T* _arr;
	T _max_value;
	T _min_value;

public:
	PVTBBMaxArray(const T* arr, T min_value):
		_arr(arr),
		_max_value(min_value),
		_min_value(min_value)
	{
	}

	PVTBBMaxArray(PVTBBMaxArray& x, tbb::split):
		_arr(x._arr),
		_max_value(x._min_value)
	{
	}
public:
	void operator()(const tbb::blocked_range<uint64_t>& r)
	{
		const T* arr = _arr;
		for (uint64_t i = r.begin(); i != r.end(); i++) {
			const T& value = arr[i];
			if (value > _max_value) {
				_max_value = value;
			}
		}
	}

	void join(const PVTBBMaxArray& y)
	{
		if (y._max_value > _max_value) {
			_max_value = y._max_value;
		}
	}

	const T& get_max_value() { return _max_value; }
};

}

#endif
