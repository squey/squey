#ifndef PVCORE_PVTBBMINMAXARRAY_H
#define PVCORE_PVTBBMINMAXARRAY_H

#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

#include <pvkernel/core/stdint.h>

namespace PVCore {

template <typename T>
class PVTBBMinMaxArray {
protected:
	const T* _arr;
	T _max_value;
	T _min_value;
	uint64_t _index_max;
	uint64_t _index_min;

public:
	PVTBBMinMaxArray(const T* arr) :
		_arr(arr),
		_max_value(arr[0]),
		_min_value(arr[0]),
		_index_max(0),
		_index_min(0)
	{
	}

	PVTBBMinMaxArray(PVTBBMinMaxArray& x, tbb::split):
		_arr(x._arr),
		_max_value(_arr[0]),
		_min_value(_arr[0])
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
				_index_max = i;
			}
			if (value < _min_value) {
				_min_value = value;
				_index_min = i;
			}
		}
	}

	void join(const PVTBBMinMaxArray& y)
	{
		if (y._max_value > _max_value) {
			_max_value = y._max_value;
			_index_max = y._index_max;
		}
		if (y._min_value < _min_value) {
			_min_value = y._min_value;
			_index_min = y._index_min;
		}
	}

	const T& get_max_value() const { return _max_value; }
	const T& get_min_value() const { return _min_value; }
	uint64_t get_max_index() const { return _index_max; }
	uint64_t get_min_index() const { return _index_min; }
};

}

#endif
