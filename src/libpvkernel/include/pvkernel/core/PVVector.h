/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVCORE_PVVECTOR_H
#define PVCORE_PVVECTOR_H

#include <pvkernel/core/PVAllocators.h>

// for memcpy()
#include <string.h>

namespace PVCore
{

template <class C, int INCREMENT = 1000, class Alloc = PVCore::PVReallocableCAllocator<C>>
class PVVector
{
  public:
	typedef C value_type;
	typedef unsigned size_type;

	PVVector() : _array(0), _size(0), _index(0) {}

	explicit PVVector(const unsigned size) : _array(0), _index(0)
	{
		if (size != 0) {
			reallocate(size);
		} else {
			_size = size;
		}
	}

	~PVVector() { clear(); }

	void reserve(const unsigned size) { reallocate(size); }

	void compact()
	{
		if (_index) {
			reallocate(_index);
		} else {
			clear();
		}
	}

	void reset() { _index = 0; }

	void clear()
	{
		if (_array) {
			Alloc().deallocate(_array, _size);
			_array = 0;
			_size = 0;
			_index = 0;
		}
	}

	inline unsigned size() const { return _index; }

	inline unsigned capacity() const { return _size; }

	inline size_t memory() const { return sizeof(PVVector) + _size * sizeof(C); }

	inline bool is_null() const { return (_array == 0); }

	inline C& at(const int i) { return _array[i]; }

	inline C const& at(const int i) const { return _array[i]; }

	inline void push_back(const C& c)
	{
		if (_index == _size) {
			reallocate(_size + INCREMENT);
		}
		_array[_index++] = c;
	}

	PVVector<C>& operator=(const PVVector<C>& v)
	{
		clear();
		if (v._size) {
			_index = v._index;
			reallocate(v._size);
			memcpy(_array, v._array, _index * sizeof(C));
		}
		return *this;
	}

	bool operator==(const PVVector<C>& v) const
	{
		if (_index != v._index) {
			return false;
		} else if (_array == 0) {
			return (v._array == 0);
		} else if (v._array == 0) {
			return false;
		} else {
			return (memcmp(_array, v._array, _index * sizeof(C)) == 0);
		}
	}

	value_type* get_pointer() const { return _array; }

	void set_index(unsigned index) { _index = index; }

  private:
	void reallocate(const unsigned size)
	{
		_array = Alloc().reallocate(_array, _size, size);
		_size = size;
	}

  private:
	C* _array;
	unsigned _size;
	unsigned _index;
};
} // namespace PVCore

#endif // PVCORE_PVVECTOR_H
