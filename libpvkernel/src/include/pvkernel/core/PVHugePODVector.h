#ifndef PVCORE_PVHUGEPODVECTOR_H
#define PVCORE_PVHUGEPODVECTOR_H

#include <pvkernel/core/PVAllocators.h>
#include <cassert>

namespace PVCore {

template <typename T, size_t Align = 16>
class PVHugePODVector
{
	static_assert(std::is_pod<T>::value, "PVHugePODVector: T must be a POD!");

public:
	typedef T value_type;
	typedef PVNUMAHugePagedInterleavedAllocator<value_type> allocator_type;
	typedef typename allocator_type::pointer pointer;
	typedef typename allocator_type::const_pointer const_pointer;
	typedef typename allocator_type::reference reference;
	typedef typename allocator_type::const_reference const_reference;
	typedef typename allocator_type::size_type size_type;

	typedef pointer       iterator;
	typedef const_pointer const_iterator;

private:
	typedef typename allocator_type::template rebind<char>::other char_allocator_type;

	static size_type page_size;

public:
	PVHugePODVector():
		_buf(nullptr),
		_aligned_buf(nullptr),
		_size(0)
	{ }

	PVHugePODVector(PVHugePODVector const& o):
		_buf(nullptr),
		_aligned_buf(nullptr),
		_size(0)
	{
		copy(o);
	}

	PVHugePODVector(PVHugePODVector && o)
	{
		move(o);
	}

	~PVHugePODVector()
	{
		if (_buf) {
			free();
		}
	}

public:
	PVHugePODVector& operator=(PVHugePODVector && o)
	{
		if (&o != this) {
			move(o);
		}
		return *this;
	}

	PVHugePODVector& operator=(PVHugePODVector const& o)
	{
		if (&o != this) {
			copy(o);
		}
		return *this;
	}

public:
	void resize(const size_type n)
	{
		if (n == size()) {
			return;
		}

		if (_buf) {
			reallocate(n);
		}
		else {
			allocate(n);
		}
	}
	
	void clear()
	{
		if (_buf) {
			free();
			_buf = nullptr;
			_aligned_buf = nullptr;
		}
	}

public:
	inline reference at(size_type i) { assert(i < size()); return _aligned_buf[i]; }
	inline const_reference at(size_type i) const { assert(i < size()); return _aligned_buf[i]; }

	inline reference operator[](size_type i) { return at(i); }
	inline const_reference operator[](size_type i) const { return at(i); }

	inline size_type size() const { return _size; }

public:
	iterator begin() { return _aligned_buf; }
	const_iterator begin() const { return _aligned_buf; }

	iterator end() { return (_buf) ? (&_aligned_buf[size()]) : nullptr; }
	const_iterator end() const { return (_buf) ? (&_aligned_buf[size()]) : nullptr; }

private:
	void allocate(const size_t n)
	{
		assert(!_buf);
		_buf = reinterpret_cast<pointer>(char_allocator_type().allocate(n*sizeof(T) + Align));
		if (_buf == (pointer)-1) {
			throw std::bad_alloc();
		}
		set_aligned_buf();
		_size = n;
	}

	void reallocate(const size_t n)
	{
		assert(n != size());
		_buf = reinterpret_cast<pointer>(mremap(_buf, real_buffer_size(), n*sizeof(T) + Align, MREMAP_MAYMOVE));
		if (_buf == (pointer)-1) {
			throw std::bad_alloc();
		}
		set_aligned_buf();
		_size = n;
	}

	inline void set_aligned_buf()
	{
		_aligned_buf = reinterpret_cast<pointer>((((uintptr_t)(_buf) + Align - 1)/Align)*Align);
	}

	inline size_t real_buffer_size() const
	{
		assert(_aligned_buf >= _buf);
		return (_size*sizeof(T)) + ((uintptr_t)_aligned_buf - (uintptr_t)_buf);
	}

	void free()
	{
		assert(_buf);
		char_allocator_type().deallocate((char*)_buf, real_buffer_size());
		_size = 0;
	}

	void move(PVHugePODVector && o)
	{
		assert(&o != this);
		_buf = o._buf;
		_aligned_buf = o._aligned_buf;
		_size = o._size;
		o._buf = nullptr;
		o._n = 0;
	}

	void copy(PVHugePODVector const& o)
	{
		resize(o._size);
		memcpy(begin(), o.begin(), size()*sizeof(T));
	}

private:
	pointer _buf;
	pointer _aligned_buf;
	size_t  _size;
};

}

#endif
