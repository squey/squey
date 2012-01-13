#ifndef PVCORE_PVALLOCATORS_H
#define PVCORE_PVALLOCATORS_H

#include <pvkernel/core/general.h>

#include <cstddef>
#include <typeinfo>

#include <stdint.h>
#include <malloc.h>
#include <sys/mman.h>

namespace PVCore {

template <class T, class FallbackAllocator = std::allocator<T> >
class PVPreAllocatedAllocator
{
public:
	typedef T value_type;
	typedef T* pointer;
	typedef T& reference;
	typedef const T* const_pointer;
	typedef const T& const_reference;
	typedef size_t size_type;
	typedef ptrdiff_t difference_type;
	template<class U> struct rebind {
		typedef typename FallbackAllocator::template rebind<U>::other OtherFallbackAllocator;
		typedef PVPreAllocatedAllocator<U, OtherFallbackAllocator> other;
	}; 
public:
	PVPreAllocatedAllocator() throw():
		_p(NULL), _n(0), _cur(0)
	{ }
	
	PVPreAllocatedAllocator(pointer buf, size_type n) throw():
		_p(buf), _n(n), _cur(0)
	{ }
	PVPreAllocatedAllocator(const PVPreAllocatedAllocator& o) throw():
		_p(o._p), _n(o._n), _cur(o._cur)
	{
		o._p = NULL;
		o._n = 0;
		o._cur = 0;
	}

	template<class U, class FA>
	PVPreAllocatedAllocator(const PVPreAllocatedAllocator<U,FA>& o) throw():
		_p(NULL), _n(0), _cur(0)
	{ }

public:
	pointer address(reference x) const { return &x; }
	const_pointer address(const_reference x) const { return &x; }

	pointer allocate(size_type n)
	{
		if ((_p != NULL) && (_cur + n < _n)) {
			_cur += n;
			return _p;
		}

		// Use the fallback allocator
		return _fall_alloc.allocate(n);
	}

	void deallocate(pointer p, size_type n)
	{
		if (_p != NULL && p >= _p && p < _p + _n) {
			// The user is responsible for this buffer
			return;
		}

		_fall_alloc.deallocate(p, n);
	}

	size_type max_size() const throw()
	{
		return _fall_alloc.max_size();
	}

	void construct(pointer p, const_reference val)
	{
		::new((void*) p) value_type(val);
	}

	void destroy(pointer p)
	{
		p->~value_type();
	}

private:
	mutable pointer _p;
	mutable size_type _n;
	mutable size_type _cur;
	FallbackAllocator _fall_alloc;
};

// AG: this is a really ugly hack to have an std::list with pre-allocation from a more global
// pointer. This is used to have a better control (and better performances) on PVField
// allocations and desallocations !
template <class T, class FallbackAllocator = std::allocator<T> >
class PVPreAllocatedListAllocator
{
public:
	typedef T value_type;
	typedef T* pointer;
	typedef T& reference;
	typedef const T* const_pointer;
	typedef const T& const_reference;
	typedef size_t size_type;
	typedef ptrdiff_t difference_type;
	template<class U> struct rebind {
		typedef typename FallbackAllocator::template rebind<U>::other OtherFallbackAllocator;
		typedef PVPreAllocatedListAllocator<U, OtherFallbackAllocator> other;
	}; 
public:
	PVPreAllocatedListAllocator() throw():
		_p(NULL), _size(0), _scur(0)
	{ }
	
	PVPreAllocatedListAllocator(void* buf, size_type n) throw():
		_p(buf), _size(n), _scur(0)
	{ }
	PVPreAllocatedListAllocator(const PVPreAllocatedListAllocator& o) throw():
		_p(o._p), _size(o._size), _scur(o._scur)
	{ }

	template<class U, class FA>
	PVPreAllocatedListAllocator(const PVPreAllocatedListAllocator<U,FA>& o) throw():
		_p(o._p), _size(o._size), _scur(o._scur)
	{ }

public:
	pointer address(reference x) const { return &x; }
	const_pointer address(const_reference x) const { return &x; }

	pointer allocate(size_type n)
	{
		size_t size_buf = sizeof(value_type)*n;
		if ((_p != NULL) && (_scur + size_buf < _size)) {
			void* p_ret = (void*) ((uintptr_t) _p + _scur);
			_scur += size_buf;
			return (pointer) p_ret;
		}

		//PVLOG_WARN("(PVPreAllocatedListAllocator::allocate) using fallback allocator for type %s...\n", typeid(value_type).name());
		pointer ret = _fall_alloc.allocate(n);
		// Use the fallback allocator
		return ret;
	}

	void deallocate(pointer p, size_type n)
	{
		if (_p != NULL && (void*)p >= _p && (uintptr_t)p < (uintptr_t)_p + _size) {
			// The user is responsible for this buffer
			return;
		}

		_fall_alloc.deallocate(p, n);
	}

	size_type max_size() const throw()
	{
		return _fall_alloc.max_size();
	}

	void construct(pointer p, const_reference val)
	{
		::new((void*) p) value_type(val);
	}

	void destroy(pointer p)
	{
		p->~value_type();
	}

	bool operator!=(const PVPreAllocatedListAllocator& /*o*/) const { return false; }

public:
	mutable void* _p;
	mutable size_type _size;
	mutable size_type _scur;
	FallbackAllocator _fall_alloc;
};

template <class T>
class PVMMapAllocator
{
public:
	typedef T value_type;
	typedef T* pointer;
	typedef T& reference;
	typedef const T* const_pointer;
	typedef const T& const_reference;
	typedef size_t size_type;
	typedef ptrdiff_t difference_type;
	template<class U> struct rebind {
		typedef PVMMapAllocator<U> other;
	}; 

public:
	pointer address(reference x) const { return &x; }
	const_pointer address(const_reference x) const { return &x; }

	pointer allocate(size_type n)
	{
		return (pointer) mmap(NULL, sizeof(value_type)*n, PROT_WRITE|PROT_READ, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
	}

	void deallocate(pointer p, size_type n)
	{
		if (munmap(p, sizeof(value_type)*n) != 0) {
			PVLOG_ERROR("munmap failed.\n");
		}
	}

	size_type max_size() const throw()
	{
		// From TBB's scalable allocator
		size_type absolutemax = static_cast<size_type>(-1) / sizeof (value_type);
		return (absolutemax > 0 ? absolutemax : 1);
	}

	void construct(pointer p, const_reference val)
	{
		::new((void*) p) value_type(val);
	}

	void destroy(pointer p)
	{
		p->~value_type();
	}
};

template <class T, int Align>
class PVAlignedAllocator
{
public:
	typedef T DECLARE_ALIGN(Align) value_type;
	typedef value_type* pointer;
	typedef const value_type* const_pointer;
	typedef value_type& reference;
	typedef const value_type& const_reference;
	typedef size_t size_type;
	typedef ptrdiff_t difference_type;
	template<class U> struct rebind {
		typedef PVAlignedAllocator<U, Align> other;
	}; 

public:
	pointer address(reference x) const { return &x; }
	const_pointer address(const_reference x) const { return &x; }

	pointer allocate(size_type n)
	{
		pointer p;
#ifdef WIN32
		p = (pointer) _aligned_malloc(sizeof(size_type)*n, Align);
		if (p == NULL) {
			throw std::bad_alloc();
		}
#else
		int ret = posix_memalign((void**) &p, Align, sizeof(size_type)*n);
		if (ret != 0) {
			throw std::bad_alloc();
		}
#endif
		return p;
	}

	void deallocate(pointer p, size_type /*n*/)
	{
		free(p);
	}

	size_type max_size() const throw()
	{
		// From TBB's scalable allocator
		size_type absolutemax = static_cast<size_type>(-1) / sizeof (value_type);
		return (absolutemax > 0 ? absolutemax : 1);
	}

	void construct(pointer p, const_reference val)
	{
		::new((void*) p) value_type(val);
	}

	void destroy(pointer p)
	{
		p->~value_type();
	}
};

}

#endif
