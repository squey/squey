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

#ifndef PVCORE_PVALLOCATORS_H
#define PVCORE_PVALLOCATORS_H

#include <pvkernel/core/PVLogger.h> // for PVLOG_ERROR, PVLOG_WARN

#include <pvbase/types.h> // for DECLARE_ALIGN

#include <cstddef>     // for size_t, ptrdiff_t
#include <cstdint>     // for uintptr_t
#include <cstdlib>     // for free, malloc, realloc
#include <mm_malloc.h> // for posix_memalign
#include <numa.h>      // for numa_alloc_interleaved, etc
#include <sys/mman.h>  // for mmap, madvise, munmap

#include <algorithm> // for forward
#include <memory>    // for allocator
#include <new>       // for operator new, bad_alloc

namespace PVCore
{

/*! \brief Defines a C++ compliant allocator that pre-allocates memory for std::list objects (hack).
 *
 * This class defines a C++ compliant allocator that pre-allocates memory for std::list objects.
 *This is a really ugly hack to have an std::list with pre-allocation from a more global
 * pointer. This is used to have a better control (and better performances) on PVField allocations
 *and desallocations.
 * This is (again) a hack and should not be used if you don't understand what you are doing.
 *
 * \todo Improves the way PVField objects are created and destroyed, so that such a thing won't be
 *necessary anymore.
 */
template <class T, class FallbackAllocator = std::allocator<T>>
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
	template <class U>
	struct rebind {
		typedef typename FallbackAllocator::template rebind<U>::other OtherFallbackAllocator;
		typedef PVPreAllocatedListAllocator<U, OtherFallbackAllocator> other;
	};

  public:
	PVPreAllocatedListAllocator() throw() : _p(nullptr), _size(0), _scur(0) {}

	PVPreAllocatedListAllocator(void* buf, size_type n) throw() : _p(buf), _size(n), _scur(0) {}
	PVPreAllocatedListAllocator(const PVPreAllocatedListAllocator& o) throw()
	    : _p(o._p), _size(o._size), _scur(o._scur)
	{
	}

	template <class U, class FA>
	explicit PVPreAllocatedListAllocator(const PVPreAllocatedListAllocator<U, FA>& o) throw()
	    : _p(o._p), _size(o._size), _scur(o._scur)
	{
	}

  public:
	pointer address(reference x) const { return &x; }
	const_pointer address(const_reference x) const { return &x; }

	pointer allocate(size_type n)
	{
		size_t size_buf = sizeof(value_type) * n;
		if ((_p != nullptr) && (_scur + size_buf < _size)) {
			void* p_ret = (void*)((uintptr_t)_p + _scur);
			_scur += size_buf;
			return (pointer)p_ret;
		}

		// PVLOG_WARN("(PVPreAllocatedListAllocator::allocate) using fallback allocator for type
		// %s...\n", typeid(value_type).name());
		pointer ret = _fall_alloc.allocate(n);
		// Use the fallback allocator
		return ret;
	}

	void deallocate(pointer p, size_type n)
	{
		if (_p != nullptr && (void*)p >= _p && (uintptr_t)p < (uintptr_t)_p + _size) {
			// The user is responsible for this buffer
			return;
		}

		_fall_alloc.deallocate(p, n);
	}

	size_type max_size() const throw() { return _fall_alloc.max_size(); }

	template <class... U>
	void construct(pointer p, U&&... val)
	{
		::new ((void*)p) value_type(std::forward<U>(val)...);
	}

	void destroy(pointer p) { p->~value_type(); }

	bool operator!=(const PVPreAllocatedListAllocator& /*o*/) const { return false; }

  public:
	mutable void* _p;
	mutable size_type _size;
	mutable size_type _scur;
	FallbackAllocator _fall_alloc;
};

/*! \brief C++ compliant allocator class that uses direct calls to mmap/munmap.
 *
 * This allocator allocates memory thanks to direct calls to mmap/munmap. This can be useful in
 *situations where we need to be sure that memory
 * will be given back to the system, or when zero-copy reallocation are needed. This is mainly used
 *for the NRAW table in PVRush::PVNraw.
 *
 * \todo More generally, we should improve the way some objects (like the NRAW table) are allocated,
 *so that direct nmap calls won't be needed.
 * \todo Make this work under Windows (which has a different memory management system than Linux..
 *!)
 */
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
	template <class U>
	struct rebind {
		typedef PVMMapAllocator<U> other;
	};

  public:
	pointer address(reference x) const { return &x; }
	const_pointer address(const_reference x) const { return &x; }

	pointer allocate(size_type n)
	{
		return (pointer)mmap(nullptr, sizeof(value_type) * n, PROT_WRITE | PROT_READ,
		                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	}

	void deallocate(pointer p, size_type n)
	{
		if (munmap(p, sizeof(value_type) * n) != 0) {
			PVLOG_ERROR("munmap failed.\n");
		}
	}

	size_type max_size() const throw()
	{
		// From TBB's scalable allocator
		size_type absolutemax = static_cast<size_type>(-1) / sizeof(value_type);
		return (absolutemax > 0 ? absolutemax : 1);
	}

	template <class... U>
	void construct(pointer p, U&&... val)
	{
		::new ((void*)p) value_type(std::forward<U>(val)...);
	}

	void destroy(pointer p) { p->~value_type(); }
};

/*! \brief C++ compliant allocator class that uses libnuma's allocation schemes.
 *
 * \todo Make this work under Windows (which has a different memory management system than Linux..
 *!)
 */
template <class T>
class PVNUMAHugePagedInterleavedAllocator
{
  public:
	typedef T value_type;
	typedef T* pointer;
	typedef T& reference;
	typedef const T* const_pointer;
	typedef const T& const_reference;
	typedef size_t size_type;
	typedef ptrdiff_t difference_type;
	template <class U>
	struct rebind {
		typedef PVNUMAHugePagedInterleavedAllocator<U> other;
	};

  public:
	pointer address(reference x) const { return &x; }
	const_pointer address(const_reference x) const { return &x; }

	pointer allocate(size_type n)
	{
		pointer mem;

		mem = (pointer)numa_alloc_interleaved(sizeof(value_type) * n);
		if (mem) {
			if (madvise(mem, sizeof(value_type) * n, MADV_HUGEPAGE) < 0) {
				PVLOG_WARN("can not set hugepage attribute.\n");
			}
		}

		return mem;
	}

	void deallocate(pointer p, size_type n) { numa_free(p, n); }

	size_type max_size() const throw()
	{
		// From TBB's scalable allocator
		size_type absolutemax = static_cast<size_type>(-1) / sizeof(value_type);
		return (absolutemax > 0 ? absolutemax : 1);
	}

	template <class... U>
	void construct(pointer p, U&&... val)
	{
		::new ((void*)p) value_type(std::forward<U>(val)...);
	}

	void destroy(pointer p) { p->~value_type(); }
};

/*! \brief C++ compliant allocator class that returns aligned pointers.
 *
 * This class uses posix_memalign to ensure that returned pointer are aligned on Align bytes.
 * This is used for buffers that are loaded/stored with SSE/AVX instructions for instance (thus
 *improving performances).
 * The value_type declares an aligned type, and thus can be used to give hints to the compiler about
 *the alignement of a pointer.
 *
 * \todo Provide a similar behavior under Windows.
 */
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
	template <class U>
	struct rebind {
		typedef PVAlignedAllocator<U, Align> other;
	};

  public:
	pointer address(reference x) const { return &x; }
	const_pointer address(const_reference x) const { return &x; }

	pointer allocate(size_type n)
	{
		pointer p;
		int ret = posix_memalign((void**)&p, Align, sizeof(value_type) * n);
		if (ret != 0) {
			throw std::bad_alloc();
		}
		return p;
	}

	void deallocate(pointer p, size_type /*n*/) { free(p); }

	size_type max_size() const throw()
	{
		// From TBB's scalable allocator
		size_type absolutemax = static_cast<size_type>(-1) / sizeof(value_type);
		return (absolutemax > 0 ? absolutemax : 1);
	}

	template <class... U>
	void construct(pointer p, U&&... val)
	{
		::new ((void*)p) value_type(std::forward<U>(val)...);
	}

	void destroy(pointer p) { p->~value_type(); }

	// C++11 compatibility
	bool operator==(const PVAlignedAllocator&) const { return true; }
	bool operator!=(const PVAlignedAllocator&) const { return false; }
};

template <class T>
class PVReallocableCAllocator
{
  public:
	typedef T value_type;
	typedef value_type* pointer;
	typedef const value_type& const_reference;
	typedef size_t size_type;

	pointer allocate(size_type n)
	{
		pointer p = (pointer)malloc(n * sizeof(value_type));
		if (p == nullptr) {
			throw std::bad_alloc();
		}
		return p;
	}

	pointer reallocate(pointer p, size_type /*on*/, size_type nn)
	{
		pointer np = (pointer)realloc(p, nn * sizeof(value_type));
		if (np == nullptr) {
			throw std::bad_alloc();
		}
		return np;
	}

	void deallocate(pointer p, size_type /*n*/) { free(p); }

	size_type max_size() const throw()
	{
		// From TBB's scalable allocator
		size_type absolutemax = static_cast<size_type>(-1) / sizeof(value_type);
		return (absolutemax > 0 ? absolutemax : 1);
	}

	template <class... U>
	void construct(pointer p, U&&... val)
	{
		::new ((void*)p) value_type(std::forward<U>(val)...);
	}

	void destroy(pointer p) { p->~value_type(); }
};

namespace PVMemory
{

void get_memory_usage(double& vm_usage, double& resident_set);
} // namespace PVMemory
} // namespace PVCore

#endif
