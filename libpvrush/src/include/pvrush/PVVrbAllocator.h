#ifndef PVVRBALLOCATOR_FILE_H
#define PVVRBALLOCATOR_FILE_H

#ifdef PICVIZ_USE_VRB

extern "C" {
#include <vrb.h>
}

#include <iostream>
#include <QMutex>
#include <pvcore/general.h>

namespace PVRush {

class LibRushDecl unorder_deallocation : public std::exception {};

template<typename T>
class LibRushDecl PVVrbAllocator {
public : 
	typedef T value_type;
	typedef value_type* pointer;
	typedef const value_type* const_pointer;
	typedef value_type& reference;
	typedef const value_type& const_reference;
	typedef std::size_t size_type;
	typedef std::ptrdiff_t difference_type;

public : 
	template<typename U>
		struct rebind {
			typedef PVVrbAllocator<U> other;
		};

public : 
	PVVrbAllocator(vrb_p v) : _v(v) {}
	~PVVrbAllocator() {}
	PVVrbAllocator(PVVrbAllocator const& org)
	{
		_v = org._v;
		_mutex = org._mutex;
	}
	template<typename U>
		inline PVVrbAllocator(PVVrbAllocator<U> const& org)
		{
			_v = org._v;
			_mutex = org._mutex;
		}

	inline pointer address(reference r) { return &r; }
	inline const_pointer address(const_reference r) { return &r; }

	inline pointer allocate(size_type cnt,	typename std::allocator<void>::const_pointer = 0) { 
		_mutex.lock();
		size_type alloc_size = sizeof(T)*cnt;
		if (vrb_space_len(_v) < alloc_size) {
			PVLOG_ERROR("PVVrbAllocator: unable to allocate  %d bytes, %d remaining !\n", alloc_size, vrb_space_len(_v));
			_mutex.unlock();
			throw std::bad_alloc();
		}
		PVLOG_HEAVYDEBUG("vrb: allocating %d bytes\n", alloc_size);
		pointer pret = (pointer)(vrb_space_ptr(_v));
		vrb_give(_v, alloc_size);
		_mutex.unlock();
		return pret;
	}
	
	inline void deallocate(pointer p, size_type n) { 
		_mutex.lock();
		PVLOG_HEAVYDEBUG("vrb: deallocating %d bytes\n", sizeof(T)*n);
		pointer pbegin = (pointer) vrb_data_ptr(_v);
		if (((intptr_t)p - (intptr_t)pbegin) % vrb_capacity(_v) != 0) {
			PVLOG_ERROR("PVVrbAllocator: unordered deallocation occured : p=%x, pbegin=%x\n", (uintptr_t)p, (uintptr_t)pbegin);
			_mutex.unlock();
			throw unorder_deallocation();
		}
		vrb_take(_v, n);
		_mutex.unlock();
	}

	inline size_type max_size() const {
		_mutex.lock();
		size_t sfree = vrb_space_len(_v);
		_mutex.unlock();
		return  sfree / sizeof(T);	
	}

	inline void construct(pointer p, const T& t) { new(p) T(t); }
	inline void destroy(pointer p) { p->~T(); }

	inline bool operator==(PVVrbAllocator const& a) { return a._v == _v; }
	inline bool operator!=(PVVrbAllocator const& a) { return a._v != _v; }
public:
	vrb_p _v;
private:
	QMutex _mutex;
};

}

#endif //PICVIZ_USE_VRB


#endif
