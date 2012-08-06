/**
 * \file PVSharedPointer.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_SHAREDPOINTER_H
#define PVCORE_SHAREDPOINTER_H

#include <cstdint>
#include <ostream>
#include <limits>
#include <cassert>
#include <type_traits>

#include <pvkernel/core/PVCounters.h>
#include <pvkernel/core/PVCountedBase.h>
#include <pvkernel/core/PVEnableSharedFromThis.h>

namespace PVCore
{

template <typename T>
class PVSharedPtr;

template <typename T>
class PVWeakPtr;

namespace __impl
{
struct static_cast_tag {};
struct const_cast_tag {};
struct dynamic_cast_tag {};
}

/**
 * @class PVSharedPtr
 *
 * @brief a shared pointer with a deleter which can be changed at run-time.
 *
 */
template <typename T>
class PVSharedPtr
{
public:
	template<typename X> friend class PVWeakPtr;
	template<typename X> friend class PVSharedPtr;

public:
	typedef T type;
	typedef PVSharedCounter shared_counter_t;
	typedef type* pointer;
	typedef void(*deleter)(pointer);


	PVSharedPtr() : _shared_count()
	{
	}

	PVSharedPtr(pointer p) : _shared_count(p)
	{
		enable_shared_from_this(this, p);
	}

	template<class Y>
	explicit PVSharedPtr(Y* p) : _shared_count(p)
	{
		enable_shared_from_this(this, p);
	}

	PVSharedPtr(pointer p, deleter d) : _shared_count(p, d)
	{
		enable_shared_from_this(this, p);
	}

	PVSharedPtr(const PVSharedPtr<T> &r) : _shared_count(r._shared_count)
	{
	}

	template <typename Y>
	explicit PVSharedPtr(PVSharedPtr<Y> const & r)
	{
		static_assert(std::is_convertible<Y, T>::value, "type Y is not derived from type T");
		_shared_count = r._shared_count;
	}

	 // Create PVSharedPtr from PVWeakPtr
	PVSharedPtr(PVWeakPtr<T> const & r) : _shared_count(r._weak_count)
	{
	}

	// for PVEnableSharedFromThis::_internal_accept_owner
	template <typename Y, typename Z>
	PVSharedPtr(PVSharedPtr<Y> const & r, Z* p): _shared_count(r._shared_count)
	{
		_shared_count.set((Y*)p);
	}

	template <class Y>
	PVSharedPtr(PVSharedPtr<Y> const & r, __impl::static_cast_tag) : _shared_count(r._shared_count)
	{
		_shared_count.set(static_cast<T*>(_shared_count.get()));
	}

	template<class Y>
	PVSharedPtr(PVSharedPtr<Y> const & r, __impl::const_cast_tag) : _shared_count(r._shared_count)
	{
		_shared_count.set(const_cast<T*>(_shared_count.get()));
	}

	template<class Y>
	PVSharedPtr(PVSharedPtr<Y> const & r, __impl::dynamic_cast_tag) : _shared_count(r._shared_count)
	{
		_shared_count.set(dynamic_cast<T*>(get()));
		if(get() == nullptr)
		{
			_shared_count = shared_counter_t();
		}
	}

	inline void reset()
	{
		PVSharedPtr<T>().swap(*this);
	}

	template<class Y> void reset(Y* p)
	{
		PVSharedPtr<T>(p).swap(*this);
	}

	template<class Y, class D> void reset(Y * p, D d)
	{
		PVSharedPtr<T>(p, d).swap(*this);
	}

	PVSharedPtr<T>& operator=(const PVSharedPtr<T> &rhs)
	{
		PVSharedPtr<T>(rhs).swap(*this);
		return *this;
	}

	inline pointer get() const { return (pointer) _shared_count.get(); }
	inline operator bool() const { return (_shared_count.get() != nullptr); }

	inline T& operator*() const
	{
		assert(_shared_count.get() != nullptr);
		return * (T*)_shared_count.get();
	}

	inline T* operator->() const
	{
		assert(_shared_count.get() != nullptr);
		return (T*) _shared_count.get();
	}

	void set_deleter(deleter d)	{ _shared_count.set_deleter((void*)d); }
	deleter get_deleter() const	{ return (deleter) _shared_count.get_deleter(); }

	long use_count() const { return _shared_count.use_count(); }

	inline void swap(PVSharedPtr<T>& other) { assert(this != &other); std::swap(_shared_count, other._shared_count); }

private:
	template<class X, class Y>
	inline void enable_shared_from_this(PVSharedPtr<X>* sp, PVCore::PVEnableSharedFromThis<Y>* const p)
	{
		if(p != nullptr)
		{
			p->_internal_accept_owner(sp, (Y*)p);
		}
	}

	inline void enable_shared_from_this(...)
	{
	}

private:
	shared_counter_t _shared_count;
};

template <class T, class U>
bool operator==(const PVSharedPtr<T>& lhs, const PVSharedPtr<U>& rhs)
{
	return (lhs.get() == rhs.get());
}

template <class T, class U>
bool operator!=(const PVSharedPtr<T>& lhs, const PVSharedPtr<U>& rhs)
{
	return (lhs.get() != rhs.get());
}

template <class T, class U>
bool operator<(const PVSharedPtr<T>& lhs, const PVSharedPtr<U>& rhs)
{
	return (lhs.get() < rhs.get());
}

template <class T, class U>
bool operator<=(const PVSharedPtr<T>& lhs, const PVSharedPtr<U>& rhs)
{
	return (lhs.get() <= rhs.get());
}

template <class T, class U>
bool operator>(const PVSharedPtr<T>& lhs, const PVSharedPtr<U>& rhs)
{
	return (lhs.get() > rhs.get());
}

template <class T, class U>
bool operator>=(const PVSharedPtr<T>& lhs, const PVSharedPtr<U>& rhs)
{
	return (lhs.get() >= rhs.get());
}

template <typename T>
PVSharedPtr<T> make_shared(T *t)
{
	return PVSharedPtr<T>(t);
}

template <typename T, typename D>
PVSharedPtr<T> make_shared(T *t, D d)
{
	return PVSharedPtr<T>(t, d);
}

template <class T, class U, class V>
std::basic_ostream<U, V>& operator<<(std::basic_ostream<U, V>& os, const PVSharedPtr<T>& rhs)
{
	return os << rhs.get();
}

template <class T>
void swap(PVSharedPtr<T>& a, PVSharedPtr<T>& b)
{
	a.swap(b);
}

template<class T, class U>
PVSharedPtr<T> static_pointer_cast(PVSharedPtr<U> const & r)
{
    return PVSharedPtr<T>(r, __impl::static_cast_tag());
}

template<class T, class U>
PVSharedPtr<T> const_pointer_cast(PVSharedPtr<U> const & r)
{
    return PVSharedPtr<T>(r, __impl::const_cast_tag());
}

template<class T, class U>
PVSharedPtr<T> dynamic_pointer_cast(PVSharedPtr<U> const & r)
{
    return PVSharedPtr<T>(r, __impl::dynamic_cast_tag());
}

}

#endif // PVCORE_SHAREDPOINTER_H
