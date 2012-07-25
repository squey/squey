/**
 * \file PVSharedPointer.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_SHAREDPOINTER_H
#define PVCORE_SHAREDPOINTER_H

#include <pvkernel/core/PVSpinLock.h>
#include <tbb/atomic.h>

#include <cstdint>
#include <ostream>
#include <limits>

namespace PVCore
{

template <typename T>
class PVSharedPtr;

namespace __impl
{

template <typename T>
class PVRefCounter
{
	friend class PVSharedPtr<T>;

public:
	typedef T        type;
	typedef T*       pointer;
	typedef    void(*deleter)(pointer);

	PVRefCounter() : _data(nullptr), _deleter(nullptr)
	{
		_count = 1;
	}

	PVRefCounter(pointer p, deleter d) : _data(p), _deleter(d)
	{
		_count = 1;
	}

	~PVRefCounter()
	{ }

	inline long operator++()
	{ return _count.fetch_and_add(1) + 1; }

	inline long operator--()
	{ return _count.fetch_and_add(-1) - 1; }

	inline pointer get() { return _data; }

	inline operator long() const { return _count.fetch_and_add((long)0); }

	inline void set_deleter(deleter d)
	{
		pv_spin_lock_guard_t guard(_spin_lock);
		_deleter = d;
	}

	inline deleter get_deleter()
	{
		pv_spin_lock_guard_t guard(_spin_lock);
		return _deleter;
	}

protected:
	pointer               _data;
	deleter               _deleter;
	mutable tbb::atomic<long>     _count;
	pv_spin_lock_t _spin_lock;
};

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
	typedef T                          type;
	typedef __impl::PVRefCounter<type> ref_counter_t;
	typedef type*                      pointer;
	typedef void(*deleter)(pointer);

	PVSharedPtr()
	{
		_ref_count = new ref_counter_t(nullptr, nullptr);
	}

	explicit PVSharedPtr(pointer p)
	{
		_ref_count = new ref_counter_t(p, nullptr);
	}

	PVSharedPtr(pointer p, deleter d)
	{
		_ref_count = new ref_counter_t(p, d);
	}

	PVSharedPtr(const PVSharedPtr<type> &rhs)
	{
		_ref_count = rhs._ref_count;
		++(*_ref_count);
	}

	~PVSharedPtr()
	{
		if ((--(*_ref_count)) == 0) {
			if (_ref_count->_data) {
				// Delete this shared ptr's data
				if (_ref_count->_deleter) {
					(_ref_count->_deleter)(_ref_count->_data);
				}
				else {
					delete _ref_count->_data;
				}
			}
			delete _ref_count;
		}
	}

	inline void reset(pointer p = 0)
	{
		PVSharedPtr(p, get_deleter()).swap(*this);
	}

	PVSharedPtr<type>& operator=(const PVSharedPtr<type> &rhs)
	{
		PVSharedPtr<type>(rhs).swap(*this);
		return *this;
	}

	inline pointer get() const { return _ref_count->get(); }
	inline operator bool() const { return (get() != nullptr); }

	inline T &operator*() const
	{
		assert(get() != 0);
		return *get();
	}

	T *operator->() const
	{
		assert(get() != 0);
		return get();
	}

	void set_deleter(deleter d)	{ _ref_count->set_deleter(d); }
	deleter get_deleter() const	{ return _ref_count->get_deleter(); }

	long use_count() const { return *_ref_count; }

	inline void swap(PVSharedPtr<T>& other) { std::swap(_ref_count, other._ref_count); }

private:
	ref_counter_t *_ref_count;
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

template<class T> void swap(PVSharedPtr<T>& a, PVSharedPtr<T>& b)
{
	a.swap(b);
}

}

#endif // PVCORE_SHAREDPOINTER_H
