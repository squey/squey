/**
 * \file PVSharedPointer.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_SHAREDPOINTER_H
#define PVCORE_SHAREDPOINTER_H

#include <pvkernel/core/PVSpinLock.h>

#include <cstdint>
#include <ostream>

namespace PVCore
{

namespace __impl
{

template <typename T>
class pv_ref_counter
{
public:
	typedef T        type;
	typedef T*       pointer;
	typedef    void(*deleter)(pointer);

	pv_ref_counter() : _data(nullptr), _deleter(nullptr), _count(0), _spin_lock()
	{}

	pv_ref_counter(pointer p, deleter d) : _data(p), _deleter(d), _count(1)
	{}

	~pv_ref_counter()
	{}

	void inc_count()
	{
		pv_spin_lock_guard_t guard(_spin_lock);
		++_count;
	}

	void dec_count()
	{
		pv_spin_lock_guard_t guard(_spin_lock);
		--_count;
		if(_count == 0) {
			if (_deleter) {
				_deleter(_data);
			} else {
				delete _data;
			}
		}
	}

	uint32_t use_count()
	{
		pv_spin_lock_guard_t guard(_spin_lock);
		return _count;
	}

	pointer get()
	{
		pv_spin_lock_guard_t guard(_spin_lock);
		return _data;
	}

	void set_deleter(deleter d)
	{
		pv_spin_lock_guard_t guard(_spin_lock);
		_deleter = d;
	}

	deleter get_deleter()
	{
		pv_spin_lock_guard_t guard(_spin_lock);
		return _deleter;
	}

private:
	pointer        _data;
	deleter        _deleter;
	uint32_t       _count;
	pv_spin_lock_t _spin_lock;
};

}

/**
 * @class pv_shared_ptr
 *
 * @brief a shared pointer with a deleter which can be changed at run-time.
 *
 */
template <typename T>
class pv_shared_ptr
{
public:
	typedef __impl::pv_ref_counter<T> ref_counter_t;
	typedef T                         type;
	typedef T                        *pointer;
	typedef                     void(*deleter)(pointer);

	pv_shared_ptr()
	{
		_ref_count = new __impl::pv_ref_counter<type>(nullptr, nullptr);
	}

	explicit pv_shared_ptr(pointer p)
	{
		_ref_count = new __impl::pv_ref_counter<type>(p, nullptr);
	}

	pv_shared_ptr(pointer p, deleter d)
	{
		_ref_count = new __impl::pv_ref_counter<type>(p, d);
	}

	pv_shared_ptr(const pv_shared_ptr<type> &rhs)
	{
		rhs._ref_count->inc_count();
		_ref_count = rhs._ref_count;
	}

	~pv_shared_ptr()
	{
		reset();
	}

	void reset()
	{
		internal_reset();
		_ref_count = new __impl::pv_ref_counter<type>();
	}

	pv_shared_ptr<type> &operator=(const pv_shared_ptr<type> &rhs)
	{
		internal_reset();
		rhs._ref_count->inc_count();
		_ref_count = rhs._ref_count;
		return *this;
	}

	operator bool() const
	{
		return (_ref_count->get() != nullptr);
	}

	pointer get() const
	{
		return _ref_count->get();
	}

	T &operator*() const
	{
		return *(_ref_count->get());
	}

	T *operator->() const
	{
		return _ref_count->get();
	}

	void set_deleter(deleter d)
	{
		_ref_count->set_deleter(d);
	}

	deleter get_deleter() const
	{
		return _ref_count->get_deleter();
	}

	uint32_t use_count() const
	{
		return _ref_count->use_count();
	}

private:
	void internal_reset()
	{
		_ref_count->dec_count();
		if (_ref_count->use_count() == 0) {
			delete _ref_count;
		}
	}

private:
	ref_counter_t *_ref_count;
};

template <class T, class U>
bool operator==(const pv_shared_ptr<T>& lhs, const pv_shared_ptr<U>& rhs)
{
	return (lhs.get() == rhs.get());
}

template <class T, class U>
bool operator!=(const pv_shared_ptr<T>& lhs, const pv_shared_ptr<U>& rhs)
{
	return (lhs.get() != rhs.get());
}

template <class T, class U>
bool operator<(const pv_shared_ptr<T>& lhs, const pv_shared_ptr<U>& rhs)
{
	return (lhs.get() < rhs.get());
}

template <class T, class U>
bool operator<=(const pv_shared_ptr<T>& lhs, const pv_shared_ptr<U>& rhs)
{
	return (lhs.get() <= rhs.get());
}

template <class T, class U>
bool operator>(const pv_shared_ptr<T>& lhs, const pv_shared_ptr<U>& rhs)
{
	return (lhs.get() > rhs.get());
}

template <class T, class U>
bool operator>=(const pv_shared_ptr<T>& lhs, const pv_shared_ptr<U>& rhs)
{
	return (lhs.get() >= rhs.get());
}

template <typename T>
pv_shared_ptr<T> make_shared(T *t)
{
	return pv_shared_ptr<T>(t);
}

template <typename T, typename D>
pv_shared_ptr<T> make_shared(T *t, D d)
{
	return pv_shared_ptr<T>(t, d);
}

template <class T, class U, class V>
std::basic_ostream<U, V>& operator<<(std::basic_ostream<U, V>& os, const pv_shared_ptr<T>& rhs)
{
	return os << rhs.get();
}

}

#endif // PVCORE_SHAREDPOINTER_H
