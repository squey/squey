/**
 * \file PVWeakPointer.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef PVWEAKPOINTER_H_
#define PVWEAKPOINTER_H_

#include <pvkernel/core/PVSharedPointer.h>
#include <pvkernel/core/PVCountedBase.h>

namespace PVCore
{

template <typename T>
class PVSharedPtr;

template <typename T>
class PVWeakPtr
{
public:
	template<typename X> friend class PVSharedPtr;

public:
	typedef T type;
    typedef PVWeakCounter weak_counter_t;
    typedef type* pointer;

public:
	PVWeakPtr(): _weak_count(), _px(nullptr)
	{
	}

	PVWeakPtr(PVWeakPtr<T> & r): _weak_count(r._weak_count), _px(r._px)
	{
	}

	PVWeakPtr(PVSharedPtr<T> & r): _weak_count(r._shared_count), _px(r._px)
	{
	}

	template <typename Y>
	PVWeakPtr& operator=(PVWeakPtr<Y> const & r)
	{
		_weak_count = r._weak_count;
		_px = static_cast<Y*>(r._px);
		return *this;
	}

	template <typename Y>
	PVWeakPtr& operator=(PVSharedPtr<Y> const & r)
	{
		_weak_count = r._shared_count;
		_px = static_cast<Y*>(r._px);
		return *this;
	}

    PVSharedPtr<T> lock() const
	{
		return PVSharedPtr<T>(*this);
	}

	long use_count() const
	{
		return _weak_count.use_count();
	}

	bool expired() const
	{
		return _weak_count.use_count() == 0;
	}

	inline void set_deleter(typename PVSharedPtr<T>::deleter d)
	{
		lock().set_deleter(d);
	}

	void reset()
	{
		PVWeakPtr<T>().swap(*this);
	}

	void swap(PVWeakPtr<T> & other)
	{
		//_weak_count.swap(other._weak_count);
		std::swap(_weak_count, other._weak_count);
	}

private:
	weak_counter_t _weak_count;
	pointer _px;
};

}

#endif /* PVWEAKPOINTER_H_ */
