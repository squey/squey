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

namespace __impl
{

template <typename T>
class PVSharedCounter;

template <typename T>
class PVWeakCounter
{
public:
	template<typename X> friend class PVSharedCounter;
    friend class PVCountedBase<T>;

public:
	typedef T type;
	typedef T* pointer;

public:

    PVWeakCounter(pointer p = nullptr): _px(p), _counted_base()
    {
    }

    PVWeakCounter(PVWeakCounter<T> const & r) : _px(r._px), _counted_base(r._counted_base)
    {
        if(_counted_base != nullptr) {
        	_counted_base->weak_add_ref();
        }
    }

    PVWeakCounter(PVSharedCounter<T>& r) : _px(r._px), _counted_base(r._counted_base)
	{
		if(_counted_base != nullptr) {
			_counted_base->weak_add_ref();
		}
	}

    ~PVWeakCounter()
    {
        if(_counted_base != nullptr) {
        	_counted_base->weak_release();
        }
    }

    template <typename Y>
    PVWeakCounter<T>& operator=(PVWeakCounter<Y> const& r)
    {
    	PVCountedBase<Y>* tmp = r._counted_base;

        if(tmp != _counted_base)
        {
            if(tmp != nullptr) {
            	tmp->weak_add_ref();
            }
            if(_counted_base != nullptr) {
            	_counted_base->weak_release();
            }
            _counted_base = tmp;
            _px = r._px;
        }

        return *this;
    }

    template <typename Y>
    PVWeakCounter<T>& operator=(PVSharedCounter<Y> const& r)
	{
    	PVCountedBase<Y>* tmp = r._counted_base;

		if(tmp != (PVCountedBase<Y>*) _counted_base)
		{
			if(tmp != nullptr) {
				tmp->weak_add_ref();
			}
			if(_counted_base != nullptr) {
				_counted_base->weak_release();
			}
			_counted_base = (PVCountedBase<T>*) tmp;
			_px = r._px;
		}

		return *this;
	}

    void swap(PVWeakCounter<T> & r)
    {
    	pointer* tmp_px = r._px;
    	PVCountedBase<T> * tmp_cb = r._counted_base;
        r._counted_base = _counted_base;
        r._px = _px;
        _counted_base = tmp_cb;
        _px = tmp_px;
    }

    long use_count() const
    {
        return !empty() ? _counted_base->use_count(): 0;
    }

    bool empty() const
    {
        return _counted_base == nullptr;
    }

    friend inline bool operator==(PVWeakCounter<T> const & a, PVWeakCounter<T> const & b)
    {
        return a._counted_base == b._counted_base;
    }

public://private:
    pointer _px;
	PVCountedBase<T>* _counted_base;
};


}

template <typename T>
class PVWeakPtr
{
public:
	template<typename X> friend class PVSharedPtr;

public:
	typedef T type;
    typedef __impl::PVWeakCounter<type> weak_counter_t;
    typedef type* pointer;

public:
	PVWeakPtr(): _weak_count()
	{
	}

	PVWeakPtr(PVWeakPtr<T> & r): _weak_count(r._weak_count)
	{
	}

	PVWeakPtr(PVSharedPtr<T> & r): _weak_count(r._shared_count)
	{
	}

	template <typename Y>
	PVWeakPtr& operator=(PVWeakPtr<Y> const & r)
	{
		_weak_count = r._weak_count;
		return *this;
	}

	template <typename Y>
	PVWeakPtr& operator=(PVSharedPtr<Y> const & r)
	{
		_weak_count = r._shared_count;
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

	void reset()
	{
		PVWeakPtr<T>().swap(*this);
	}

	void swap(PVWeakPtr<T> & other)
	{
		_weak_count.swap(other._weak_count);
	}

public://private:
	weak_counter_t _weak_count;
};

}

#endif /* PVWEAKPOINTER_H_ */
