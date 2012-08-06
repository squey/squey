/**
 * \file PVCounters.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef PVCOUNTERS_H_
#define PVCOUNTERS_H_

#include <pvkernel/core/PVSharedPointer.h>
#include <pvkernel/core/PVCountedBase.h>

namespace PVCore
{

class PVWeakCounter;

class PVSharedCounter
{
	friend class PVWeakCounter;

public:
	PVSharedCounter() : _counted_base(nullptr)
	{
	}

	template <typename T>
	PVSharedCounter(T* p)
	{
		_counted_base = new __impl::PVCountedBasePD<T*, void(*)(T*)>(p);
	}

	template <typename T>
	PVSharedCounter(T* p, void(*d)(T*))
	{
		_counted_base = new __impl::PVCountedBasePD<T*, decltype(d)>(p, d);
	}

	PVSharedCounter(PVSharedCounter const & r) : _counted_base(r._counted_base)
	{
		if(_counted_base != nullptr ) {
			_counted_base->add_ref_copy();
		}
	}

	PVSharedCounter(PVWeakCounter const & r);

	~PVSharedCounter()
	{
		 if(_counted_base != nullptr) {
			 _counted_base->release();
		 }
	}

	/*void swap(PVSharedCounter & r)
	{
		PVCountedBase* tmp = r._counted_base;
		r._counted_base = _counted_base;
		_counted_base = tmp;
	}*/

	inline long add_ref_copy()
	{
		return _counted_base->add_ref_copy();
	}

	inline void set(void* p) { _counted_base->set(p); }
	inline void* get() const { return _counted_base->get(); }

	inline void set_deleter(void* d = nullptr) { _counted_base->set_deleter(d); }
	inline void* get_deleter() const { return _counted_base->get_deleter(); }

	inline long use_count() const { return _counted_base ? _counted_base->use_count() : 0; }

	inline bool empty() const { return _counted_base == nullptr; }

private:
	PVCountedBase* _counted_base;
};

class PVWeakCounter
{
public:
	friend class PVSharedCounter;
    friend class PVCountedBase;

public:

    PVWeakCounter() : _counted_base()
    {
    }

    template <typename T>
    PVWeakCounter(PVWeakCounter const & r) : _counted_base(r._counted_base)
    {
        if(_counted_base != nullptr) {
        	_counted_base->weak_add_ref();
        }
    }

    PVWeakCounter(PVSharedCounter & r);
    ~PVWeakCounter();

    PVWeakCounter& operator=(PVWeakCounter const& r)
    {
    	PVCountedBase* tmp = r._counted_base;

        if(tmp != _counted_base)
        {
            if(tmp != nullptr) {
            	tmp->weak_add_ref();
            }
            if(_counted_base != nullptr) {
            	_counted_base->weak_release();
            }
            _counted_base = tmp;
        }

        return *this;
    }

    PVWeakCounter& operator=(PVSharedCounter const& r);

    /*void swap(PVWeakCounter & r)
    {
    	PVCountedBase* tmp = r._counted_base;
        r._counted_base = _counted_base;
        _counted_base = tmp;
    }*/

    long use_count() const
    {
        return !empty() ? _counted_base->use_count(): 0;
    }

    bool empty() const
    {
        return _counted_base == nullptr;
    }

    friend inline bool operator==(PVWeakCounter const & a, PVWeakCounter const & b)
    {
        return a._counted_base == b._counted_base;
    }

private:
	PVCountedBase* _counted_base;
};


inline PVSharedCounter::PVSharedCounter(PVWeakCounter const & r) : _counted_base(r._counted_base)
{
	if(_counted_base != nullptr && !_counted_base->add_ref_lock()) {
		_counted_base = nullptr;
	}
}

inline PVWeakCounter::PVWeakCounter(PVSharedCounter & r) :_counted_base(r._counted_base)
{
	if(_counted_base != nullptr) {
		_counted_base->weak_add_ref();
	}
}

inline PVWeakCounter::~PVWeakCounter()
{
    if(_counted_base != nullptr) {
    	_counted_base->weak_release();
    }
}

inline PVWeakCounter& PVWeakCounter::operator=(PVSharedCounter const& r)
{
	PVCountedBase* tmp = r._counted_base;

	if(tmp != _counted_base)
	{
		if(tmp != nullptr) {
			tmp->weak_add_ref();
		}
		if(_counted_base != nullptr) {
			_counted_base->weak_release();
		}
		_counted_base = tmp;
	}

	return *this;
}

}


#endif /* PVCOUNTERS_H_ */
