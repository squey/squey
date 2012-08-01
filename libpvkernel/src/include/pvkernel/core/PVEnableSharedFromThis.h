/**
 * \file PVEnableSharedFromThis.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef PVENABLESHAREDFROMTHIS_H_
#define PVENABLESHAREDFROMTHIS_H_

#include <pvkernel/core/PVSharedPointer.h>
#include <pvkernel/core/PVWeakPointer.h>

namespace PVCore
{

template<class T>
class PVEnableSharedFromThis
{
	//template<typename X> friend class PVSharedPtr<X>::EnableSharedFromThis;

public:

	PVEnableSharedFromThis()
    {
    }

	PVEnableSharedFromThis(PVEnableSharedFromThis const &)
    {
    }

	PVEnableSharedFromThis & operator=(PVEnableSharedFromThis const &)
    {
        return *this;
    }

    ~PVEnableSharedFromThis()
    {
    }

public:

    PVSharedPtr<T> shared_from_this()
    {
    	PVSharedPtr<T> p(_weak_this);
        assert(p.get() == this);
        return p;
    }

    PVSharedPtr<T const> shared_from_this() const
    {
    	PVSharedPtr<T const> p(_weak_this);
        assert(p.get() == this);
        return p;
    }

public: // actually private, but avoids compiler template friendship issues

    // Note: invoked automatically by shared_ptr; do not call
    template<typename X, typename Y>
    void _internal_accept_owner(PVSharedPtr<X>* sp, Y* p)
    {
        if(_weak_this.expired())
        {
        	_weak_this = PVSharedPtr<X>(*sp, p);
        }
    }

private:

    PVWeakPtr<T> _weak_this;
};

}


#endif /* PVENABLESHAREDFROMTHIS_H_ */
