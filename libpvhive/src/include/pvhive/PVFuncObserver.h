#ifndef LIBPVHIVE_PVFUNCOBSERVER_H
#define LIBPVHIVE_PVFUNCOBSERVER_H

#include <cassert>
#include <pvkernel/core/PVFunctionTraits.h>
#include <pvhive/PVObserverObjectBase.h>

namespace PVHive {

class PVHive;

class PVFuncObserverBase: public PVObserverObjectBase
{
public:
	virtual ~PVFuncObserverBase();

protected:
	void* _f;
};

/**
 * @class PVFuncObserver
 *
 * A template class to specify observers on a given type/class, filtered by a function of this class.
 *
 * All subclasses must implements PVObserverBase::update()
 * 
 */
template <class T, class F, F f>
class PVFuncObserver : public PVFuncObserverBase
{
public:
	typedef F f_type;
	constexpr static f_type bound_function = f;
	typedef PVCore::PVTypeTraits::function_traits<f_type> f_traits;
	typedef typename f_traits::arguments_type arguments_type;

public:
	friend class PVHive;

public:
	PVFuncObserver()
	{
		_f = (void*) f;
	}

public:
	virtual void update(arguments_type const& args) const = 0;
};


}

#endif // LIBPVHIVE_PVOBSERVER_H
