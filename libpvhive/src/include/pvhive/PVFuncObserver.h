/**
 * \file PVFuncObserver.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef LIBPVHIVE_PVFUNCOBSERVER_H
#define LIBPVHIVE_PVFUNCOBSERVER_H

#include <cassert>
#include <pvkernel/core/PVFunctionTraits.h>
#include <pvhive/PVObserverObjectBase.h>
#include <QSemaphore>
#include <QObject>

namespace PVHive {

class PVHive;

class PVFuncObserverBase: public PVObserverObjectBase
{
public:
	virtual ~PVFuncObserverBase();

	void do_about_to_be_updated(const void* args) const { do_about_to_be_updated_impl(args); }
	void do_update(const void* args) const { do_update_impl(args); }

protected:
	virtual void do_about_to_be_updated_impl(const void* args) const { about_to_be_updated(args); }
	virtual void do_update_impl(const void* args) const { update(args); }

public:
	virtual void about_to_be_updated(const void*) const {}
	virtual void update(const void*) const {}

protected:
	void* _f;
};

}

namespace __impl
{

// Qt signals/slots model doesn't support template classes, that's why we need a non-template base class for using signals/slots to access the Qt thread.
// Plus, Qt is confused by PVHive being a namespace and a class, so the implementation of PVFuncObservelSignalBase is directly put in __impl.

class PVFuncObserverSignalBase: public QObject, public PVHive::PVFuncObserverBase
{
	Q_OBJECT;

public:
	PVFuncObserverSignalBase();
	virtual ~PVFuncObserverSignalBase() {};

protected:
	virtual void do_about_to_be_updated_impl(const void*) const;
	virtual void do_update_impl(const void*) const;

public:
	virtual void about_to_be_updated(const void*) const  {}
	virtual void update(const void*) const {}

	virtual void call_about_to_be_updated_with_casted_args(const void*) const = 0;
	virtual void call_update_with_casted_args(const void*) const = 0;

private slots:
	void about_to_be_refreshed_slot(const void*) const;
	void refresh_slot(const void*) const;

signals:
	void about_to_be_refreshed_signal(const void*) const;
	void refresh_signal(const void*) const;
};

}

namespace PVHive {

template <class B, class T, class F, F f>
class PVFuncObserverTemplatedBase : public B
{
	friend class PVHive;

public:
	typedef B observer_type;
	typedef F f_type;
	constexpr static f_type bound_function = f;
	typedef PVCore::PVTypeTraits::function_traits<f_type> f_traits;
	typedef typename f_traits::arguments_type arguments_type;

protected:
	virtual void call_about_to_be_updated_with_casted_args(const void* args) const
	{
		arguments_type* casted_args = (arguments_type*) args;
		about_to_be_updated(*(casted_args));
		delete casted_args;
	}
	virtual void call_update_with_casted_args(const void* args) const
	{
		arguments_type* casted_args = (arguments_type*) args;
		update(*(casted_args));
		delete casted_args;
	}

public:
	virtual void about_to_be_updated(const arguments_type&) const {}
	virtual void update(const arguments_type&) const {}
};

/**
 * @class PVFuncObserver
 *
 * A template class to specify observers on a given type/class, filtered by a function of this class.
 *
 * All subclasses must implements PVObserverBase::update() and/or PVObserverBase::about_to_be_updated()
 *
 */
template <class T, class F, F f>
class PVFuncObserver : public PVFuncObserverTemplatedBase<PVFuncObserverBase, T, F, f>
{
};

/**
 * @class PVFuncObserverSignal
 *
 * A Qt compliant version of PVFuncObserver.
 *
 * All subclasses must implements PVObserverBase::update() and/or PVObserverBase::about_to_be_updated()
 *
 */
template <class T, class F, F f>
class PVFuncObserverSignal : public PVFuncObserverTemplatedBase<__impl::PVFuncObserverSignalBase, T, F, f>
{
};

}

#endif // LIBPVHIVE_PVOBSERVER_H
