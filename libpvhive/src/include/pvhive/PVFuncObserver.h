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
	void do_about_to_be_updated(const void* args) const { do_about_to_be_updated_split(args); }
	void do_update(const void* args) const { do_update_split(args); }

	virtual void do_about_to_be_updated_split(const void* args) const { about_to_be_updated(args); }
	virtual void do_update_split(const void* args) const { update(args); }

	virtual void about_to_be_updated(const void* /*args*/) const  {}
	virtual void update(const void* /*args*/) const {}

protected:
	void* _f;
};

}

namespace __impl
{

class PVFuncObserverQtBase: public QObject, public PVHive::PVFuncObserverBase
{
	Q_OBJECT;

public:
	PVFuncObserverQtBase();
	virtual ~PVFuncObserverQtBase() {};

protected:
	virtual void do_about_to_be_updated_split(const void*) const;
	virtual void do_update_split(const void*) const;

	virtual void about_to_be_updated(const void*) const  {}
	virtual void update(const void*) const {}

	virtual void about_to_be_updated_cast(const void*) const {}; // =0;
	virtual void update_cast(const void*) const {}; // =0;

private slots:
	void about_to_be_refreshed_slot(const void*) const;
	void refresh_slot(const void*) const;

signals:
	void about_to_be_refreshed_signal(const void*) const;
	void refresh_signal(const void*) const;
};

}

namespace PVHive {

/**
 * @class PVFuncObserver
 *
 * A template class to specify observers on a given type/class, filtered by a function of this class.
 *
 * All subclasses must implements PVObserverBase::update() and/or PVObserverBase::about_to_be_updated()
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

	virtual void about_to_be_updated_cast(const void* args) const { about_to_be_updated( *((arguments_type*) args)); }
	virtual void update_cast(const void* args) const
	{
		arguments_type* real_args = (arguments_type*) args;
		update(*(real_args));
		delete real_args;
	}

	virtual void about_to_be_updated(const arguments_type& /*args*/) const {}
	virtual void update(const arguments_type& /*args*/) const {}

public:
	friend class PVHive;

public:
	PVFuncObserver()
	{
		_f = (void*) f;
	}
};

template <class T, class F, F f>
class PVFuncObserverSignal : public __impl::PVFuncObserverQtBase
{
public:
	typedef F f_type;
	constexpr static f_type bound_function = f;
	typedef PVCore::PVTypeTraits::function_traits<f_type> f_traits;
	typedef typename f_traits::arguments_type arguments_type;

	virtual void about_to_be_updated_cast(const void* args) const { about_to_be_updated( *((arguments_type*) args)); }
	virtual void update_cast(const void* args) const
	{
		arguments_type* real_args = (arguments_type*) args;
		update(*(real_args));
		delete real_args;
	}

	virtual void about_to_be_updated(const arguments_type& /*args*/) const {}
	virtual void update(const arguments_type& /*args*/) const {}

public:
	friend class PVHive;

public:
	PVFuncObserverSignal()
	{
		_f = (void*) f;
	}
};

}

#endif // LIBPVHIVE_PVOBSERVER_H
