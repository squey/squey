//! \file PVFilterLibrary.h
//! $Id: PVFilterLibrary.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVFILTER_PVFILTERLIBRARY_H
#define PVFILTER_PVFILTERLIBRARY_H

#include <pvcore/general.h>

#include <QHash>
#include <QString>

namespace PVFilter {

// This class act as a singleton and a factory for filters
// Each filter register to this class thanks to the IMPL_FILTER macro !

// This is used to register the filter T as FilterT
// AG: WARNING: there is *no* LibExport and this is *wanted* !
//              check the wiki for more informations
template<class FilterT>
class PVFilterLibrary {
public:
	// PF is a shared pointer to a filter's base class
	typedef typename FilterT::p_type PF;
	typedef QHash<QString,PF> list_filters;
	typedef PVFilterLibrary<FilterT> C;

private:
	PVFilterLibrary()
	{
	}

public:
	static C& get()
	{
		static C obj;
		return obj;
	}

public:
	template<class T>
	void register_filter(QString const& name, T const& f)
	{
		PF pf = f.template clone<T>();
		_filters.insert(name, pf);
	}

	list_filters const& get_list() const { return _filters; }

	// A shared pointer is returned, which means that parameters can be saved accross this
	// saved pointer. If this is not wanted, a clone can be made thanks to the clone() method
	PF get_filter_by_name(QString const& name)
	{
		if (!_filters.contains(name))
			return PF();
		return _filters[name];
	}

private:
	list_filters _filters;
};

class LibExport PVFilterLibraryLibLoader {
public:
	static bool load_library(QString const& path);
	static int load_library_from_dir(QString const& pluginsdir, QString const& prefix);
	static int load_library_from_dirs(QStringList const& pluginsdirs, QString const& prefix);
};

#define REGISTER_FILTER_AS(name, T, FilterT) \
	PVFilter::PVFilterLibrary<FilterT>::get().register_filter<T>(name, T());
#define REGISTER_FILTER(name, T) REGISTER_FILTER_AS(name, T, T::FilterT)
	
#define REGISTER_FILTER_AS_WITH_ARGS(name, T, FilterT, args) \
	PVFilter::PVFilterLibrary<FilterT>::get().register_filter<T>(name, T(args));
#define REGISTER_FILTER_WITH_ARGS(name, T, args) REGISTER_FILTER_AS_WITH_ARGS(name, T, T::FilterT, args)

#define LIB_FILTER(T) \
	PVFilter::PVFilterLibrary<T::FilterT>
}

#endif
