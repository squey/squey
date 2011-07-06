//! \file PVClassLibrary.h
//! $Id: PVClassLibrary.h 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PVCORE_PVCLASSLIBRARY_H
#define PVCORE_PVCLASSLIBRARY_H

#include <pvcore/general.h>

#include <QHash>
#include <QString>

namespace PVCore {

// This is used to register the class T as RegAs 
// AG: WARNING: there is *no* LibExport and this is *wanted* !
//              check the wiki for more informations
template<class RegAs>
class PVClassLibrary {
public:
	// PF is a shared pointer to a registered class's base class
	typedef typename RegAs::p_type PF;
	typedef QHash<QString,PF> list_classes;
	typedef PVClassLibrary<RegAs> C;

private:
	PVClassLibrary()
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
	void register_class(QString const& name, T const& f)
	{
		PF pf = f.template clone<T>();
		_classes.insert(name, pf);
	}

	list_classes const& get_list() const { return _classes; }

	// A shared pointer is returned, which means that parameters can be saved accross this
	// saved pointer. If this is not wanted, a clone can be made thanks to the clone() method
	PF get_class_by_name(QString const& name)
	{
		if (!_classes.contains(name))
			return PF();
		return _classes[name];
	}

private:
	list_classes _classes;
};

class LibExport PVClassLibraryLibLoader {
public:
	static bool load_class(QString const& path);
	static int load_class_from_dir(QString const& pluginsdir, QString const& prefix);
	static int load_class_from_dirs(QStringList const& pluginsdirs, QString const& prefix);
};

#define REGISTER_CLASS_AS(name, T, RegAs) \
	PVCore::PVClassLibrary<RegAs>::get().register_class<T>(name, T());
#define REGISTER_CLASS(name, T) REGISTER_CLASS_AS(name, T, T::RegAs)
	
#define LIB_CLASS(T) \
	PVCore::PVClassLibrary<T::RegAs>
}

#endif
