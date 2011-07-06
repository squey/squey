//! \file PVFilterLibrary.cpp
//! $Id: PVFilterLibrary.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011


#include <pvfilter/PVFilterLibrary.h>

#include <QLibrary>
#include <QDir>
#include <QStringList>




// Helper class to load external plugins

// Register function type
typedef void (*register_filter_func)();
#define register_filter_func_string "register_filter"

bool PVFilter::PVFilterLibraryLibLoader::load_library(QString const& path)
{
	QLibrary lib(path);
	register_filter_func rf;
	rf = (register_filter_func) lib.resolve(register_filter_func_string);
	if (!rf) {
		PVLOG_ERROR("Error while loading filter %s: %s\n", qPrintable(path), qPrintable(lib.errorString()));
		return false;
	}

	// Call the registry function
	rf();

	// And that's it !
	return true;
}

int PVFilter::PVFilterLibraryLibLoader::load_library_from_dir(QString const& pluginsdir, QString const& prefix)
{
	QDir dir(pluginsdir);
	PVLOG_INFO("Reading plugins directory: %s\n", qPrintable(pluginsdir));

	// Set directory listing filters
	QStringList filters;
#ifdef WIN32	
	filters << QString("*") + prefix + QString("_*.dll");
#else
	filters << QString("*") + prefix + QString("_*.so");
#endif
	dir.setNameFilters(filters);
	
	QStringList files = dir.entryList();
	QStringListIterator filesIterator(files);
	int count = 0;
	while (filesIterator.hasNext()) {
		QString curfile = filesIterator.next();
		if (load_library(dir.absoluteFilePath(curfile))) {
			PVLOG_INFO("Successfully loaded plugin '%s'\n", qPrintable(curfile));
			count++;
		}
	}
	return count;
}

int PVFilter::PVFilterLibraryLibLoader::load_library_from_dirs(QStringList const& pluginsdirs, QString const& prefix)
{
	int count = 0;
	for (int i = 0; i < pluginsdirs.size(); i++) {
		count += load_library_from_dir(pluginsdirs[i], prefix);
	}
	return count;
}

