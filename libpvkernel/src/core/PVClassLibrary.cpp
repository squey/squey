/**
 * \file PVClassLibrary.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVClassLibrary.h>

#include <QLibrary>
#include <QDir>
#include <QStringList>


// Helper class to load external plugins

// Register function type
typedef void (*register_class_func)();
#define register_class_func_string "register_class"

bool PVCore::PVClassLibraryLibLoader::load_class(QString const& path)
{
	QLibrary lib(path);
	register_class_func rf;
	rf = (register_class_func) lib.resolve(register_class_func_string);
	if (!rf) {
		PVLOG_ERROR("Error while loading plugin %s: %s\n", qPrintable(path), qPrintable(lib.errorString()));
		return false;
	}

	// Call the registry function
	rf();

	// And that's it !
	return true;
}

int PVCore::PVClassLibraryLibLoader::load_class_from_dir(QString const& pluginsdir, QString const& prefix)
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
		QString activated_token = curfile + QString("/activated");
		int activated = pvconfig.value(activated_token, 1).toInt();
		if (activated) {
			if (load_class(dir.absoluteFilePath(curfile))) {
				PVLOG_INFO("Successfully loaded plugin '%s'\n", qPrintable(curfile));
				count++;
			}
		}
	}
	return count;
}

int PVCore::PVClassLibraryLibLoader::load_class_from_dirs(QString const& pluginsdirs, QString const& prefix)
{
	QStringList pluginsdirs_list = split_plugin_dirs(pluginsdirs);

	int count = 0;
	for (int i = 0; i < pluginsdirs_list.size(); i++) {
		count += load_class_from_dir(pluginsdirs_list[i], prefix);
	}
	return count;
}

QStringList PVCore::PVClassLibraryLibLoader::split_plugin_dirs(QString const& dirs)
{
	return dirs.split(PVCORE_DIRECTORY_SEP);
}
