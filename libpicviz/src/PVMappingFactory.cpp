//! \file PVMappingFactory.cpp
//! $Id: PVMappingFactory.cpp 2736 2011-05-12 15:12:28Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QDir>
#include <QStringList>
#include <QString>

#include <picviz/PVMappingFactory.h>
#include <picviz/plugins.h>

Picviz::PVMappingFactory::PVMappingFactory()
{
	register_all();
}

Picviz::PVMappingFactory::~PVMappingFactory()
{

}

int Picviz::PVMappingFactory::register_all()
{
	QDir dir;
	QString pluginsdirs;
	QStringList filters;
	QStringList files;

	pluginsdirs = QString(picviz_plugins_get_functions_dir());

	dir = QDir(pluginsdirs);
	PVLOG_INFO("Reading mapping functions plugins directory: %s\n", pluginsdirs.toUtf8().data());

#ifdef WIN32	
	filters << "*function_mapping_*.dll";
#else
	filters << "*function_mapping_*.so";
#endif
	dir.setNameFilters(filters);
	
	files = dir.entryList();
	QStringListIterator filesIterator(files);
	while (filesIterator.hasNext()) {
		PVMappingFunction *mf = new PVMappingFunction(dir.absoluteFilePath(filesIterator.next()));
		plugins[mf->type][mf->mode] = mf;
		if (mf->lib) {
			PVLOG_INFO("Successfully added plugin type '%s', mode '%s'\n", mf->type.toUtf8().data(), mf->mode.toUtf8().data());
		}
	}
	
	return 0;
}

