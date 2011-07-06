//! \file PVMandatoryMappingFactory.cpp
//! $Id: PVMandatoryMappingFactory.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QDir>
#include <QStringList>
#include <QString>

#include <picviz/PVMandatoryMappingFactory.h>
#include <picviz/plugins.h>

Picviz::PVMandatoryMappingFactory::PVMandatoryMappingFactory()
{
	register_all();
}

Picviz::PVMandatoryMappingFactory::~PVMandatoryMappingFactory()
{

}

int Picviz::PVMandatoryMappingFactory::register_all()
{
	QDir dir;
	QString pluginsdirs;
	QStringList filters;
	QStringList files;

	pluginsdirs = QString(picviz_plugins_get_functions_dir());

	dir = QDir(pluginsdirs);
	PVLOG_INFO("Reading mandatory mapping functions plugins directory: %s\n", pluginsdirs.toUtf8().data());
#ifdef WIN32	
	filters << "*function_mandatory_mapping_*.dll";
#else
	filters << "*function_mandatory_mapping_*.so";
#endif

	dir.setNameFilters(filters);
	
	files = dir.entryList();
	QStringListIterator filesIterator(files);
	while (filesIterator.hasNext()) {
		PVMandatoryMappingFunction *mf = new PVMandatoryMappingFunction(dir.absoluteFilePath(filesIterator.next()));
		plugins << mf;
		if (mf->lib) {
			PVLOG_INFO("Successfully added plugin named '%s'\n", mf->name.toUtf8().data());
		}
	}
	
	return 0;
}

