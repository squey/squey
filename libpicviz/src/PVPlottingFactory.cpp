//! \file PVPlottingFactory.cpp
//! $Id: PVPlottingFactory.cpp 2736 2011-05-12 15:12:28Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QDir>
#include <QStringList>
#include <QString>

#include <pvkernel/core/debug.h>

#include <picviz/PVPlottingFactory.h>
#include <picviz/plugins.h>


/******************************************************************************
 *
 * Picviz::PVPlottingFactory::PVPlottingFactory
 *
 *****************************************************************************/
Picviz::PVPlottingFactory::PVPlottingFactory()
{
	register_all();
}

/******************************************************************************
 *
 * Picviz::PVPlottingFactory::~PVPlottingFactory
 *
 *****************************************************************************/
Picviz::PVPlottingFactory::~PVPlottingFactory()
{

}

/******************************************************************************
 *
 * Picviz::PVPlottingFactory::register_all
 *
 *****************************************************************************/
int Picviz::PVPlottingFactory::register_all()
{
	QDir        dir;
	QString     pluginsdirs;
	QStringList filters;
	QStringList files;

	pluginsdirs = QString(picviz_plugins_get_functions_dir());

	dir = QDir(pluginsdirs);
	PVLOG_INFO("Reading plotting functions plugins directory: %s\n", pluginsdirs.toUtf8().data());

#ifdef WIN32	
	filters << "*function_plotting_*.dll";
#else
	filters << "*function_plotting_*.so";
#endif

	dir.setNameFilters(filters);
	
	files = dir.entryList();
	QStringListIterator filesIterator(files);
	while (filesIterator.hasNext()) {
		PVPlottingFunction *mf = new PVPlottingFunction(dir.absoluteFilePath(filesIterator.next()));
		plugins[mf->type][mf->mode] = mf;
		if (mf->lib) {
			PVLOG_INFO("Successfully added plugin type '%s', mode '%s'\n", mf->type.toUtf8().data(), mf->mode.toUtf8().data());
		}
	}
	
	return 0;
}
