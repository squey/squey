//! \file PVPlottingFunction.cpp
//! $Id: PVPlottingFunction.cpp 2495 2011-04-25 13:43:42Z psaade $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QStringList>
#include <QRegExp>

#include <picviz/PVPlottingFunction.h>

/******************************************************************************
 *
 * Picviz::PVPlottingFunction::PVPlottingFunction
 *
 *****************************************************************************/
Picviz::PVPlottingFunction::PVPlottingFunction(QString filename)
{
	QString plugin_name(filename);

	lib = new QLibrary(filename);
	userdata = NULL;

	// Type and mode are exacted directly from the filename
#ifdef WIN32
        plugin_name.remove(QRegExp(".*function_plotting_"));
        plugin_name.remove(QRegExp("\\.dll$"));
#else
	plugin_name.remove(QRegExp(".*libfunction_plotting_"));
	plugin_name.remove(QRegExp("\\.so$"));
#endif

	QStringList tmplist = plugin_name.split("_");
	type = tmplist[0];
	mode = tmplist[1];



	init_func = (plotting_init_func) lib->resolve(picviz_plotting_init_func_string);
	if (!init_func) {
		PVLOG_ERROR("Error: %s\n", 
		        lib->errorString().toUtf8().data());
		return;
	}
	init_func();

	terminate_func = (plotting_terminate_func) lib->resolve(picviz_plotting_terminate_func_string);
	if (!terminate_func) {
		PVLOG_ERROR("Error: %s\n", 
		        lib->errorString().toUtf8().data());
		return;
	}

	exec_func = (plotting_exec_func) lib->resolve(picviz_plotting_exec_func_string);
	if (!exec_func) {
		PVLOG_ERROR("Error: %s\n", 
		        lib->errorString().toUtf8().data());
		return;
	}

	test_func = (plotting_test_func) lib->resolve(picviz_plotting_test_func_string);
	// We don't test because we don't care if this function does not exists
}

/******************************************************************************
 *
 * Picviz::PVPlottingFunction::~PVPlottingFunction
 *
 *****************************************************************************/
Picviz::PVPlottingFunction::~PVPlottingFunction()
{

}
