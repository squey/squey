//! \file PVMandatoryMappingFunction.cpp
//! $Id: PVMandatoryMappingFunction.cpp 3090 2011-06-09 04:59:46Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QStringList>
#include <QRegExp>

#include <picviz/PVMandatoryMappingFunction.h>


Picviz::PVMandatoryMappingFunction::PVMandatoryMappingFunction(QString filename)
{
	QString plugin_name(filename);

	lib = new QLibrary(filename);
	userdata = NULL;

	// Type and mode are exacted directly from the filename
#ifdef WIN32
        plugin_name.remove(QRegExp(".*function_mandatory_mapping_"));
        plugin_name.remove(QRegExp("\\.dll$"));
#else
	plugin_name.remove(QRegExp(".*libfunction_mandatory_mapping_"));
	plugin_name.remove(QRegExp("\\.so$"));
#endif

	QStringList tmplist = plugin_name.split("_");
	name = tmplist[0];

	init_func = (mandatory_mapping_init_func) lib->resolve(picviz_mandatory_mapping_init_func_string);
	if (!init_func) {
		PVLOG_ERROR("Error: %s\n", 
		        lib->errorString().toUtf8().data());
		return;
	}
	init_func();

	terminate_func = (mandatory_mapping_terminate_func) lib->resolve(picviz_mandatory_mapping_terminate_func_string);
	if (!terminate_func) {
		PVLOG_ERROR("Error: %s\n", 
		        lib->errorString().toUtf8().data());
		return;
	}

	exec_func = (mandatory_mapping_exec_func) lib->resolve(picviz_mandatory_mapping_exec_func_string);
	if (!exec_func) {
		PVLOG_ERROR("Error: %s\n", 
		        lib->errorString().toUtf8().data());
		return;
	}

	test_func = (mandatory_mapping_test_func) lib->resolve(picviz_mandatory_mapping_test_func_string);
	// We don't test because we don't care if this function does not exists
}

Picviz::PVMandatoryMappingFunction::~PVMandatoryMappingFunction()
{

}
