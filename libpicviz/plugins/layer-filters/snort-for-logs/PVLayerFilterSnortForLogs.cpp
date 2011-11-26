//! \file PVLayerFilterSnortForLogs.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include "PVLayerFilterSnortForLogs.h"

#include <pvkernel/core/PVCheckBoxType.h>

#include <picviz/PVView.h>

#include <QDir>
#include <QString>

/******************************************************************************
 *
 * Picviz::PVLayerFilterSnortForLogs::PVLayerFilterSnortForLogs
 *
 *****************************************************************************/
Picviz::PVLayerFilterSnortForLogs::PVLayerFilterSnortForLogs(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterSnortForLogs, l);

	PVCore::PVPythonInitializer& python = PVCore::PVPythonInitializer::get();
	PVCore::PVPythonLocker locker;

	_python_own_namespace = python.python_main_namespace.copy();

	try {
		// Load our script
		boost::python::exec_file("snort-rules.py", _python_own_namespace, _python_own_namespace);
		snort_rules = boost::python::extract<boost::python::list>(_python_own_namespace["snort_rules"]);
		rules_number = boost::python::len(snort_rules);
		PVLOG_INFO("Number of Snort rules loaded: %d\n", rules_number);
	}
	catch (boost::python::error_already_set const&)
	{
		//std::string exc = convert_py_exception_to_string();
		PyErr_Print();
		// //throw PVPythonExecException(python_file, exc.c_str());
		// throw PVPythonExecException(python_file, "");
	}

}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterSnortForLogs)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterSnortForLogs)
{
	PVCore::PVArgumentList args;

	args["Web client"].setValue(PVCore::PVCheckBoxType(true));
	args["Web servers"].setValue(PVCore::PVCheckBoxType(true));
	args["Worms"].setValue(PVCore::PVCheckBoxType(true));
	args["Other types"].setValue(PVCore::PVCheckBoxType(true));

	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterSnortForLogs::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterSnortForLogs::operator()(PVLayer& in, PVLayer &out)
{
	bool web_clients = _args["Web client"].value<PVCore::PVCheckBoxType>().get_checked();
	bool web_servers = _args["Web servers"].value<PVCore::PVCheckBoxType>().get_checked();
	bool worms = _args["Worms"].value<PVCore::PVCheckBoxType>().get_checked();
	bool other_types = _args["Other Types"].value<PVCore::PVCheckBoxType>().get_checked();

	PVRush::PVNraw::nraw_table const& nraw = _view->get_qtnraw_parent();

	out.get_selection().select_none();

	for (int i=0; i < rules_number; i++) {
		boost::python::dict snort_alert = boost::python::extract<boost::python::dict>(snort_rules[i]);
		boost::python::list match_groups = boost::python::extract<boost::python::list>(snort_alert["match_groups"]);
		const char *current_sid = boost::python::extract<const char*>(snort_alert["sid"]);

		for (int j=0; j < boost::python::len(match_groups); j++) {
			// PVLOG_INFO("WE ARE IN MATCH GROUP %d of sid %s\n", j, current_sid);
			boost::python::list current_group = boost::python::extract<boost::python::list>(match_groups[j]);

			for (int k=0; k < boost::python::len(current_group); k++) {
				// PVLOG_INFO("Reading our group values (k=%d)\n", k);
				boost::python::list key_value = boost::python::extract<boost::python::list>(current_group[k]);
				QString key = PVCore::PVPython::get_list_index_as_qstring(key_value, 0);
				QString value = PVCore::PVPython::get_list_index_as_qstring(key_value, 1);
				if (! QString::compare(QString(key), QString("content"))) {
					if (value.startsWith("User-Agent: ") && (value.length() != 12)) {
						value.remove(0, 12);
						PVLOG_INFO("(from sid:%s) This value is content and search for %s\n", current_sid, qPrintable(value));
					}      
				}
			}
		}
	}

}

IMPL_FILTER(Picviz::PVLayerFilterSnortForLogs)
