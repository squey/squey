#include <pvkernel/core/PVPython.h>
#include <picviz/PVSource.h>

#include "PVAxisComputationPython.h"

Picviz::PVAxisComputationPython::PVAxisComputationPython(PVCore::PVArgumentList const& args)
{
	INIT_FILTER(PVAxisComputationPython, args);
}

DEFAULT_ARGS_FILTER(Picviz::PVAxisComputationPython)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("script", "Script")] = QString("");
	return args;
}

bool Picviz::PVAxisComputationPython::operator()(PVSource* src)
{
	QString script = args["script"].toString();

	PVCore::PVPythonInitializer& python = PVCore::PVPythonInitializer::get();
	PVCore::PVPythonLocker lock;

	boost::python::list out_v;
	boost::python::dict py_ns = python.python_main_namespace.copy();

	try {
		PVRush::PVNrawPython python_nraw(&source->get_rushnraw_parent());
		py_ns["nraw"] = python_nraw;
		py_ns["out_values"] = out_v;

		boost::python::exec(qPrintable(code_edit->toPlainText()), py_ns, py_ns);
	}    
	catch (boost::python::error_already_set const&)
	{    
		PyErr_Print();
		return false;
	}

	boost::python::stl_input_iterator<PVCore::PVUnicodeString> begin(out_v), end; 
	src->add_column(begin, end, axis);

	return true;
}
