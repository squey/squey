#include <pvkernel/core/PVPython.h>
#include <pvkernel/rush/PVNrawPython.h>
#include <pvkernel/rush/PVNraw.h>

#include <picviz/PVSource.h>

#include <boost/python/stl_iterator.hpp>

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

bool Picviz::PVAxisComputationPython::operator()(PVRush::PVNraw* nraw)
{
	QString script = _args["script"].toString();

	PVCore::PVPythonInitializer& python = PVCore::PVPythonInitializer::get();
	PVCore::PVPythonLocker lock;

	boost::python::list out_v;
	boost::python::dict py_ns = python.python_main_namespace.copy();

	try {
		PVRush::PVNrawPython python_nraw(nraw);
		py_ns["nraw"] = python_nraw;
		py_ns["out_values"] = out_v;

		boost::python::exec(script.toUtf8().constData(), py_ns, py_ns);
	}    
	catch (boost::python::error_already_set const&)
	{    
		PyErr_Print();
		return false;
	}

	boost::python::stl_input_iterator<PVCore::PVUnicodeString> begin(out_v), end; 
	nraw->add_column(begin, end);

	return true;
}
