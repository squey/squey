#include <pvkernel/core/PVPython.h>

PVCore::PVPythonInitializer::PVPythonInitializer()
{
	// Do not let python catch the signals !
	Py_InitializeEx(0);
	PyEval_InitThreads();
	python_main = boost::python::import("__main__");
	python_main_namespace = boost::python::extract<boost::python::dict>(python_main.attr("__dict__"));
	mainThreadState = PyEval_SaveThread();
}


PVCore::PVPythonInitializer::~PVPythonInitializer()
{
	PyEval_RestoreThread(mainThreadState);
	Py_Finalize();
}

PVCore::PVPythonInitializer& PVCore::PVPythonInitializer::get()
{
	static PVCore::PVPythonInitializer PVPyInit;

	return PVPyInit;
}

