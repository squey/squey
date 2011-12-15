#ifdef ENABLE_PYTHON_SUPPORT

#include <pvkernel/core/PVPython.h>
#include <pvkernel/core/PVPythonClassDecl.h>

boost::mutex PVCore::PVPythonInitializer::_python_init;
PVCore::PVPythonInitializer* PVCore::PVPythonInitializer::g_python = NULL;

PVCore::PVPythonInitializer::PVPythonInitializer()
{
	// Do not let python catch the signals !
	Py_InitializeEx(0);
	PyEval_InitThreads();
	python_main = boost::python::import("__main__");
	python_main_namespace = boost::python::extract<boost::python::dict>(python_main.attr("__dict__"));

	// Expose "exposable" class to Python
	std::list<PVPythonClassDecl*>::iterator it;
	for (it = get_class_list().begin(); it != get_class_list().end(); it++) {
		(*it)->declare();
	}

	mainThreadState = PyEval_SaveThread();
}


PVCore::PVPythonInitializer::~PVPythonInitializer()
{
	PyEval_RestoreThread(mainThreadState);
	Py_Finalize();

	std::list<PVPythonClassDecl*>::iterator it;
	for (it = get_class_list().begin(); it != get_class_list().end(); it++) {
		delete (*it);
	}
}

PVCore::PVPythonInitializer& PVCore::PVPythonInitializer::get()
{
	boost::mutex::scoped_lock lock(_python_init);
	if (g_python == NULL) {
		g_python = new PVCore::PVPythonInitializer();
	}

	return *g_python;
}

QString PVCore::PVPython::get_list_index_as_qstring(boost::python::list pylist, int index)
{
	boost::python::extract<const char*> extract_str(pylist[index]);
	QString value;
	if (extract_str.check()) {
		value = QString::fromUtf8(extract_str());
	} else {
		PVLOG_DEBUG("%s returning an empty string!\n", __FUNCTION__);
		value = QString("");
	}	

	return value;
}

std::list<PVCore::PVPythonClassDecl*>& PVCore::PVPythonInitializer::get_class_list()
{
	static std::list<PVCore::PVPythonClassDecl*> list;
	return list;
}

void PVCore::PVPythonInitializer::register_class(PVPythonClassDecl const& c)
{
	get_class_list().push_back(c.clone());
}

PVCore::PVPythonClassRegister::PVPythonClassRegister(PVPythonClassDecl const& c)
{
	PVPythonInitializer::register_class(c);
}

#endif
