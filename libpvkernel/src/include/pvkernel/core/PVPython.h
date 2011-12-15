#ifndef PVPYTHON_H
#define PVPYTHON_H

#ifdef ENABLE_PYTHON_SUPPORT

// AG: to investigate: pvbase/general.h, which is included by core/general.h,
// seems to conflict with boost/python.hpp. If this header is after general.h,
// them you have a compile error !!
#ifdef PVBASE_GENERAL_H
#error It seems that including pvbase/general.h before boost/python.hpp gives compilation errors. Be careful to have boost/python.hpp as the first header.
#endif
#include <boost/python.hpp>

#include <boost/thread/mutex.hpp>
#include <pvkernel/core/general.h>

#include <QString>

namespace PVCore {

class PVPythonClassDecl;

class LibKernelDecl PVPythonInitializer
{
	friend class PVPythonClassRegister;
public:
	~PVPythonInitializer();
private:
	PVPythonInitializer();
	PVPythonInitializer(const PVPythonInitializer&) { }
public:
	static PVPythonInitializer& get();
public:
	boost::python::object python_main;
	boost::python::dict python_main_namespace;
protected:
	static void register_class(PVPythonClassDecl const& c);
private:
	static std::list<PVPythonClassDecl*>& get_class_list();
private:
	PyThreadState* mainThreadState;
	static boost::mutex _python_init;
	static PVPythonInitializer* g_python;
};

class PVPythonLocker
{
public:
	PVPythonLocker()
	{
		PVLOG_INFO("PVPythonLocker construct\n");
		_state = PyGILState_Ensure();
	};
	~PVPythonLocker()
	{
		PVLOG_INFO("PVPythonLocker destruct\n");
		PyGILState_Release(_state);
	}
private:
	PVPythonLocker(const PVPythonLocker&) { }
private:
	PyGILState_STATE _state;
};

namespace PVPython {
	extern QString LibKernelDecl get_list_index_as_qstring(boost::python::list pylist, int index);
}

}

#endif

#endif //PVPYTHON_H
