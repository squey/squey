#ifndef PVPYTHON_H
#define PVPYTHON_H

// AG: to investigate: pvbase/general.h, which is included by core/general.h,
// seems to conflict with boost/python.hpp. If this header is after general.h,
// them you have a compile error !!
#include <boost/python.hpp>

#include <pvkernel/core/general.h>

#include <QString>

namespace PVCore {
class LibKernelDecl PVPythonInitializer
{
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
private:
	PyThreadState* mainThreadState;

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

#endif //PVPYTHON_H
