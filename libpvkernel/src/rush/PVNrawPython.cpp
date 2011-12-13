#include <pvkernel/core/PVPython.h>
#include <pvkernel/rush/PVNrawPython.h>

PYTHON_EXPOSE_IMPL(PVRush::PVNrawPython)
{
	PVLOG_INFO("register PVNrawPython\n");
	boost::python::class_<PVNrawPython>("PVNraw")
		.def("at", &PVNrawPython::at)
		.def("at_alias", &PVNrawPython::at_alias, boost::python::return_internal_reference<>())
		.def("set_value", &PVNrawPython::set_value)
	;
}
