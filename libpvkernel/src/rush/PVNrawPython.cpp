#include <pvkernel/core/PVPython.h>
#include <pvkernel/core/PVPythonClassDecl.h>
#include <pvkernel/rush/PVNrawPython.h>

PYTHON_EXPOSE_IMPL(PVRush::PVNrawPython)
{
	boost::python::class_<PVNrawPython>("PVNraw")
		.def("at", &PVNrawPython::at)
		.def("at_alias", &PVNrawPython::at_alias)
		.def("set_value", &PVNrawPython::set_value)
		.def("get_number_rows", &PVNrawPython::get_number_rows)
		.def("get_number_cols", &PVNrawPython::get_number_cols)
	;
}
