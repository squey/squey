/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVPython.h>
#include <pvkernel/core/PVPythonClassDecl.h>
#include <pvkernel/rush/PVNrawPython.h>

void PVRush::PVNrawPython::set_value(PVRow i, PVCol j, PVCore::PVUnicodeString const& str)
{
	if (!_nraw) {
		return;
	}
	return _nraw->set_value(i, j, str);
}

std::wstring PVRush::PVNrawPython::at(PVRow i, PVCol j)
{
	if (!_nraw) {
		return L"";
	}
	return _nraw->at(i, j).toStdWString();
}

PVRow PVRush::PVNrawPython::get_number_rows() const
{
	return _nraw->get_number_rows();
}

PVCol PVRush::PVNrawPython::get_number_cols() const
{
	return _nraw->get_number_cols();
}

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
