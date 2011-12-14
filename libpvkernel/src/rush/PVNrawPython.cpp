#include <pvkernel/core/PVPython.h>
#include <pvkernel/rush/PVNrawPython.h>

PVCore::PVUnicodeString const& PVRush::PVNrawPython::at_alias(PVRow i, PVCol j)
{
	return _nraw->at_unistr(i, j);
}
void PVRush::PVNrawPython::set_value(PVRow i, PVCol j, PVCore::PVUnicodeString const& str)
{
	if (!_nraw) {
		return;
	}
	return _nraw->set_value(i, j, str);
}

/*std::wstring PVRush::PVNrawPython::at(PVRow i, PVCol j)
{
	if (!_nraw) {
		return L"";
	}
	return _nraw->at(i, j).toStdWString();
}*/

PYTHON_EXPOSE_IMPL(PVRush::PVNrawPython)
{
	PVLOG_INFO("register PVNrawPython\n");
	boost::python::class_<PVNrawPython>("PVNraw")
		//.def("at", &PVNrawPython::at)
		.def("at_alias", &PVNrawPython::at_alias, boost::python::return_internal_reference<>())
		.def("set_value", &PVNrawPython::set_value)
	;
}