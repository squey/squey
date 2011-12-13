#ifndef PVRUSH_PVPYTHONNRAW_H
#define PVRUSH_PVPYTHONNRAW_H

#include <pvkernel/core/PVPythonClassDecl.h>
#include <pvkernel/core/general.h>

#include <pvkernel/rush/PVNraw.h>

namespace PVRush {

class LibKernelDecl PVNrawPython
{
public:
	PVNrawPython():
		_nraw(NULL)
	{ } 

	PVNrawPython(PVRush::PVNraw* nraw):
		_nraw(nraw)
	{ } 

public:
	PVCore::PVUnicodeString const& at_alias(PVRow i, PVCol j)
	{   
		return _nraw->at_unistr(i, j); 
	}   

	void set_value(PVRow i, PVCol j, PVCore::PVUnicodeString const& str)
	{   
		if (!_nraw) {
			return;
		}   
		return _nraw->set_value(i, j, str);
	}   

	std::wstring at(PVRow i, PVCol j)
	{   
		if (!_nraw) {
			return L"";
		}   
		return _nraw->at(i, j).toStdWString();
	}   
private:
	PVRush::PVNraw* _nraw;

	PYTHON_EXPOSE()
};

}

#endif
