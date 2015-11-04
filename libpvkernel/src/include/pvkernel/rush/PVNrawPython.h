/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVPYTHONNRAW_H
#define PVRUSH_PVPYTHONNRAW_H

#include <pvkernel/core/PVPythonClassDecl.h>
#include <pvkernel/core/general.h>

#include <pvkernel/rush/PVNraw.h>

namespace PVRush {

class PVNrawPython
{
public:
	PVNrawPython():
		_nraw(NULL)
	{ } 

	PVNrawPython(PVRush::PVNraw* nraw):
		_nraw(nraw)
	{ } 

public:
	PVCore::PVUnicodeString at_alias(PVRow i, PVCol j);
	void set_value(PVRow i, PVCol j, PVCore::PVUnicodeString const& str); 
	std::wstring at(PVRow i, PVCol j);
	PVRow get_number_rows() const;
	PVCol get_number_cols() const;

private:
	PVRush::PVNraw* _nraw;

	PYTHON_EXPOSE()
};

}

#endif
