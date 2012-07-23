/**
 * \file PVIntegerDefaultSortingFunc.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVINTEGERDEFAULTSORTINGFUNC_H
#define PICVIZ_PVINTEGERDEFAULTSORTINGFUNC_H

#include <pvkernel/core/general.h>
#include <picviz/PVSortingFunc.h>

namespace Picviz {

class LibPicvizDecl PVIntegerDefaultSortingFunc : public PVSortingFunc
{
public:
	PVIntegerDefaultSortingFunc(PVCore::PVArgumentList const& l = PVIntegerDefaultSortingFunc::default_args());
public:
	virtual f_type f();
	virtual fequals_type f_equals();
	virtual fless_type f_less();

private:
	static bool less_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);
	static bool equals_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);
	static int comp_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);

	CLASS_REGISTRABLE(PVIntegerDefaultSortingFunc)
	CLASS_FUNC_ARGS_PARAM(PVIntegerDefaultSortingFunc)
};

}

#endif
