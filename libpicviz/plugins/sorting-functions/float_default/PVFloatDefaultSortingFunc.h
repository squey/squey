/**
 * \file PVFloatDefaultSortingFunc.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVFLOATDEFAULTSORTINGFUNC_H
#define PICVIZ_PVFLOATDEFAULTSORTINGFUNC_H

#include <pvkernel/core/general.h>
#include <picviz/PVSortingFunc.h>

namespace Picviz {

class LibPicvizDecl PVFloatDefaultSortingFunc : public PVSortingFunc
{
public:
	PVFloatDefaultSortingFunc(PVCore::PVArgumentList const& l = PVFloatDefaultSortingFunc::default_args());
public:
	virtual f_type f();
	virtual fequals_type f_equals();
	virtual fless_type f_less();

private:
	static bool less_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);
	static bool equals_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);
	static int comp_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);

	CLASS_REGISTRABLE(PVFloatDefaultSortingFunc)
	CLASS_FUNC_ARGS_PARAM(PVFloatDefaultSortingFunc)
};

}

#endif
