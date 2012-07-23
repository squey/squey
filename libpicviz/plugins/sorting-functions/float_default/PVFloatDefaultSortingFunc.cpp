/**
 * \file PVFloatDefaultSortingFunc.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVUnicodeString.h>
#include "PVFloatDefaultSortingFunc.h"

Picviz::PVFloatDefaultSortingFunc::PVFloatDefaultSortingFunc(PVCore::PVArgumentList const& l):
	PVSortingFunc(l)
{
}

DEFAULT_ARGS_FUNC(Picviz::PVFloatDefaultSortingFunc)
{
	PVCore::PVArgumentList args;
	return args;
}

Picviz::PVSortingFunc_f Picviz::PVFloatDefaultSortingFunc::f()
{
	return &comp_asc;
}

Picviz::PVSortingFunc_fequals Picviz::PVFloatDefaultSortingFunc::f_equals()
{
	return &equals_asc;
}

Picviz::PVSortingFunc_fless Picviz::PVFloatDefaultSortingFunc::f_less()
{
	return &less_asc;
}

bool Picviz::PVFloatDefaultSortingFunc::less_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	QString s;
	float f1 = s1.get_qstr(s).toFloat();
	float f2 = s2.get_qstr(s).toFloat();
	return f1 < f2;
}

bool Picviz::PVFloatDefaultSortingFunc::equals_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1 == s2;
}

int Picviz::PVFloatDefaultSortingFunc::comp_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	QString s;
	float f1 = s1.get_qstr(s).toFloat();
	float f2 = s2.get_qstr(s).toFloat();
	if (f1 != f2) {
		return (f1 < f2) ? -1 : 1;
	}
	return 0;
}
