/**
 * \file PVIntegerDefaultSortingFunc.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVUnicodeString.h>
#include "PVIntegerDefaultSortingFunc.h"

Picviz::PVIntegerDefaultSortingFunc::PVIntegerDefaultSortingFunc(PVCore::PVArgumentList const& l):
	PVSortingFunc(l)
{
}

DEFAULT_ARGS_FUNC(Picviz::PVIntegerDefaultSortingFunc)
{
	PVCore::PVArgumentList args;
	return args;
}

Picviz::PVSortingFunc_f Picviz::PVIntegerDefaultSortingFunc::f()
{
	return &comp_asc;
}

Picviz::PVSortingFunc_fequals Picviz::PVIntegerDefaultSortingFunc::f_equals()
{
	return &equals_asc;
}

Picviz::PVSortingFunc_fless Picviz::PVIntegerDefaultSortingFunc::f_less()
{
	return &less_asc;
}

bool Picviz::PVIntegerDefaultSortingFunc::less_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	QString s;
	qlonglong f1 = s1.get_qstr(s).toLongLong();
	qlonglong f2 = s2.get_qstr(s).toLongLong();
	return f1 < f2;
}

bool Picviz::PVIntegerDefaultSortingFunc::equals_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1 == s2;
}

int Picviz::PVIntegerDefaultSortingFunc::comp_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	QString s;
	qlonglong f1 = s1.get_qstr(s).toLongLong();
	qlonglong f2 = s2.get_qstr(s).toLongLong();
	if (f1 != f2) {
		return (f1 < f2) ? -1 : 1;
	}
	return 0;
}
