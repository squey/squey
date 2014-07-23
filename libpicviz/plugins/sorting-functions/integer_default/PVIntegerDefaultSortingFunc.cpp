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

Picviz::PVSortingFunc_flesser Picviz::PVIntegerDefaultSortingFunc::f_lesser()
{
	return &lesser_asc;
}

Picviz::PVQtSortingFunc_f Picviz::PVIntegerDefaultSortingFunc::qt_f()
{
	return &qt_comp_asc;
}

Picviz::PVQtSortingFunc_fequals Picviz::PVIntegerDefaultSortingFunc::qt_f_equals()
{
	return &qt_equals_asc;
}

Picviz::PVQtSortingFunc_flesser Picviz::PVIntegerDefaultSortingFunc::qt_f_lesser()
{
	return &qt_lesser_asc;
}

bool Picviz::PVIntegerDefaultSortingFunc::lesser_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	long f1 = s1.to_long(10);
	long f2 = s2.to_long(10);
	return f1 < f2;
}

bool Picviz::PVIntegerDefaultSortingFunc::equals_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1 == s2;
}

int Picviz::PVIntegerDefaultSortingFunc::comp_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	long f1 = s1.to_long(10);
	long f2 = s2.to_long(10);
	if (f1 != f2) {
		return (f1 < f2) ? -1 : 1;
	}
	return 0;
}

bool Picviz::PVIntegerDefaultSortingFunc::qt_lesser_asc(QString const& s1, QString const& s2)
{
	return s1.toLong() < s2.toLong();
}

bool Picviz::PVIntegerDefaultSortingFunc::qt_equals_asc(QString const& s1, QString const& s2)
{
	// 10.0 == 1.0e1
	return s1.toLong() == s2.toLong();
}

int Picviz::PVIntegerDefaultSortingFunc::qt_comp_asc(QString const& s1, QString const& s2)
{
	long f1 = s1.toLong();
	long f2 = s2.toLong();
	if (f1 != f2) {
		return (f1 < f2) ? -1 : 1;
	}
	return 0;
}
