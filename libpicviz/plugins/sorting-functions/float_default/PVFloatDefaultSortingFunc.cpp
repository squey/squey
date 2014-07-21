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

Picviz::PVSortingFunc_flesser Picviz::PVFloatDefaultSortingFunc::f_lesser()
{
	return &lesser_asc;
}

Picviz::PVQtSortingFunc_f Picviz::PVFloatDefaultSortingFunc::qt_f()
{
	return &qt_comp_asc;
}

Picviz::PVQtSortingFunc_fequals Picviz::PVFloatDefaultSortingFunc::qt_f_equals()
{
	return &qt_equals_asc;
}

Picviz::PVQtSortingFunc_flesser Picviz::PVFloatDefaultSortingFunc::qt_f_lesser()
{
	return &qt_lesser_asc;
}

bool Picviz::PVFloatDefaultSortingFunc::lesser_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
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

bool Picviz::PVFloatDefaultSortingFunc::qt_lesser_asc(QString const& s1, QString const& s2)
{
	return s1.toFloat() < s2.toFloat();
}

bool Picviz::PVFloatDefaultSortingFunc::qt_equals_asc(QString const& s1, QString const& s2)
{
	// 10.0 == 1.0e1
	return s1.toFloat() == s2.toFloat();
}


int Picviz::PVFloatDefaultSortingFunc::qt_comp_asc(QString const& s1, QString const& s2)
{
	float f1 = s1.toFloat();
	float f2 = s2.toFloat();
	if (f1 != f2) {
		return (f1 < f2) ? -1 : 1;
	}
	return 0;
}
