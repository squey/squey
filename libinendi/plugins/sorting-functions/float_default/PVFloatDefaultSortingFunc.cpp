/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVUnicodeString.h>
#include "PVFloatDefaultSortingFunc.h"

Inendi::PVFloatDefaultSortingFunc::PVFloatDefaultSortingFunc(PVCore::PVArgumentList const& l):
	PVSortingFunc(l)
{
}

DEFAULT_ARGS_FUNC(Inendi::PVFloatDefaultSortingFunc)
{
	PVCore::PVArgumentList args;
	return args;
}

Inendi::PVSortingFunc_f Inendi::PVFloatDefaultSortingFunc::f()
{
	return &comp_asc;
}

Inendi::PVSortingFunc_fequals Inendi::PVFloatDefaultSortingFunc::f_equals()
{
	return &equals_asc;
}

Inendi::PVSortingFunc_flesser Inendi::PVFloatDefaultSortingFunc::f_lesser()
{
	return &lesser_asc;
}

Inendi::PVQtSortingFunc_f Inendi::PVFloatDefaultSortingFunc::qt_f()
{
	return &qt_comp_asc;
}

Inendi::PVQtSortingFunc_fequals Inendi::PVFloatDefaultSortingFunc::qt_f_equals()
{
	return &qt_equals_asc;
}

Inendi::PVQtSortingFunc_flesser Inendi::PVFloatDefaultSortingFunc::qt_f_lesser()
{
	return &qt_lesser_asc;
}

bool Inendi::PVFloatDefaultSortingFunc::lesser_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	QString s;
	float f1 = s1.get_qstr(s).toFloat();
	float f2 = s2.get_qstr(s).toFloat();
	return f1 < f2;
}

bool Inendi::PVFloatDefaultSortingFunc::equals_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1 == s2;
}

int Inendi::PVFloatDefaultSortingFunc::comp_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	QString s;
	float f1 = s1.get_qstr(s).toFloat();
	float f2 = s2.get_qstr(s).toFloat();
	if (f1 != f2) {
		return (f1 < f2) ? -1 : 1;
	}
	return 0;
}

bool Inendi::PVFloatDefaultSortingFunc::qt_lesser_asc(QString const& s1, QString const& s2)
{
	return s1.toFloat() < s2.toFloat();
}

bool Inendi::PVFloatDefaultSortingFunc::qt_equals_asc(QString const& s1, QString const& s2)
{
	// 10.0 == 1.0e1
	return s1.toFloat() == s2.toFloat();
}


int Inendi::PVFloatDefaultSortingFunc::qt_comp_asc(QString const& s1, QString const& s2)
{
	float f1 = s1.toFloat();
	float f2 = s2.toFloat();
	if (f1 != f2) {
		return (f1 < f2) ? -1 : 1;
	}
	return 0;
}
