/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVUnicodeString.h>
#include "PVIntegerDefaultSortingFunc.h"

Inendi::PVIntegerDefaultSortingFunc::PVIntegerDefaultSortingFunc(PVCore::PVArgumentList const& l):
	PVSortingFunc(l)
{
}

DEFAULT_ARGS_FUNC(Inendi::PVIntegerDefaultSortingFunc)
{
	PVCore::PVArgumentList args;
	return args;
}

Inendi::PVSortingFunc_f Inendi::PVIntegerDefaultSortingFunc::f()
{
	return &comp_asc;
}

Inendi::PVSortingFunc_fequals Inendi::PVIntegerDefaultSortingFunc::f_equals()
{
	return &equals_asc;
}

Inendi::PVSortingFunc_flesser Inendi::PVIntegerDefaultSortingFunc::f_lesser()
{
	return &lesser_asc;
}

Inendi::PVQtSortingFunc_f Inendi::PVIntegerDefaultSortingFunc::qt_f()
{
	return &qt_comp_asc;
}

Inendi::PVQtSortingFunc_fequals Inendi::PVIntegerDefaultSortingFunc::qt_f_equals()
{
	return &qt_equals_asc;
}

Inendi::PVQtSortingFunc_flesser Inendi::PVIntegerDefaultSortingFunc::qt_f_lesser()
{
	return &qt_lesser_asc;
}

bool Inendi::PVIntegerDefaultSortingFunc::lesser_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	QString s;
	long f1 = s1.get_qstr(s).toLong();
	long f2 = s2.get_qstr(s).toLong();
	return f1 < f2;
}

bool Inendi::PVIntegerDefaultSortingFunc::equals_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1 == s2;
}

int Inendi::PVIntegerDefaultSortingFunc::comp_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	QString s;
	long f1 = s1.get_qstr(s).toLong();
	long f2 = s2.get_qstr(s).toLong();
	if (f1 != f2) {
		return (f1 < f2) ? -1 : 1;
	}
	return 0;
}

bool Inendi::PVIntegerDefaultSortingFunc::qt_lesser_asc(QString const& s1, QString const& s2)
{
	return s1.toLong() < s2.toLong();
}

bool Inendi::PVIntegerDefaultSortingFunc::qt_equals_asc(QString const& s1, QString const& s2)
{
	// 10.0 == 1.0e1
	return s1.toLong() == s2.toLong();
}

int Inendi::PVIntegerDefaultSortingFunc::qt_comp_asc(QString const& s1, QString const& s2)
{
	long f1 = s1.toLong();
	long f2 = s2.toLong();
	if (f1 != f2) {
		return (f1 < f2) ? -1 : 1;
	}
	return 0;
}
