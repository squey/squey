/**
 * \file PVDefaultSortingFunc.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/PVUnicodeString.h>
#include <pvkernel/core/PVEnumType.h>
#include <picviz/PVDefaultSortingFunc.h>

Picviz::PVDefaultSortingFunc::PVDefaultSortingFunc(PVCore::PVArgumentList const& l):
	PVSortingFunc(l)
{
}

DEFAULT_ARGS_FUNC(Picviz::PVDefaultSortingFunc)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("sorting-case", "Case match")].setValue(PVCore::PVEnumType(QStringList() << "Does not match case" << "Match case", 0));
	return args;
}

Picviz::PVSortingFunc_f Picviz::PVDefaultSortingFunc::f()
{
	return &comp_case_asc;
}

Picviz::PVSortingFunc_fequals Picviz::PVDefaultSortingFunc::f_equals()
{
	return &equals_case_asc;
}

Picviz::PVSortingFunc_flesser Picviz::PVDefaultSortingFunc::f_lesser()
{
	return &lesser_case_asc;
}

Picviz::PVQtSortingFunc_f Picviz::PVDefaultSortingFunc::qt_f()
{
	return &qt_comp_case_asc;
}

Picviz::PVQtSortingFunc_fequals Picviz::PVDefaultSortingFunc::qt_f_equals()
{
	return &qt_equals_case_asc;
}

Picviz::PVQtSortingFunc_flesser Picviz::PVDefaultSortingFunc::qt_f_lesser()
{
	return &qt_lesser_case_asc;
}

bool Picviz::PVDefaultSortingFunc::lesser_case_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1.compare(s2) < 0;
}

bool Picviz::PVDefaultSortingFunc::lesser_nocase_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1.compareNoCase(s2) < 0;
}

bool Picviz::PVDefaultSortingFunc::equals_case_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1 == s2;
}

bool Picviz::PVDefaultSortingFunc::equals_nocase_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1.compareNoCase(s2) == 0;
}

int Picviz::PVDefaultSortingFunc::comp_case_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1.compare(s2);
}

int Picviz::PVDefaultSortingFunc::comp_nocase_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1.compareNoCase(s2);
}

bool Picviz::PVDefaultSortingFunc::qt_lesser_case_asc(QString const& s1, QString const& s2)
{
	return s1.compare(s2, Qt::CaseSensitive) < 0;
}

bool Picviz::PVDefaultSortingFunc::qt_equals_case_asc(QString const& s1, QString const& s2)
{
	return s1.compare(s2, Qt::CaseSensitive) == 0;
}

int Picviz::PVDefaultSortingFunc::qt_comp_case_asc(QString const& s1, QString const& s2)
{
	return s1.compare(s2, Qt::CaseSensitive);
}
