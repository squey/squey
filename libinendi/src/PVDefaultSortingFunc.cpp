/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/PVUnicodeString.h>
#include <pvkernel/core/PVEnumType.h>
#include <inendi/PVDefaultSortingFunc.h>

Inendi::PVDefaultSortingFunc::PVDefaultSortingFunc(PVCore::PVArgumentList const& l):
	PVSortingFunc(l)
{
}

DEFAULT_ARGS_FUNC(Inendi::PVDefaultSortingFunc)
{
	PVCore::PVArgumentList args;
	args[PVCore::PVArgumentKey("sorting-case", "Case match")].setValue(PVCore::PVEnumType(QStringList() << "Does not match case" << "Match case", 0));
	return args;
}

Inendi::PVSortingFunc_f Inendi::PVDefaultSortingFunc::f()
{
	return &comp_case_asc;
}

Inendi::PVSortingFunc_fequals Inendi::PVDefaultSortingFunc::f_equals()
{
	return &equals_case_asc;
}

Inendi::PVSortingFunc_flesser Inendi::PVDefaultSortingFunc::f_lesser()
{
	return &lesser_case_asc;
}

Inendi::PVQtSortingFunc_f Inendi::PVDefaultSortingFunc::qt_f()
{
	return &qt_comp_case_asc;
}

Inendi::PVQtSortingFunc_fequals Inendi::PVDefaultSortingFunc::qt_f_equals()
{
	return &qt_equals_case_asc;
}

Inendi::PVQtSortingFunc_flesser Inendi::PVDefaultSortingFunc::qt_f_lesser()
{
	return &qt_lesser_case_asc;
}

bool Inendi::PVDefaultSortingFunc::lesser_case_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1.compare(s2) < 0;
}

bool Inendi::PVDefaultSortingFunc::lesser_nocase_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1.compareNoCase(s2) < 0;
}

bool Inendi::PVDefaultSortingFunc::equals_case_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1 == s2;
}

bool Inendi::PVDefaultSortingFunc::equals_nocase_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1.compareNoCase(s2) == 0;
}

int Inendi::PVDefaultSortingFunc::comp_case_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1.compare(s2);
}

int Inendi::PVDefaultSortingFunc::comp_nocase_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1.compareNoCase(s2);
}

bool Inendi::PVDefaultSortingFunc::qt_lesser_case_asc(QString const& s1, QString const& s2)
{
	return s1.compare(s2, Qt::CaseSensitive) < 0;
}

bool Inendi::PVDefaultSortingFunc::qt_equals_case_asc(QString const& s1, QString const& s2)
{
	return s1.compare(s2, Qt::CaseSensitive) == 0;
}

int Inendi::PVDefaultSortingFunc::qt_comp_case_asc(QString const& s1, QString const& s2)
{
	return s1.compare(s2, Qt::CaseSensitive);
}
