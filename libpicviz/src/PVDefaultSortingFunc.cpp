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

Picviz::PVSortingFunc_fless Picviz::PVDefaultSortingFunc::f_less()
{
	return &less_case_asc;
}

bool Picviz::PVDefaultSortingFunc::less_case_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
{
	return s1.compare(s2) < 0;
}

bool Picviz::PVDefaultSortingFunc::less_nocase_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2)
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
