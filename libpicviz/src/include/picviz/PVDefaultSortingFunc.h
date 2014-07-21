/**
 * \file PVDefaultSortingFunc.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVDEFAULTSORTINGFUNC_H
#define PICVIZ_PVDEFAULTSORTINGFUNC_H

#include <pvkernel/core/general.h>
#include <picviz/PVSortingFunc.h>

namespace Picviz {

class LibPicvizDecl PVDefaultSortingFunc : public PVSortingFunc
{
public:
	PVDefaultSortingFunc(PVCore::PVArgumentList const& l = PVDefaultSortingFunc::default_args());
public:
	virtual f_type f();
	virtual fequals_type f_equals();
	virtual flesser_type f_lesser();

	virtual qt_f_type qt_f();
	virtual qt_flesser_type qt_f_lesser();
	virtual qt_fequals_type qt_f_equals();

private:
	static bool lesser_case_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);
	static bool lesser_nocase_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);

	static bool equals_case_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);
	static bool equals_nocase_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);

	static int comp_case_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);
	static int comp_nocase_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);

	static bool qt_lesser_case_asc(QString const& s1, QString const& s2);
	static bool qt_equals_case_asc(QString const& s1, QString const& s2);
	static int qt_comp_case_asc(QString const& s1, QString const& s2);

	CLASS_REGISTRABLE(PVDefaultSortingFunc)
	CLASS_FUNC_ARGS_PARAM(PVDefaultSortingFunc)
};

}

#endif
