/**
 * \file PVIntegerDefaultSortingFunc.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVINTEGERDEFAULTSORTINGFUNC_H
#define PICVIZ_PVINTEGERDEFAULTSORTINGFUNC_H

#include <pvkernel/core/general.h>
#include <picviz/PVSortingFunc.h>

namespace Picviz {

class LibPicvizDecl PVIntegerDefaultSortingFunc : public PVSortingFunc
{
public:
	PVIntegerDefaultSortingFunc(PVCore::PVArgumentList const& l = PVIntegerDefaultSortingFunc::default_args());
public:
	virtual f_type f();
	virtual fequals_type f_equals();
	virtual flesser_type f_lesser();

	virtual qt_f_type qt_f();
	virtual qt_flesser_type qt_f_lesser();
	virtual qt_fequals_type qt_f_equals();

private:
	static bool lesser_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);
	static bool equals_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);
	static int comp_asc(PVCore::PVUnicodeString const& s1, PVCore::PVUnicodeString const& s2);

	static bool qt_lesser_asc(QString const& s1, QString const& s2);
	static bool qt_equals_asc(QString const& s1, QString const& s2);
	static int qt_comp_asc(QString const& s1, QString const& s2);

	CLASS_REGISTRABLE(PVIntegerDefaultSortingFunc)
	CLASS_FUNC_ARGS_PARAM(PVIntegerDefaultSortingFunc)
};

}

#endif
