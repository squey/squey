/**
 * \file PVIPv4DefaultSortingFunc.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVIPV4DEFAULTSORTINGFUNC_H
#define PICVIZ_PVIPV4DEFAULTSORTINGFUNC_H

#include <pvkernel/core/general.h>
#include <picviz/PVSortingFunc.h>

namespace Picviz {

class LibPicvizDecl PVIPv4DefaultSortingFunc : public PVSortingFunc
{
public:
	PVIPv4DefaultSortingFunc(PVCore::PVArgumentList const& l = PVIPv4DefaultSortingFunc::default_args());
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

	CLASS_REGISTRABLE(PVIPv4DefaultSortingFunc)
	CLASS_FUNC_ARGS_PARAM(PVIPv4DefaultSortingFunc)
};

}

#endif // PICVIZ_PVIPV4DEFAULTSORTINGFUNC_H
