/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVFLOATDEFAULTSORTINGFUNC_H
#define INENDI_PVFLOATDEFAULTSORTINGFUNC_H

#include <pvkernel/core/general.h>
#include <inendi/PVSortingFunc.h>

namespace Inendi {

class PVFloatDefaultSortingFunc : public PVSortingFunc
{
public:
	PVFloatDefaultSortingFunc(PVCore::PVArgumentList const& l = PVFloatDefaultSortingFunc::default_args());
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

	CLASS_REGISTRABLE(PVFloatDefaultSortingFunc)
	CLASS_FUNC_ARGS_PARAM(PVFloatDefaultSortingFunc)
};

}

#endif