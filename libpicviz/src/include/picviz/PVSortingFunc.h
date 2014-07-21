/**
 * \file PVSortingFunc.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVSORTINGFUNC_H
#define PICVIZ_PVSORTINGFUNC_H

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVFunctionArgs.h>
#include <pvkernel/core/PVRegistrableClass.h>

#include <boost/shared_ptr.hpp>

namespace PVCore {
class PVUnicodeString;
}

namespace Picviz {

namespace __impl {
	typedef bool(*unicode_sorting_lesser_func)(PVCore::PVUnicodeString const&, PVCore::PVUnicodeString const&);
	typedef bool(*unicode_sorting_equals_func)(PVCore::PVUnicodeString const&, PVCore::PVUnicodeString const&);
	typedef int(*unicode_sorting_comp_func)(PVCore::PVUnicodeString const&, PVCore::PVUnicodeString const&);

	typedef bool(*qt_sorting_lesser_func)(QString const&, QString const&);
	typedef bool(*qt_sorting_equals_func)(QString const&, QString const&);
	typedef int(*qt_sorting_comp_func)(QString const&, QString const&);
}

class LibPicvizDecl PVSortingFunc : public PVCore::PVFunctionArgs<__impl::unicode_sorting_comp_func>, public PVCore::PVRegistrableClass<PVSortingFunc>
{
public:
	typedef boost::shared_ptr<PVSortingFunc> p_type;
	typedef __impl::unicode_sorting_comp_func f_type;
	typedef __impl::unicode_sorting_lesser_func flesser_type;
	typedef __impl::unicode_sorting_equals_func fequals_type;

	typedef __impl::qt_sorting_comp_func qt_f_type;
	typedef __impl::qt_sorting_lesser_func qt_flesser_type;
	typedef __impl::qt_sorting_equals_func qt_fequals_type;

public:
	PVSortingFunc(PVCore::PVArgumentList const& l = PVSortingFunc::default_args());

public:
	virtual f_type f() = 0;
	virtual flesser_type f_lesser() = 0;
	virtual fequals_type f_equals() = 0;

	virtual qt_f_type qt_f() = 0;
	virtual qt_flesser_type qt_f_lesser() = 0;
	virtual qt_fequals_type qt_f_equals() = 0;

	CLASS_FUNC_ARGS_NOPARAM()
};

typedef PVSortingFunc::p_type PVSortingFunc_p;
typedef PVSortingFunc::f_type PVSortingFunc_f;
typedef PVSortingFunc::flesser_type PVSortingFunc_flesser;
typedef PVSortingFunc::fequals_type PVSortingFunc_fequals;

typedef PVSortingFunc::qt_f_type PVQtSortingFunc_f;
typedef PVSortingFunc::qt_flesser_type PVQtSortingFunc_flesser;
typedef PVSortingFunc::qt_fequals_type PVQtSortingFunc_fequals;

}

#endif
