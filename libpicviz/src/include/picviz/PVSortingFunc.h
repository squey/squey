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
	typedef bool(*unicode_sorting_less_func)(PVCore::PVUnicodeString const&, PVCore::PVUnicodeString const&);
	typedef bool(*unicode_sorting_equals_func)(PVCore::PVUnicodeString const&, PVCore::PVUnicodeString const&);
	typedef int(*unicode_sorting_comp_func)(PVCore::PVUnicodeString const&, PVCore::PVUnicodeString const&);
}

class LibPicvizDecl PVSortingFunc : public PVCore::PVFunctionArgs<__impl::unicode_sorting_comp_func>, public PVCore::PVRegistrableClass<PVSortingFunc>
{
public:
	typedef boost::shared_ptr<PVSortingFunc> p_type;
	typedef __impl::unicode_sorting_comp_func f_type;
	typedef __impl::unicode_sorting_less_func fless_type;
	typedef __impl::unicode_sorting_equals_func fequals_type;

public:
	PVSortingFunc(PVCore::PVArgumentList const& l = PVSortingFunc::default_args());

public:
	virtual f_type f() = 0;
	virtual fless_type f_less() = 0;
	virtual fequals_type f_equals() = 0;

	CLASS_FUNC_ARGS_NOPARAM()
};

typedef PVSortingFunc::p_type PVSortingFunc_p;
typedef PVSortingFunc::f_type PVSortingFunc_f;
typedef PVSortingFunc::fless_type PVSortingFunc_fless;
typedef PVSortingFunc::fequals_type PVSortingFunc_fequals;

}

#endif
